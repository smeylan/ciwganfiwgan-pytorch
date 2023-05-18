import argparse
import os
import re
import itertools as it
import sys

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from scipy.io.wavfile import read, write
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from infowavegan import WaveGANGenerator, WaveGANDiscriminator, WaveGANQNetwork
from utils import get_continuation_fname
import tempfile
import scipy
import uuid
import copy
import string
from Levenshtein import distance as lev
import glob 

# For Whisper 
import faster_whisper
import string
import signal
from Levenshtein import distance as lev
from contextlib import contextmanager

torch.autograd.set_detect_anomaly(True)

# For Nemo Q2
# import multiprocessing
# import nemo.collections.asr as nemo_asr
# import pyctcdecode
# import kenlm


class AudioDataSet:
    def __init__(self, datadir, slice_len, NUM_CATEG, timit_words):
        print("Loading data")
        dir = os.listdir(datadir)
        x = np.zeros((len(dir), 1, slice_len))
        y = np.zeros((len(dir), NUM_CATEG+1)) # +1 for UNK

        i = 0
        for file in tqdm(dir):
            audio = read(os.path.join(datadir, file))[1]
            if audio.shape[0] < slice_len:
                audio = np.pad(audio, (0, slice_len - audio.shape[0]))
            audio = audio[:slice_len]

            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32767
            elif audio.dtype == np.float32:
                pass
            else:
                raise NotImplementedError('Scipy cannot process atypical WAV files.')
            audio /= np.max(np.abs(audio))
            x[i, 0, :] = audio

            # extract the label
            word = file.split('_')[0]
            j = timit_words.index(word)
            y[i, j] = 1            
            i += 1

        self.len = len(x)
        self.audio = torch.from_numpy(np.array(x, dtype=np.float32))
        self.labels = torch.from_numpy(np.array(y, dtype=np.float32))

    def __getitem__(self, index):
        return ((self.audio[index], self.labels[index]))

    def __len__(self):
        return self.len


def gradient_penalty(G, D, real, fake, epsilon):
    x_hat = epsilon * real + (1 - epsilon) * fake
    scores = D(x_hat)
    grad = torch.autograd.grad(
        outputs=scores,
        inputs=x_hat,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True
    )[0]
    grad_norm = grad.view(grad.shape[0], -1).norm(p=2, dim=1)  # norm along each batch
    penalty = ((grad_norm - 1) ** 2).unsqueeze(1)
    return penalty

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    #signal.alarm(seconds)
    signal.setitimer(signal.ITIMER_REAL,seconds) 
    try:
        yield
    finally:
        signal.alarm(0)

def fast_whisper_recognize_wav(filename, whisper_model, timit_words, vocab, Q2_TIMEOUT):    
    
    ip = 'A person on the radio just said one of the following words: "'+'", "'.join(timit_words)+'." The word was "'
    
    try:
        with time_limit(Q2_TIMEOUT):
            segments, info = whisper_model.transcribe(filename, language="en", initial_prompt = ip)
            transcription = [x for x in segments][0]    
            timeout = False

    except TimeoutException as e:
        timeout = True
    
    if not timeout:
        best_guess_of_string = transcription.text.lower().strip().replace(' ','')
        best_guess_of_string = best_guess_of_string.translate(str.maketrans('', '', string.punctuation))
    
        #likelihoods: compute levenshtein distance to all n
    
        distances = np.array([lev(x,best_guess_of_string) for x in vocab.word])
    
        alpha = 4
        likelihoods = np.exp(-1. * alpha * distances)
        
        # priors: 
        # priors = np.ones(len(timit_words)) * 1./len(timit_words)
        priors = vocab.probability
    
        unnormalized = priors * likelihoods
        posteriors = unnormalized / np.sum(unnormalized)
            
        candidates = pd.DataFrame({"hypothesis":vocab.word,"prob":posteriors}).sort_values(by=['prob'], ascending=False)
        candidates.to_csv(filename.replace('.wav','_candidates.csv'))
        
        limited_probs = candidates.loc[candidates.hypothesis.isin(timit_words)]
        remainder_prob =  np.sum(candidates.loc[~candidates.hypothesis.isin(timit_words)].prob)
        remainder_row = pd.DataFrame({'hypothesis':["UNK"], "logit_score":[np.nan], 
                                  "combined_score":[np.nan], "prob":[remainder_prob]})

        simplified = pd.concat([limited_probs, remainder_row])
        simplified.to_csv(filename.replace('.wav','_simplified.csv'))
    
        # make sure all timit words are present
        simplified = pd.DataFrame({'hypothesis':timit_words}).merge(simplified[['hypothesis', 'prob']], how="left") 
        simplified = simplified.fillna(0)         

        rdf = pd.DataFrame.from_records([{
            'candidates': candidates.iloc[0:10],
            'simplified': simplified,
            'prob': simplified.prob.values,        
            'decoding_prob':np.exp(transcription.avg_logprob),
            'no_speech_prob': transcription.no_speech_prob,
            'best_guess_of_string': best_guess_of_string,
            'filename':filename,
            'unk_prob': simplified.prob.values[-1],
            "timeout": timeout
            
        }], index=[0])
        
    else:
        dummy_prob = np.zeros(len(timit_words))
        dummy_prob[-1] = 1 
        rdf = pd.DataFrame.from_records([{
            'candidates': None,
            'simplified': None,
            'prob': dummy_prob,        
            'decoding_prob':None,
            'no_speech_prob': None,
            'best_guess_of_string': None,
            'filename':filename,
            'unk_prob': None,
            "timeout": timeout
            
        }], index=[0])
        
        
    return(rdf)

def whisper_recognize_wavs(filenames, fast_whisper_model, vocab, timit_words):        
    
    test_files = pd.DataFrame({"filename":filenames}) 
    
    sents = []    
    for filename in tqdm(test_files.filename):
        sents.append(fast_whisper_recognize_wav(filename, fast_whisper_model, timit_words, vocab, Q2_TIMEOUT))

    results= pd.concat(sents)
    test_files = test_files.merge(results)    
    test_files['index'] = range(test_files.shape[0])

    indices_of_recognized_words = test_files.loc[(test_files.decoding_prob > MIN_DECODING_PROB) & (test_files.no_speech_prob < MAX_NOSPEECH_PROB) & (test_files.unk_prob < MAX_UNK_PROB) &  ~test_files.timeout]['index'].values

    if len(indices_of_recognized_words) == 0:
        probabilities = None
    else:
        probabilities = np.vstack(test_files.prob)    

    return(indices_of_recognized_words, probabilities)


def Q2_whisper(G_z, labels, fast_whisper_model, timit_words, vocab, epoch, batch_size, write_only=False):
    # returns probabilities and a set of indices; takes a smaller number of arguments
    files_for_asr = []
    epoch_path = os.path.join("temp",str(epoch))
    if not os.path.exists(epoch_path):
        os.makedirs(epoch_path)    

    labels_local = labels.cpu().detach().numpy()
    # for each file in the batch, write out a wavfile
    for j in range(batch_size):
        audio_buffer = G_z[j,:].detach().cpu().numpy()          
        true_word = timit_words[np.argwhere(labels_local[j,:])[0][0]]
        tf = os.path.join(epoch_path,true_word + '_' + str(uuid.uuid4())+".wav")
        write(tf, 16000, audio_buffer)
        files_for_asr.append(copy.copy(tf))

    if write_only:
        return(None)
    else:
        indices_of_recognized_words, probs = whisper_recognize_wavs(files_for_asr, fast_whisper_model, vocab, timit_words)
        return(indices_of_recognized_words, probs, files_for_asr)

def Q2(G_z, labels, batch_size, unigrams, asr_model, timit_words, epoch_i, decoder, num_cores):
    raise ValueError('Deprecated in favor of Q2_whisper')    
    
    # make a directory to save all outputs for the epoch
    files_for_asr = []
    epoch_path = os.path.join("temp",str(epoch_i))
    if not os.path.exists(epoch_path):
        os.makedirs(epoch_path)

    labels_local = labels.cpu().detach().numpy()
    # for each file in the batch, write out a wavfile
    for j in range(batch_size):
        audio_buffer = G_z[j, 0, :].detach().cpu().numpy()          
        true_word = timit_words[np.argwhere(labels_local[j,:])[0][0]]
        tf = os.path.join(epoch_path,true_word + '_' + str(uuid.uuid4())+".wav")
        write(tf, 16000, audio_buffer)
        files_for_asr.append(copy.copy(tf))
    
    #send the audio buffer through NeMO recognizer
    # Q2 output needs to be over the 11 + 1 UNK categories
    candidates, probabilities = batch_transcribe_with_pyctcdecoder(files_for_asr, lm_path, unigrams, asr_model, decoder, num_cores, timit_words)    

    return(probabilities)

def get_limited_prob(candidates, timit_words): 
    raise ValueError('No longer necessary because of Q2_whisper')               
    exp_d =np.exp(-1.*candidates.combined_score )    
    candidates['prob'] = exp_d / np.sum(exp_d)
    limited_probs = candidates.loc[candidates.hypothesis.isin(timit_words)]
    remainder_prob =  np.sum(candidates.loc[~candidates.hypothesis.isin(timit_words)].prob)
    remainder_row = pd.DataFrame({'hypothesis':["UNK"], "logit_score":[np.nan], "combined_score":[np.nan], "prob":[remainder_prob]})    
    return(pd.concat([limited_probs, remainder_row]))

def process_one_beam(beam_results_tuple, filename, timit_words):
    raise ValueError('No longer necessary because of Q2_whisper')               
    candidates = pd.DataFrame({'hypothesis':[x[0] for x in beam_results_tuple], 'logit_score': -1. * np.array([x[-2] for x in beam_results_tuple]), 'combined_score': -1. * np.array([x[-1] for x in beam_results_tuple])}) 

    candidates.to_csv(filename.replace('.wav','_candidates.csv')) 

    if candidates.shape[0] > len(np.unique(candidates.hypothesis)):
        raise ValueError('Redundant hypotheses!')    

    words_from_lang = get_limited_prob(candidates, timit_words)    
    rdf = pd.DataFrame({'hypothesis':timit_words}).merge(words_from_lang[['hypothesis', 'prob']], how="left") 
    rdf = rdf.fillna(0) 

    rdf.to_csv(filename.replace('.wav','.csv'))

    return((candidates, np.array(rdf.prob.to_list())))    


def batch_transcribe_with_pyctcdecoder(files, lm_path, unigram_list, asr_model, decoder, num_cores, timit_words):
    raise ValueError('No longer necessary because of Q2_whisper')               
    
    logits_for_batch= asr_model.transcribe(files, logprobs=True) # length equal to batch_size
    
    print('Decoding...')
    with multiprocessing.Pool(processes=num_cores) as pool:
        batch_beam_results_tuple = decoder.decode_beams_batch(
            logits_list=logits_for_batch,
            pool=pool,
            beam_prune_logp=-500.,
            token_min_logp=-20.,
            beam_width=32
        )
    print('Done decoding!')

    guesses = [process_one_beam(x[0],x[1], timit_words) for x in  zip(batch_beam_results_tuple, files)]

    candidates = [x[0] for x in guesses]
    probs = [x[1] for x in guesses]

    return(candidates, np.vstack(probs))    
    

if __name__ == "__main__":
    # Training Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--datadir',
        type=str,
        required=True,
        help='Training Directory'
    )
    parser.add_argument(
        '--logdir',
        type=str,
        required=True,
        help='Log/Results Directory'
    )
    parser.add_argument(
        '--num_categ',
        type=int,
        default=0,
        help='Q-net categories'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=5000,
        help='Epochs'
    )
    parser.add_argument(
        '--slice_len',
        type=int,
        default=16384,
        help='Length of training data'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size'
    )
    parser.add_argument(
        '--cont',
        type=str,
        default="",
        help='''continue: default from the last saved iteration. '''
             '''Provide the epoch number if you wish to resume from a specific point'''
             '''Or set "last" to continue from last available'''
    )

    parser.add_argument(
        '--save_int',
        type=int,
        default=25,
        help='Save interval in epochs'
    )

    parser.add_argument(
        '--Q2',
        action='store_true',
        help='Include a secondary Q network that runs the production through an adult listener'
    )

    # Q-net Arguments
    Q_group = parser.add_mutually_exclusive_group()
    Q_group.add_argument(
        '--ciw',
        action='store_true',
        help='Trains a ciwgan'
    )
    Q_group.add_argument(
        '--fiw',
        action='store_true',
        help='Trains a fiwgan'
    )

    args = parser.parse_args()
    train_Q = args.ciw or args.fiw
    train_Q2 = args.Q2
    if args.fiw and args.Q2:
        raise ValueError('Untested -- what happens with the feature representations')
    if args.Q2:
        timit_words = "she had your suit in dark greasy wash water all year".split(' ')+['UNK']
        
        # For the Whisper recognizer
        vocab = pd.read_csv('data/vocab.csv')
        vocab = vocab.loc[vocab['count'] > 20]
        vocab['probability'] = vocab['count'] / np.sum(vocab['count'])
        vocab.word = vocab.word.astype('str')
        faster_whisper_model = faster_whisper.WhisperModel('medium.en', device="cuda", compute_type="float16")



        # For the NeMO recognizer
        # asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained("nvidia/stt_en_conformer_ctc_large")
        # lm_path = 'LM/timit.LM'
        # unigrams = "she had your suit in dark greasy wash water all year".split(' ')
        # unigrams = [str(x) for x in unigrams]
        # if not os.path.exists('temp'):
        #     os.makedirs('temp')
        # decoder = pyctcdecode.build_ctcdecoder(
        #     asr_model.decoder.vocabulary,
        #     unigrams = unigrams,
        #     kenlm_model_path=lm_path,  # either .arpa or .bin file
        #     alpha=4.0,  # tuned on a val set
        #     beta=3.0,  # tuned on a val set
        #     unk_score_offset=-50.
        # )
        # num_cores = multiprocessing.cpu_count()


    # Parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datadir = args.datadir
    logdir = args.logdir
    
    # Epochs and Intervals
    NUM_EPOCHS = args.num_epochs
    WAVEGAN_DISC_NUPDATES = 5
    WAVEGAN_Q2_NUPDATES = 10
    Q2_EPOCH_START = 1000
    WAV_OUTPUT_N = 25
    SAVE_INT = args.save_int
    PRODUCTION_START_EPOCH = 0
    COMPREHENSION_INTERVAL = 100000

    # RL Parameters
    #RL_DISCOUNT_FACTOR = .99
    RL_LEARNING_RATE = .001

    #Sizes of things
    SLICE_LEN = args.slice_len
    NUM_CATEG = args.num_categ
    BATCH_SIZE = args.batch_size

    # GAN Learning rates
    LAMBDA = 10
    LEARNING_RATE = 1e-4
    BETA1 = 0.5
    BETA2 = 0.9
    
    # Verbosity
    label_stages = True

    # Q2 parameters. Only if satisfied will the Q2 network be used. Corresponds to adult decision rule about when to take an action
    Q2_BATCH_SIZE = 6
    MIN_DECODING_PROB = .1
    MAX_NOSPEECH_PROB = .1
    MAX_UNK_PROB = .25
    Q2_TIMEOUT = .4

    CONT = args.cont
    

    # Load data
    dataset = AudioDataSet(datadir, SLICE_LEN, NUM_CATEG, timit_words)
    dataloader = DataLoader(
        dataset,
        BATCH_SIZE,
        shuffle=True,
        num_workers=1,
        drop_last=True
    )

    def make_new():
        G = WaveGANGenerator(slice_len=SLICE_LEN, ).to(device).train()
        D = WaveGANDiscriminator(slice_len=SLICE_LEN).to(device).train()

        # Optimizers
        optimizer_G = optim.Adam(G.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
        optimizer_D = optim.Adam(D.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))

        Q, optimizer_Q_to_G, optimizer_Q_to_Q, criterion_Q  = (None, None, None, None)
        if train_Q:
            Q = WaveGANQNetwork(slice_len=SLICE_LEN, num_categ=NUM_CATEG).to(device).train()
            optimizer_Q_to_G = optim.RMSprop(G.parameters(), lr=LEARNING_RATE)
            optimizer_Q_to_Q = optim.RMSprop(Q.parameters(), lr=LEARNING_RATE)
            # just update the G parameters
            if args.Q2:
                optimizer_Q2_to_Q = optim.RMSprop(Q.parameters(), lr=LEARNING_RATE)

            if args.fiw:
                print("Training a fiwGAN with ", NUM_CATEG, " categories.")
                criterion_Q = torch.nn.BCEWithLogitsLoss()
            elif args.ciw:
                print("Training a ciwGAN with ", NUM_CATEG, " categories.")
                # NOTE: one hot -> category nr. transformation
                # CE loss needs logit, category -> loss
                criterion_Q = lambda inpt, target: torch.nn.CrossEntropyLoss()(inpt, target.max(dim=1)[1])
                
            # if args.Q2:
            #     criterion_Q2 = lambda adult_interp, target: torch.nn.CrossEntropyLoss()(adult_interp, target.max(dim=1)[1]) 
            #     criterion_QQ = lambda child_expected_interp, adult_interp: torch.nn.CrossEntropyLoss()(child_expected_interp, adult_interp.max(dim=1)[1]) 
            # else: 
            #     criterion_Q2 = None
            #     criterion_QQ = None
            

        return G, D, optimizer_G, optimizer_D, Q, optimizer_Q_to_G, optimizer_Q_to_Q, optimizer_Q2_to_Q, criterion_Q

    # Load models
    G, D, optimizer_G, optimizer_D, Q, optimizer_Q_to_G, optimizer_Q_to_Q, optimizer_Q2_to_Q, criterion_Q = make_new()
        
    
    start_epoch = 0
    start_step = 0

    if str(CONT).lower() != "":
        
        try:
            print("Loading model from existing checkpoints...")
            fname, start_epoch = get_continuation_fname(CONT, logdir)

            G.load_state_dict(torch.load(f=os.path.join(logdir, fname + "_G.pt")))
            D.load_state_dict(torch.load(f=os.path.join(logdir, fname + "_D.pt")))
            if train_Q:
                Q.load_state_dict(torch.load(f=os.path.join(logdir, fname + "_Q.pt")))
                # there is no change over time to the Q2 network

            optimizer_G.load_state_dict(torch.load(f=os.path.join(logdir, fname + "_Gopt.pt")))
            optimizer_D.load_state_dict(torch.load(f=os.path.join(logdir, fname + "_Dopt.pt")))

            if train_Q:
                optimizer_Q_to_G.load_state_dict(torch.load(f=os.path.join(logdir, fname + "_Q_to_Gopt.pt")))
                optimizer_Q_to_Q.load_state_dict(torch.load(f=os.path.join(logdir, fname + "_Q_to_Qopt.pt")))
                optimizer_Q2_to_Q.load_state_dict(torch.load(f=os.path.join(logdir, fname + "_Q2_to_Qopt.pt")))

            start_step = int(re.search(r'_step(\d+).*', fname).group(1))
            print(f"Successfully loaded model. Continuing training from epoch {start_epoch},"
                  f" step {start_step}")
        except:
            "Problem loading existing model!"
        
    else:
        print("Starting a new training")

    # Set Up Writer
    writer = SummaryWriter(logdir)
    step = start_step

    for epoch in range(start_epoch + 1, NUM_EPOCHS):

        print("Epoch {} of {}".format(epoch, NUM_EPOCHS))
        print("-----------------------------------------")


        pbar = tqdm(dataloader)        

        for i, trial in enumerate(pbar):
            
            reals = trial[0].to(device)
            labels = trial[1].to(device)

            if (epoch <= PRODUCTION_START_EPOCH) or (epoch % COMPREHENSION_INTERVAL == 0):
                # Just train the Q network from external data
                if label_stages:
                    print('Updating Q network to identify referents')

                optimizer_Q_to_Q.zero_grad()
                child_recovers_from_adult = Q(reals)    
                Q_comprehension_loss = criterion_Q(child_recovers_from_adult, labels[:,0:NUM_CATEG]) # Note we exclude the UNK label --  child never intends to produce unk
                Q_comprehension_loss.backward()
                writer.add_scalar('Loss/Q_comprehension', Q_comprehension_loss.detach().item(), step)
                optimizer_Q_to_Q.step()
                step += 1
                            

            else:
                # Discriminator Update
                optimizer_D.zero_grad()            

                epsilon = torch.rand(BATCH_SIZE, 1, 1).repeat(1, 1, SLICE_LEN).to(device)
                _z = torch.FloatTensor(BATCH_SIZE, 100 - (NUM_CATEG + 1)).uniform_(-1, 1).to(device)

                if train_Q:
                    #zeros = torch.zeros(BATCH_SIZE, 1).to(device)
                    if args.fiw:
                        raise NotImplementedError
                        c = torch.FloatTensor(BATCH_SIZE, NUM_CATEG).bernoulli_().to(device)
                    
                    else:                    
                        c = labels  
                        
                        # c = torch.nn.functional.one_hot(torch.randint(0, NUM_CATEG, (BATCH_SIZE,)),
                        #                                 num_classes=NUM_CATEG).to(device)
                    z = torch.cat((c, _z), dim=1)
                    #z = torch.cat((c, zeros, _z), dim=1)
                else:
                    raise NotImplementedError
                    z = _z

                fake = G(z)
                penalty = gradient_penalty(G, D, reals, fake, epsilon)

                D_loss = torch.mean(D(fake) - D(reals) + LAMBDA * penalty)
                writer.add_scalar('Loss/Discriminator', D_loss.detach().item(), step)
                D_loss.backward()
                if label_stages:
                    print('Discriminator update!')
                optimizer_D.step()            

                if i % WAVEGAN_DISC_NUPDATES == 0:
                    optimizer_G.zero_grad()
                                                                
                    _z = torch.FloatTensor(BATCH_SIZE, 100 - (NUM_CATEG + 1)).uniform_(-1, 1).to(device)


                    if train_Q:
                        optimizer_Q_to_G.zero_grad()
                        if args.fiw:
                            raise NotImplementedError
                            c = torch.FloatTensor(BATCH_SIZE, NUM_CATEG).bernoulli_().to(device)                        
                        else:
                            c = labels 
                            #c = torch.nn.functional.one_hot(torch.randint(0, NUM_CATEG, (BATCH_SIZE,)),
                            # num_classes=NUM_CATEG).to(device)

                        z = torch.cat((c, _z), dim=1)
                    else:
                        z = _z

                    G_z = G(z)

                    # G Loss
                    G_loss = torch.mean(-D(G_z))
                    G_loss.backward(retain_graph=True)
                    # Update
                    optimizer_G.step()
                    if label_stages:
                        print('Generator update!')
                    writer.add_scalar('Loss/Generator', G_loss.detach().item(), step)


                    if (epoch % WAV_OUTPUT_N == 0) & (i <= 1) & (epoch < Q2_EPOCH_START):
                         
                        print('Sampling .wave outputs (but not running them through Q2)...')
                        wav_output = Q2_whisper(G_z, c, faster_whisper_model, timit_words, vocab, epoch, BATCH_SIZE, write_only=True)
                        # but don't do anything with it; just let it write out all of the audio files

                    # Q2 Loss: Update G and Q to better imitate the Q2 model
                    if (i != 0) and train_Q2 and (i % WAVEGAN_Q2_NUPDATES == 0) & (epoch >= Q2_EPOCH_START):
                        
                        if label_stages:
                            print('Starting Q2 -> Q update')                        

                        predicted_value_loss = torch.nn.CrossEntropyLoss()                    
                        Q2_prediction_error = []
                        recoveries = []
                        total_recognized_words = 0

                        for i in range(NUM_CATEG):

                            for j in tqdm(range(Q2_BATCH_SIZE)):

                                selected_referents = np.zeros([BATCH_SIZE, NUM_CATEG+1], dtype=np.float32)
                                selected_referents[:,i] = 1
                                selected_referents = torch.Tensor(selected_referents).to(device)
                                _z = torch.FloatTensor(BATCH_SIZE, 100 - (NUM_CATEG + 1)).uniform_(-1, 1).to(device)

                                candidates = G(torch.cat((selected_referents, _z), dim=1))
                                Q_applied_to_candidates = Q(candidates)

                                # select the Q2_BATCH_SIZE items that are most likely to produce the correct response
                                predicted_values = torch.Tensor([predicted_value_loss(Q_applied_to_candidates[i], selected_referents[i,0:NUM_CATEG]) for i in range(Q_applied_to_candidates.shape[0])])
                                
                                # get the n best candidates from the argmax of the scores
                                print('Select the n best productions (lowest cross entropy) for this word right here')                            
                                selected_candidate_index = torch.argsort(predicted_values, dim=- 1, descending=False, stable=False)[0]

                                #predicted_values[torch.argsort(predicted_values, dim=- 1, descending=False, stable=False)[0:Q2_BATCH_SIZE]]                                
                                selected_candidate = candidates[selected_candidate_index,:]

                                # send the candidate through Whisper
                                print('Recognizing words with Whisper model...')  

                                indices_of_recognized_words, Q2_probs, filenames = Q2_whisper(selected_candidate, selected_referents[0:1,], faster_whisper_model, timit_words, vocab, epoch, 1)
                                total_recognized_words += len(indices_of_recognized_words)
                                                        
                                if len(indices_of_recognized_words) > 0:
                                
                                    assert(Q2_probs.shape[1] == NUM_CATEG+1)
                                    Q2_output = torch.from_numpy(Q2_probs.astype(np.float32)).to(device) # for padding the UNKs, in logit space

                                    zero_tensor = torch.Tensor([float(0)]).to(device)
                                    optimizer_Q2_to_Q.zero_grad() # clear the gradients for the Q update
                                    # update Q    
                                    print('Child > Adult Trial with referent '+str(i)+', instance '+str(j))
                                    #optimizer_Q_to_Q.zero_grad() # clear the gradients for the Q update
                                
                                    # compute the expected value of each utterance
                                    # this is a one shot game for each reference, so implicitly the value before taking the action is 0. I might update this later, i.e., think about this in terms of sequences
                                    # recall that actions were G_z / candidates ; the value function is Q(G_z)

                                    Q_prediction = torch.softmax(Q_applied_to_candidates[selected_candidate_index:selected_candidate_index+1,:], dim=1)[0]                                    
                                    #Q_prediction = Q(selected_candidates[index_of_recognized_word,:])[0] # regenerate because of the zero_grad
                                    augmented_Q_prediction = torch.cat((Q_prediction, zero_tensor))

                                    # is the Q prediction the same as selected_referents?
                                    if not torch.eq(torch.argmax(selected_referents[0:1,]), torch.argmax(augmented_Q_prediction)).detach().cpu().numpy():
                                        print("Child model produced an utterance that they don't think will invoke the correct action. Consider choosing action from a larger set of actions")
                                                    
                                    #predicted_val = predicted_value_loss(augmented_Q_prediction, selected_referents[index_of_recognized_word])
                                                                
                                    # compute the actual cross entropy for corresponding index_of_recognized_word. Reward depends on if the adult is able to pull out the indended message, indexed by label. Detaching so that it doesn't try to backprop through reward
                                    #reward = actual_loss(selected_referents[index_of_recognized_word], Q2_output[index_of_recognized_word]).detach()
                                    #print('Reward: '+str(reward))
                                    #rewards.append(reward.cpu().numpy())

                                    # Q2 outputs are the correct class labels

                                    Q2_loss = predicted_value_loss(augmented_Q_prediction, Q2_output[0])
                                    #Q2 is prediction error of the Q network with respect to the Q2 outputs
                                    print(augmented_Q_prediction)
                                    print(Q2_output[0])
                                    print(Q2_loss)
                                
                                    # count the number of words correctly recovered: referent -> recognized by listener
                                    recovered = torch.eq(torch.argmax(selected_referents[0,]), torch.argmax(Q2_output[0])).cpu().numpy().tolist()
                                    recoveries.append(recovered)                                    

                                    Q2_prediction_error.append(Q2_loss.detach().cpu().numpy())                                    
                                                                                                                 
                                    Q2_loss.backward(retain_graph=True)                            
                                
                                    print('Gradients on the Q network:')
                                    print('Q layer 0: '+str(np.round(torch.sum(torch.abs(Q.downconv_0.conv.weight.grad)).cpu().numpy(), 10)))
                                    print('Q layer 1: '+str(np.round(torch.sum(torch.abs(Q.downconv_1.conv.weight.grad)).cpu().numpy(), 10)))
                                    print('Q layer 2: '+str(np.round(torch.sum(torch.abs(Q.downconv_2.conv.weight.grad)).cpu().numpy(), 10)))
                                    print('Q layer 3: '+str(np.round(torch.sum(torch.abs(Q.downconv_3.conv.weight.grad)).cpu().numpy(), 10)))
                                    print('Q layer 4: '+str(np.round(torch.sum(torch.abs(Q.downconv_4.conv.weight.grad)).cpu().numpy(), 10)))

                                    print('Q2 step!')
                                    optimizer_Q2_to_Q.step()

                        writer.add_scalar('Metric/Proportion Recovered Words Among Recognized', np.sum(recoveries) / total_recognized_words, step)
                        writer.add_scalar('Metric/Proportion Recovered Words Among Total', np.sum(recoveries) / (Q2_BATCH_SIZE *NUM_CATEG), step)
                        writer.add_scalar('Metric/Proportion Recognized Words Among Total', total_recognized_words / (Q2_BATCH_SIZE *NUM_CATEG), step)

                        writer.add_scalar('Loss/Q2: Prediction Error from the Q network', np.mean(Q2_prediction_error), step)


                    elif train_Q:         
                        if label_stages:
                            print('Q -> G update')
                        
                        optimizer_Q_to_G.zero_grad()
                        Q_production_loss = criterion_Q(Q(G(z)), c[:,0:NUM_CATEG]) # Note we exclude the UNK label --  child never intends to produce unk
                        Q_production_loss.backward()
                        writer.add_scalar('Loss/Q_production', Q_production_loss.detach().item(), step)
                        optimizer_Q_to_G.step()

                    
                step += 1

        if epoch % SAVE_INT == 0:
            if G is not None:
                torch.save(G.state_dict(), os.path.join(logdir, f'epoch{epoch}_step{step}_G.pt'))
            if D is not None:
                torch.save(D.state_dict(), os.path.join(logdir, f'epoch{epoch}_step{step}_D.pt'))
            if train_Q:
                torch.save(Q.state_dict(), os.path.join(logdir, f'epoch{epoch}_step{step}_Q.pt'))                
                # these is no Q2 network to save, nor QQ            

            if optimizer_G is not None:
                torch.save(optimizer_G.state_dict(), os.path.join(logdir, f'epoch{epoch}_step{step}_Gopt.pt'))
            if optimizer_D is not None:
                torch.save(optimizer_D.state_dict(), os.path.join(logdir, f'epoch{epoch}_step{step}_Dopt.pt'))
            if train_Q:
                torch.save(optimizer_Q_to_G.state_dict(), os.path.join(logdir, f'epoch{epoch}_step{step}_Q_to_Gopt.pt'))
            if train_Q2 and optimizer_Q_to_Q is not None:
                torch.save(optimizer_Q_to_Q.state_dict(), os.path.join(logdir, f'epoch{epoch}_step{step}_Q_to_Qopt.pt'))
            if train_Q2 and optimizer_Q2_to_Q is not None:
                torch.save(optimizer_Q2_to_Q.state_dict(), os.path.join(logdir, f'epoch{epoch}_step{step}_Q_to_Q2opt.pt'))

