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
import nemo.collections.asr as nemo_asr
import pyctcdecode
import kenlm
import tempfile
import scipy
import uuid
import multiprocessing
import copy

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

def Q2(G_z, labels, batch_size, unigrams, asr_model, timit_words, epoch_i, decoder, num_cores):
    
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
    guesses = batch_transcribe_with_pyctcdecoder(files_for_asr, lm_path, unigrams, asr_model, decoder, num_cores)    

    return(guesses)

def get_limited_softmax(candidates, timit_words):
    candidates['prob'] = scipy.special.softmax(np.array(candidates.logit_score))
    limited_logits = candidates.loc[candidates.hypothesis.isin(timit_words)]
    remainder_prob =  np.sum(candidates.loc[~candidates.hypothesis.isin(timit_words)].prob)
    remainder_row = pd.DataFrame({'hypothesis':["UNK"], "logit_score":[np.nan], "combined_score":[np.nan], "prob":[remainder_prob]})
    
    return(pd.concat([limited_logits, remainder_row]))


def process_one_beam(beam_results_tuple, filename, timit_words):
    candidates = pd.DataFrame({'hypothesis':[x[0] for x in beam_results_tuple], 'logit_score': -1. * np.array([x[-2] for x in beam_results_tuple]), 'combined_score': -1. * np.array([x[-1] for x in beam_results_tuple])}) 

    if candidates.shape[0] > len(np.unique(candidates.hypothesis)):
        raise ValueError('Redundant hypotheses!')    

    words_from_lang = get_limited_softmax(candidates, timit_words)    
    rdf = pd.DataFrame({'hypothesis':timit_words}).merge(words_from_lang[['hypothesis', 'prob']], how="left") 
    rdf = rdf.fillna(0) 

    rdf.to_csv(filename.replace('.wav','.csv'))

    return(np.array(rdf.prob.to_list()))    


def batch_transcribe_with_pyctcdecoder(files, lm_path, unigram_list, asr_model, decoder, num_cores):
    
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

    return(np.vstack(guesses))    
    

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
        default=50,
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
        asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained("nvidia/stt_en_conformer_ctc_large")

        lm_path = '/home/stephan/notebooks/cdl-asr/lowercase_3-gram.pruned.1e-7.LM'
        unigrams = pd.read_table('/home/stephan/notebooks/cdl-asr/lowercase_3-gram.pruned.1e-7_unigram.txt', header=None)[0].to_list()
        unigrams = [str(x) for x in unigrams]
        if not os.path.exists('temp'):
            os.makedirs('temp')
        decoder = pyctcdecode.build_ctcdecoder(
            asr_model.decoder.vocabulary,
            unigrams = unigrams,
            kenlm_model_path=lm_path,  # either .arpa or .bin file
            alpha=2.0,  # tuned on a val set
            beta=1.5,  # tuned on a val set
            unk_score_offset=-50.
        )
        num_cores = multiprocessing.cpu_count()


    # Parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datadir = args.datadir
    logdir = args.logdir
    SLICE_LEN = args.slice_len
    NUM_CATEG = args.num_categ
    NUM_EPOCHS = args.num_epochs
    WAVEGAN_DISC_NUPDATES = 5
    WAVEGAN_Q2_NUPDATES = 10
    Q2_EPOCH_START = 3000
    BATCH_SIZE = args.batch_size
    LAMBDA = 10
    LEARNING_RATE = 1e-4
    BETA1 = 0.5
    BETA2 = 0.9
    label_stages = False

    CONT = args.cont
    SAVE_INT = args.save_int

    # Load data
    dataset = AudioDataSet(datadir, SLICE_LEN, NUM_CATEG, timit_words)
    dataloader = DataLoader(
        dataset,
        BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        drop_last=True
    )

    def make_new():
        G = WaveGANGenerator(slice_len=SLICE_LEN, ).to(device).train()
        D = WaveGANDiscriminator(slice_len=SLICE_LEN).to(device).train()

        # Optimizers
        optimizer_G = optim.Adam(G.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
        optimizer_D = optim.Adam(D.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))

        Q, optimizer_Q, optimizer_Q2, criterion_Q, criterion_Q2  = (None, None, None, None, None)
        if train_Q:
            Q = WaveGANQNetwork(slice_len=SLICE_LEN, num_categ=NUM_CATEG).to(device).train()
            optimizer_Q = optim.RMSprop(it.chain(G.parameters(), Q.parameters()),
                                        lr=LEARNING_RATE)
            # just update the G parameters
            if args.Q2:
                optimizer_Q2 = optim.RMSprop(G.parameters(), lr=LEARNING_RATE)

            if args.fiw:
                print("Training a fiwGAN with ", NUM_CATEG, " categories.")
                criterion_Q = torch.nn.BCEWithLogitsLoss()
            elif args.ciw:
                print("Training a ciwGAN with ", NUM_CATEG, " categories.")
                # NOTE: one hot -> category nr. transformation
                # CE loss needs logit, category -> loss
                criterion_Q = lambda inpt, target: torch.nn.CrossEntropyLoss()(inpt, target.max(dim=1)[1])
                
            if args.Q2:
                criterion_Q2 = lambda adult_interp, target: torch.nn.CrossEntropyLoss()(adult_interp, target.max(dim=1)[1]) 
            else: 
                criterion_Q2 = None
            

        return G, D, optimizer_G, optimizer_D, Q, optimizer_Q, optimizer_Q2, criterion_Q, criterion_Q2

    # Load models
    G, D, optimizer_G, optimizer_D, Q, optimizer_Q, optimizer_Q2, criterion_Q, criterion_Q2  = make_new()
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
                optimizer_Q.load_state_dict(torch.load(f=os.path.join(logdir, fname + "_Qopt.pt")))

            if train_Q2:
                optimizer_Q2.load_state_dict(torch.load(f=os.path.join(logdir, fname + "_Q2opt.pt")))


            start_step = int(re.search(r'_step(\d+).*', fname).group(1))
            print(f"Successfully loaded model. Continuing training from epoch {start_epoch},"
                  f" step {start_step}")

        # Don't care why it failed
        except Exception as e:
            print("Could not load from existing checkpoint, initializing new model...")
            print(e)
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
            # D Update
            optimizer_D.zero_grad()
            
            reals = trial[0].to(device)
            labels = trial[1].to(device)

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

                if train_Q:
                    optimizer_Q.zero_grad()
                if train_Q2:    
                    optimizer_Q2.zero_grad()                    

                _z = torch.FloatTensor(BATCH_SIZE, 100 - (NUM_CATEG + 1)).uniform_(-1, 1).to(device)


                if train_Q:
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
                writer.add_scalar('Loss/Generator', G_loss.detach().item(), step)

                # Q Loss
                if train_Q:                                                
                    Q_loss = criterion_Q(Q(G_z), c[:,0:NUM_CATEG]) # exclude the UNK label --  child never intends to produce unk
                    Q_loss.backward()
                    writer.add_scalar('Loss/Q_Network', Q_loss.detach().item(), step)
                    optimizer_Q.step()
                    if label_stages:
                        print('Q network update!')
                    # based on the Q loss, update G and D, right?

                # Q2 Loss: From the secondary network
                if (i != 0) and train_Q2 and (i % WAVEGAN_Q2_NUPDATES == 0) & (epoch >= Q2_EPOCH_START):                
                    Q2_loss = criterion_Q2(torch.from_numpy(Q2(G_z, labels, BATCH_SIZE, unigrams, asr_model, timit_words, epoch, decoder, num_cores)).to(device).requires_grad_(), c)                              
                    Q2_loss.backward()
                    writer.add_scalar('Loss/Q2_Network', Q2_loss, step)                    
                    optimizer_Q2.step()
                    if label_stages:
                        print('Q2 network update!')

                # Update
                optimizer_G.step()
                if label_stages:
                    print('Generator update!')
            step += 1

        if epoch % SAVE_INT == 0:
            torch.save(G.state_dict(), os.path.join(logdir, f'epoch{epoch}_step{step}_G.pt'))
            torch.save(D.state_dict(), os.path.join(logdir, f'epoch{epoch}_step{step}_D.pt'))
            if train_Q:
                torch.save(Q.state_dict(), os.path.join(logdir, f'epoch{epoch}_step{step}_Q.pt'))
            # these is no Q2 network to save            

            torch.save(optimizer_G.state_dict(), os.path.join(logdir, f'epoch{epoch}_step{step}_Gopt.pt'))
            torch.save(optimizer_D.state_dict(), os.path.join(logdir, f'epoch{epoch}_step{step}_Dopt.pt'))
            if train_Q:
                torch.save(optimizer_Q.state_dict(), os.path.join(logdir, f'epoch{epoch}_step{step}_Qopt.pt'))
            if train_Q2:
                torch.save(optimizer_Q2.state_dict(), os.path.join(logdir, f'epoch{epoch}_step{step}_Q2opt.pt'))
