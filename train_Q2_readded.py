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
import gc
import wandb

torch.autograd.set_detect_anomaly(True)

class AudioDataSet:
    def __init__(self, datadir, slice_len, NUM_CATEG, vocab):
        print("Loading data")
        dir = os.listdir(datadir)
        x = np.zeros((len(dir), 1, slice_len))
        y = np.zeros((len(dir), NUM_CATEG+1)) # +1 for UNK

        i = 0
        files = []
        for file in tqdm(dir):
            files.append(file)
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
            j = vocab.index(word)
            y[i, j] = 1            
            i += 1

        self.len = len(x)
        self.audio = torch.from_numpy(np.array(x, dtype=np.float32))
        self.labels = torch.from_numpy(np.array(y, dtype=np.float32))

    def __getitem__(self, index):
        return ((self.audio[index], self.labels[index]))

    def __len__(self):
        return self.len


def gradient_penalty(G, D, reals, fakes, epsilon):
    x_hat = epsilon * reals + (1 - epsilon) * fakes
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


def Q2_cnn(selected_candidate_wavs, Q2):
    print('in Q2_cnn')
    Q2_probs = torch.softmax(Q2(selected_candidate_wavs), dim=1)
    # add a column for UNKs
    zeros = torch.zeros([Q2_probs.shape[0],1], device = device) + .00000001
    Q_network_probs_with_unk = torch.hstack((Q2_probs, zeros))
    #indices_of_recognized_words = range(Q_network_probs_with_unk.shape[0])
    return(Q_network_probs_with_unk)

def write_out_wavs(G_z_2d, labels, vocab, logdir, epoch):
    # returns probabilities and a set of indices; takes a smaller number of arguments
    files_for_asr = []
    epoch_path = os.path.join(logdir,'audio_files',str(epoch))
    if not os.path.exists(epoch_path):
        os.makedirs(epoch_path)    

    labels_local = labels.cpu().detach().numpy()
    # for each file in the batch, write out a wavfile
    for j in range(G_z_2d.shape[0]):
        audio_buffer = G_z_2d[j,:].detach().cpu().numpy()          
        true_word = vocab[np.argwhere(labels_local[j,:])[0][0]]
        tf = os.path.join(epoch_path,true_word + '_' + str(uuid.uuid4())+".wav")
        write(tf, 16000, audio_buffer[0])
        files_for_asr.append(copy.copy(tf))
    return(files_for_asr)
    

def mark_unks_in_Q2(Q_network_probs, threshold, device):
    print('in mark_unks_in_Q2')
    # need a little prob mass on the UNKs to avoid numerical errors if this is the Q network     

    #unk_tensor = torch.zeros([1, Q_network_probs.shape[1]], device = device)
    #unk_tensor += .0001
    #unk_tensor[0,-1] = .999999

    # compute entropies
    log_probs = torch.log(Q_network_probs + .000000001)
    prod = Q_network_probs * log_probs
    entropy = -torch.sum(prod, dim =1)        
    
    indices_of_recognized_words = torch.argwhere( entropy <= torch.Tensor([threshold]).to(device)).flatten()

    # unks_to_mark = torch.argwhere( entropy > torch.Tensor([threshold]).to(device))
    # if len(unks_to_mark) > 0:
    #     Q_network_probs[unks_to_mark,] = unk_tensor    

    return(indices_of_recognized_words, Q_network_probs)


if __name__ == "__main__":
    # Training Arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--architecture',
        type=str,
        required=True,
        help='Architecure. Can be ciwgan for fiwgan (fiwgan is not implemented yet)'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        required=True,
        help='Log/Results Directory. Results will be stored by wandb_group / wandb_id / epoch (see below)'
    )

    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='directory with labeled waveforms'
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
        '--track_q2',
        type=int,
        help='Track the results with the Q2 network; to backpropagate from Q2 to Q see "backprop_from_Q2"'
    )
    parser.add_argument(
        '--backprop_from_Q2',
        type=int,
        help='Update the Q network from the Q2 network'
    )

    parser.add_argument(
        '--production_start_epoch',
        type=int,
        help='Do n-1 epochs of pretraining the child Q network in the reference game. 0 means produce from the beginning',
        default=0
    )

    parser.add_argument(
        '--comprehension_interval',
        type=int,
        help='How often, in terms of epochs should the Q network be re-trained in the reference game. THe high default means that this is never run',
        default = 10000000
    )

    parser.add_argument(
        '--q2_entropy_threshold',
        type=float,
        help="Float representing the entropy interval. If none of the referents reaches .25, put all probability mass on the final category of UNK, and then don't packprop from it",
        default=.25
    )

    parser.add_argument(
        '--keep_intermediate_epochs',
        action='store_true',
        help='Keep previous checkpoints after a new one is saved?',
        default=False
    )

    parser.add_argument(
        '--wandb_project',
        type=str,
        help='Name of the project for tracking in Weights and Biases',        
    )

    parser.add_argument(
        '--wandb_group',
        type=str,
        help='Name of the group / experiment to which this version (id) belongs',        
    )

    parser.add_argument(
        '--wandb_id',
        type=str,
        help='Name of this specific run (distinguishing it from others in the group)',        
    )

    parser.add_argument(
        '--wavegan_disc_nupdates',
        type=int,
        help='On what interval, in steps, should the discriminator be updated? On other steps, the model updates the generator',
        default = 4
    )

    parser.add_argument(
        '--wavegan_q2_nupdates',
        type=int,
        help='On what interval, in steps, should the loss on the Q prediction of the Q2 labels be used to update the ! network ',
        default = 8
    )

    parser.add_argument(
        '--learning_rate',
        type=float,
        help="Float for the learning rate",
        default=1e-4
    )

    parser.add_argument(
        '--num_q2_training_epochs',
        type=int,
        help='Number of epochs to traine the adult model',
        default = 25
    )

    parser.add_argument(
        '--vocab',
        type=str,
        required=True,
        help='Space-separated vocabulary. Indices of words here will be used as the ground truth.'
    )

    parser.add_argument(
        '--q2_batch_size',
        type=int,
        help='Number of candidates to evaluate for each word to choose the best candidate',
        default = 6
    )

    args = parser.parse_args()
    train_Q = True
    track_Q2 = bool(args.track_q2)
    if args.architecture == 'fiwgan':
        raise ValueError('Untested -- what happens with the feature representations')
    if track_Q2:        
        vocab = args.vocab.split(' ')+['UNK']

    # Parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datadir = args.data_dir
    logdir = os.path.join(args.log_dir, args.wandb_group, args.wandb_id)
    
    # Epochs and Intervals
    NUM_EPOCHS = args.num_epochs
    WAVEGAN_DISC_NUPDATES = args.wavegan_disc_nupdates
    WAVEGAN_Q2_NUPDATES = args.wavegan_q2_nupdates
    Q2_EPOCH_START = 0 # in case we want to only run Q2 after a certain epoch. Less of a concern when a fast Q2 is used
    WAV_OUTPUT_N = 5
    SAVE_INT = args.save_int
    PRODUCTION_START_EPOCH = args.production_start_epoch
    COMPREHENSION_INTERVAL = args.comprehension_interval


    #Sizes of things
    SLICE_LEN = args.slice_len
    NUM_CATEG = len(args.vocab.split(' '))
    BATCH_SIZE = args.batch_size

    # GAN Learning rates
    LAMBDA = 10
    LEARNING_RATE = args.learning_rate
    BETA1 = 0.5
    BETA2 = 0.9
    
    # Verbosity
    label_stages = True

    # Q2 parameters
    NUM_Q2_TRAINING_EPOCHS = args.num_q2_training_epochs
    Q2_BATCH_SIZE = args.q2_batch_size
    Q2_ENTROPY_THRESHOLD = args.q2_entropy_threshold

    CONT = args.cont    

    wandb.init(        
        project=args.wandb_project,        
        config=args.__dict__,
        group=args.wandb_group,
        id = args.wandb_id
    )

    # Load data
    dataset = AudioDataSet(datadir, SLICE_LEN, NUM_CATEG, vocab)
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
            #optimizer_Q_to_G = optim.RMSprop(G.parameters(), lr=LEARNING_RATE)
            optimizer_Q_to_QG = optim.RMSprop(it.chain(G.parameters(), Q.parameters()), lr=LEARNING_RATE)
            optimizer_Q_to_Q = optim.RMSprop(Q.parameters(), lr=LEARNING_RATE)            
            # just update the G parameters
            if track_Q2:
                Q2 = WaveGANQNetwork(slice_len=SLICE_LEN, num_categ=NUM_CATEG).to(device).train()
                optimizer_Q2_to_QG = optim.RMSprop(it.chain(G.parameters(), Q.parameters()), lr=LEARNING_RATE)
                optimizer_Q2_to_Q2 = optim.RMSprop(Q2.parameters(), lr=LEARNING_RATE)

            if args.architecture == 'fiwgan':
                print("Training a fiwGAN with ", NUM_CATEG, " categories.")
                criterion_Q = torch.nn.BCEWithLogitsLoss()
            elif args.architecture == 'ciwgan':
                print("Training a ciwGAN with ", NUM_CATEG, " categories.")
                # NOTE: one hot -> category nr. transformation
                # CE loss needs logit, category -> loss
                criterion_Q = lambda inpt, target: torch.nn.CrossEntropyLoss()(inpt, target.max(dim=1)[1])
            else:
                raise ValueError('Architecure not recognized! Must be fiwgan or ciwgan')                
                
            # if args.Q2:
            #     criterion_Q2 = lambda adult_interp, target: torch.nn.CrossEntropyLoss()(adult_interp, target.max(dim=1)[1]) 
            #     criterion_QQ = lambda child_expected_interp, adult_interp: torch.nn.CrossEntropyLoss()(child_expected_interp, adult_interp.max(dim=1)[1]) 
            # else: 
            #     criterion_Q2 = None
            #     criterion_QQ = None
            

        return G, D, optimizer_G, optimizer_D, Q, Q2, optimizer_Q_to_QG, optimizer_Q2_to_QG, optimizer_Q2_to_Q2, criterion_Q, optimizer_Q_to_Q

    # Load models
    G, D, optimizer_G, optimizer_D, Q, Q2, optimizer_Q_to_QG, optimizer_Q2_to_QG, optimizer_Q2_to_Q2, criterion_Q, optimizer_Q_to_Q = make_new()
        
    
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
                Q2.load_state_dict(torch.load(f=os.path.join(logdir, fname + "_Q2.pt")))

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


    regenerate_Q2 = False
    if regenerate_Q2:
        print('Training an Adult Q2 CNN Network')
        step = start_step
        for epoch in range(start_epoch + 1, NUM_Q2_TRAINING_EPOCHS):
            print("Epoch {} of {}".format(epoch, NUM_Q2_TRAINING_EPOCHS))
            print("-----------------------------------------")

            pbar = tqdm(dataloader)            
            for i, trial in enumerate(pbar):            
                reals = trial[0].to(device)
                labels = trial[1].to(device)        
                optimizer_Q2_to_Q2.zero_grad()
                Q2_logits = Q2(reals)    
                
                Q2_comprehension_loss = criterion_Q(Q2_logits, labels[:,0:NUM_CATEG]) # Note we exclude the UNK label --  child never intends to produce unk
                Q2_comprehension_loss.backward()
                writer.add_scalar('Loss/Q2 to Q2', Q2_comprehension_loss.detach().item(), step)
                wandb.log({"Loss/Q2 to Q2": Q2_comprehension_loss.detach().item()}, step=step)
                optimizer_Q2_to_Q2.step()
                step += 1
        torch.save(Q2, 'saved_networks/adult_pretrained_Q_network_'+str(NUM_CATEG)+'.torch')

    else:
        print('Loading a Previous Adult Q2 CNN Network')
        Q2 = torch.load('saved_networks/adult_pretrained_Q_network_'+str(NUM_CATEG)+'.torch')

    # freeze it
    Q2.eval()        
    for p in Q2.parameters():
        p.requires_grad = False

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
                    print('Updating Child Q network to identify referents')

                optimizer_Q_to_Q.zero_grad()
                child_recovers_from_adult = Q(reals)    
                Q_comprehension_loss = criterion_Q(child_recovers_from_adult, labels[:,0:NUM_CATEG]) # Note we exclude the UNK label --  child never intends to produce unk
                Q_comprehension_loss.backward()
                writer.add_scalar('Loss/Q to Q', Q_comprehension_loss.detach().item(), step)
                wandb.log({"Loss/Q to Q": Q_comprehension_loss.detach().item()}, step=step)
                optimizer_Q_to_Q.step()
                step += 1
                            

            else:
                # Discriminator Update
                optimizer_D.zero_grad()                 

                epsilon = torch.rand(BATCH_SIZE, 1, 1).repeat(1, 1, SLICE_LEN).to(device)
                
                c = torch.nn.functional.one_hot(torch.randint(0, NUM_CATEG, (BATCH_SIZE,)),
                                                         num_classes=NUM_CATEG).to(device)
                zeros = torch.zeros([BATCH_SIZE,1], device = device)
                _z = torch.FloatTensor(BATCH_SIZE, 100 - (NUM_CATEG + 1)).uniform_(-1, 1).to(device)
                z = torch.cat((c, zeros, _z), dim=1)
                fakes = G(z)

                # shuffle the reals so that the matched item for discrim is not necessarily from the same referent                
                shuffled_reals = reals[torch.randperm(reals.shape[0]),:,:]
                
                penalty = gradient_penalty(G, D, shuffled_reals, fakes, epsilon)
                D_loss = torch.mean(D(fakes) - D(shuffled_reals) + LAMBDA * penalty)
                writer.add_scalar('Loss/D', D_loss.detach().item(), step)
                if (step % 6)  == 0:  
                    wandb.log({"Loss/D": D_loss.detach().item()}, step=step)
                D_loss.backward()
                if label_stages:
                    print('Discriminator update!')
                optimizer_D.step()            
                optimizer_D.zero_grad()

                if i % WAVEGAN_DISC_NUPDATES == 0:
                    optimizer_G.zero_grad()
                                                                
                    
                    if label_stages:
                        print('D -> G  update')


                    c = torch.nn.functional.one_hot(torch.randint(0, NUM_CATEG, (BATCH_SIZE,)),
                             num_classes=NUM_CATEG).to(device)
                    _z = torch.FloatTensor(BATCH_SIZE, 100 - (NUM_CATEG + 1)).uniform_(-1, 1).to(device)
                    zeros = torch.zeros([BATCH_SIZE,1], device = device)
                    z = torch.cat((c, zeros, _z), dim=1)

                    G_z_for_G_update = G(z) # generate again using the same labels

                    # G Loss
                    G_loss = torch.mean(-D(G_z_for_G_update))
                    G_loss.backward(retain_graph=True)
                    # Update
                    optimizer_G.step()
                    optimizer_G.zero_grad()
                    if label_stages:
                        print('Generator update!')
                    writer.add_scalar('Loss/G', G_loss.detach().item(), step)
                    wandb.log({"Loss/G": G_loss.detach().item()}, step=step)


                    if (epoch % WAV_OUTPUT_N == 0) & (i <= 1):
                         
                        print('Sampling .wav outputs (but not running them through Q2)...')
                        write_out_wavs(G_z_for_G_update, c, vocab, logdir, epoch)                        
                        # but don't do anything with it; just let it write out all of the audio files
                
                    # Q2 Loss: Update G and Q to better imitate the Q2 model
                    if (i != 0) and track_Q2 and (i % WAVEGAN_Q2_NUPDATES == 0) & (epoch >= Q2_EPOCH_START):
                        
                        if label_stages:
                            print('Starting Q2 evaluation...')                        


                        optimizer_Q2_to_QG.zero_grad() # clear the gradients for the Q update

                        predicted_value_loss = torch.nn.CrossEntropyLoss()                    
                        selected_candidate_wavs = []
                        selected_referents = []
                        selected_Q_estimates = []

                        print('Choosing '+str(Q2_BATCH_SIZE)+' best candidates for each word...')
                        for i in range(NUM_CATEG):
                            
                            num_candidates_to_consider_per_word = 1 # increasing this breaks stuff. Results in considering a larger space
                            # generate a large numver of possible candidates
                            candidate_referents = np.zeros([Q2_BATCH_SIZE*num_candidates_to_consider_per_word, NUM_CATEG+1], dtype=np.float32)
                            candidate_referents[:,i] = 1                            
                            candidate_referents = torch.Tensor(candidate_referents).to(device)
                            _z = torch.FloatTensor(Q2_BATCH_SIZE*num_candidates_to_consider_per_word, 100 - (NUM_CATEG + 1)).uniform_(-1, 1).to(device)

                            # generate new candidate wavs
                            candidate_wavs = G(torch.cat((candidate_referents, _z), dim=1))
                            candidate_Q_estimates = Q(candidate_wavs)

                            # select the Q2_BATCH_SIZE items that are most likely to produce the correct response
                            candidate_predicted_values = torch.Tensor([predicted_value_loss(candidate_Q_estimates[i], candidate_referents[i,0:NUM_CATEG]) for i in range(candidate_referents.shape[0])])                                

                            # order by their predicted score
                            candidate_ordering = torch.argsort(candidate_predicted_values, dim=- 1, descending=False, stable=False)

                            # select a subset of the candidates
                            selected_candidate_wavs.append(torch.narrow(candidate_wavs[candidate_ordering,:], dim=0, start=0, length=Q2_BATCH_SIZE)[:,0].clone())
                            selected_referents.append(torch.narrow(candidate_referents[candidate_ordering,:], dim=0, start=0, length=Q2_BATCH_SIZE).clone())
                            selected_Q_estimates.append(torch.narrow(candidate_Q_estimates[candidate_ordering,:], dim=0, start=0, length=Q2_BATCH_SIZE).clone())


                            del candidate_referents
                            del candidate_wavs
                            del candidate_Q_estimates
                            gc.collect()
                            torch.cuda.empty_cache()
                        

                        print('collapsing candidates')
                        selected_candidate_wavs = torch.vstack(selected_candidate_wavs)
                        selected_referents =  torch.vstack(selected_referents)
                        selected_Q_estimates = torch.vstack(selected_Q_estimates)  


                        print('Recognizing G output with Q2 model...')                        
                        Q2_probs = Q2_cnn(selected_candidate_wavs.unsqueeze(1), Q2) 

                        indices_of_recognized_words, Q2_probs_with_unks  = mark_unks_in_Q2(Q2_probs, Q2_ENTROPY_THRESHOLD, device)

                        print('Finished marking unks')
                        print(indices_of_recognized_words)

                        total_recognized_words = len(indices_of_recognized_words)
                        print('Word recognition complete. Found '+str(len(indices_of_recognized_words))+' of '+str(Q2_BATCH_SIZE*NUM_CATEG)+' words')

                        
                        criterion_Q2 = lambda inpt, target: torch.nn.CrossEntropyLoss()(inpt, target.max(dim=1)[1])


                        if len(indices_of_recognized_words) > 0:
                            
                            #assert(Q2_probs.shape[1] == NUM_CATEG+1)
                            
                            print('Comparing Q predictions to Q2 output')        
                            #Q2_output = torch.from_numpy(Q2_probs.astype(np.float32)).to(device) 

                            # Q_of_selected_candidates is the expected value of each utterance

                            Q_prediction = torch.softmax(selected_Q_estimates, dim=1)
                            zero_tensor = torch.zeros(selected_Q_estimates.shape[0],1).to(device)  # for padding the UNKs, in logit space
                            augmented_Q_prediction = torch.log(torch.hstack((Q_prediction, zero_tensor)) + .0000001)                            
                                    
                            # this is a one shot game for each reference, so implicitly the value before taking the action is 0. I might update this later, i.e., think about this in terms of sequences
                                    
                            # is the Q prediction the same as selected_referents?

                            if not torch.equal(torch.argmax(selected_referents, dim=1), torch.argmax(Q_prediction, dim =1)):
                                print("Child model produced an utterance that they don't think will invoke the correct action. Consider choosing action from a larger set of actions. Disregard if this is early in training and the Q network is not trained yet.")
                   
                                                    
                            # compute the cross entropy between the Q network and the Q2 outputs, which are class labels recovered by the adults                            
                            Q2_loss = criterion_Q2(augmented_Q_prediction[indices_of_recognized_words], Q2_probs_with_unks[indices_of_recognized_words])    
                                                        

                            # count the number of words that Q recovers the same thing as Q2
                            Q_recovers_Q2 = torch.eq(torch.argmax(augmented_Q_prediction[indices_of_recognized_words], dim=1), torch.argmax(Q2_probs_with_unks[indices_of_recognized_words], dim=1)).cpu().numpy().tolist()
                            Q2_recovers_child = torch.eq(torch.argmax(selected_referents[indices_of_recognized_words], dim=1), torch.argmax(Q2_probs_with_unks[indices_of_recognized_words], dim=1)).cpu().numpy().tolist()

                                                                                                                 
                            #this is where we would compute the loss
                            if args.backprop_from_Q2:
                                Q2_loss.backward(retain_graph=True)
                            else:
                                print('Computing Q2 network loss but not backpropagating...')
                                
                            print('Gradients on the Q network:')
                            print('Q layer 0: '+str(np.round(torch.sum(torch.abs(Q.downconv_0.conv.weight.grad)).cpu().numpy(), 10)))
                            print('Q layer 1: '+str(np.round(torch.sum(torch.abs(Q.downconv_1.conv.weight.grad)).cpu().numpy(), 10)))
                            print('Q layer 2: '+str(np.round(torch.sum(torch.abs(Q.downconv_2.conv.weight.grad)).cpu().numpy(), 10)))
                            print('Q layer 3: '+str(np.round(torch.sum(torch.abs(Q.downconv_3.conv.weight.grad)).cpu().numpy(), 10)))
                            print('Q layer 4: '+str(np.round(torch.sum(torch.abs(Q.downconv_4.conv.weight.grad)).cpu().numpy(), 10)))

                            #print('Q2 -> Q update!')
                            #this is where we would do the step
                            if args.backprop_from_Q2:
                                print('Q2 -> Q update!')
                                optimizer_Q2_to_QG.step()
                            optimizer_Q2_to_QG.zero_grad()


                            total_Q_recovers_Q2 = np.sum(Q_recovers_Q2)
                            total_Q2_recovers_child = np.sum(Q2_recovers_child)
                            
                            writer.add_scalar('Loss/Q2 to Q', Q2_loss.detach().item(), step)
                            wandb.log({"Loss/Q2 to Q": Q2_loss.detach().item()}, step=step)



                        else:
                            total_Q_recovers_Q2 = 0
                            total_Q2_recovers_child = 0


                        # proportion of words that Q2 assigns a referent to (ie, what proportion are not unknown)
                        writer.add_scalar('Metric/Proportion Recognized Words Among Total', total_recognized_words / (Q2_BATCH_SIZE *NUM_CATEG), step)
                        wandb.log({"Metric/Proportion Recognized Words Among Total": total_recognized_words / (Q2_BATCH_SIZE *NUM_CATEG)}, step=step)                        
                        
                        # Among those assigned a referent, how often does that agree with what the child intended
                        #writer.add_scalar('Metric/Proportion of Referents Recovered from Q2', total_Q2_recovers_child / total_recognized_words, step)

                        # How often does the Q2 nework get back the right referent
                        writer.add_scalar('Metric/Number of Referents Recovered by Q2', total_Q2_recovers_child, step)
                        wandb.log({"Metric/Number of Referents Recovered by Q2": total_Q2_recovers_child}, step=step)

                        # Among those assigned a referent, how often does that agree with what the child intended
                        #writer.add_scalar('Metric/Proportion that Q recovers from Q2 recognized', total_Q_recovers_Q2 / total_recognized_words, step)

                        # How often does the Q network repliacte the Q2 network
                        writer.add_scalar('Metric/Number of Q2 references replicated by Q', total_Q_recovers_Q2, step)
                        wandb.log({"Metric/Number of Q2 references replicated by Q": total_Q_recovers_Q2}, step=step)


                        
                    if label_stages:
                        print('Q -> G, Q update')

                    c = torch.nn.functional.one_hot(torch.randint(0, NUM_CATEG, (BATCH_SIZE,)),
                             num_classes=NUM_CATEG).to(device)
                    _z = torch.FloatTensor(BATCH_SIZE, 100 - (NUM_CATEG + 1)).uniform_(-1, 1).to(device)
                    zeros = torch.zeros([BATCH_SIZE,1], device = device)
                    z = torch.cat((c, zeros, _z), dim=1)

                    G_z_for_Q_update = G(z) # generate again using the same labels

                    
                    optimizer_Q_to_QG.zero_grad()                        
                    Q_production_loss = criterion_Q(Q(G_z_for_Q_update), c[:,0:NUM_CATEG]) # Note we exclude the UNK label --  child never intends to produce unk
                    Q_production_loss.backward()
                    writer.add_scalar('Loss/Q to G', Q_production_loss.detach().item(), step)
                    wandb.log({"Loss/Q to G": Q_production_loss.detach().item()}, step=step)
                    optimizer_Q_to_QG.step()
                    optimizer_Q_to_QG.zero_grad()

                    
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
                torch.save(optimizer_Q_to_QG.state_dict(), os.path.join(logdir, f'epoch{epoch}_step{step}_Q_to_Gopt.pt'))            
            if track_Q2 and optimizer_Q2_to_QG is not None:
                torch.save(optimizer_Q2_to_QG.state_dict(), os.path.join(logdir, f'epoch{epoch}_step{step}_Q_to_Q2opt.pt'))
            

            if args.keep_intermediate_epochs:
                pass
            else:
                if ('last_path_prefix' in locals()) or ('last_path_prefix' in globals()):
                    os.system('rm '+last_path_prefix)

            last_path_prefix = os.path.join(logdir, 'epoch'+str(epoch)+'_step'+str(step)+'*')
