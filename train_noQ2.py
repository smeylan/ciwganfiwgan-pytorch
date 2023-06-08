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
import gc
from varname import nameof

torch.autograd.set_detect_anomaly(True)

class AudioDataSet:
    def __init__(self, datadir, slice_len, NUM_CATEG, timit_words):
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

def write_out_wavs(G_z_2d, labels, timit_words, epoch):
    # returns probabilities and a set of indices; takes a smaller number of arguments
    files_for_asr = []
    epoch_path = os.path.join("temp",str(epoch))
    if not os.path.exists(epoch_path):
        os.makedirs(epoch_path)    

    labels_local = labels.cpu().detach().numpy()
    # for each file in the batch, write out a wavfile
    for j in range(G_z_2d.shape[0]):
        audio_buffer = G_z_2d[j,:].detach().cpu().numpy()          
        true_word = timit_words[np.argwhere(labels_local[j,:])[0][0]]
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


    # Parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datadir = args.datadir
    logdir = args.logdir
    
    # Epochs and Intervals
    NUM_EPOCHS = args.num_epochs
    WAVEGAN_DISC_NUPDATES = 5
    WAVEGAN_Q2_NUPDATES = 10
    Q2_EPOCH_START = 0
    WAV_OUTPUT_N = 5
    SAVE_INT = args.save_int
    PRODUCTION_START_EPOCH = 0
    COMPREHENSION_INTERVAL = 100000


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

    # Q2 parameters
    NUM_Q2_TRAINING_EPOCHS = 25
    Q2_BATCH_SIZE = 6
    Q2_ENTROPY_THRESHOLD = 1000


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
            #optimizer_Q_to_G = optim.RMSprop(G.parameters(), lr=LEARNING_RATE)
            optimizer_Q_to_QG = optim.RMSprop(it.chain(G.parameters(), Q.parameters()), lr=LEARNING_RATE)            
            # just update the G parameters
            if args.Q2:
                Q2 = WaveGANQNetwork(slice_len=SLICE_LEN, num_categ=NUM_CATEG).to(device).train()
                optimizer_Q2_to_QG = optim.RMSprop(it.chain(G.parameters(), Q.parameters()), lr=LEARNING_RATE)
                optimizer_Q2_to_Q2 = optim.RMSprop(Q2.parameters(), lr=LEARNING_RATE)

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
            

        return G, D, optimizer_G, optimizer_D, Q, Q2, optimizer_Q_to_QG, optimizer_Q2_to_QG, optimizer_Q2_to_Q2, criterion_Q

    # Load models
    G, D, optimizer_G, optimizer_D, Q, Q2, optimizer_Q_to_QG, optimizer_Q2_to_QG, optimizer_Q2_to_Q2, criterion_Q = make_new()
        
    
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

            if train_Q2:
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
                incdices_of_recognized_words, adult_recovers_from_adult = Q2_cnn(reals, Q2)    
                Q2_comprehension_loss = criterion_Q(adult_recovers_from_adult, labels[:,0:NUM_CATEG]) # Note we exclude the UNK label --  child never intends to produce unk
                Q2_comprehension_loss.backward()
                writer.add_scalar('Loss/Q2 to Q2', Q_comprehension_loss.detach().item(), step)
                optimizer_Q2_to_Q2.step()
                step += 1
        torch.save(Q2, 'saved_networks/adult_pretrained_Q_network.torch')

    else:
        print('Loading a Previous Adult Q2 CNN Network')
        Q2 = torch.load('saved_networks/adult_pretrained_Q_network.torch')

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
                        c = torch.nn.functional.one_hot(torch.randint(0, NUM_CATEG, (BATCH_SIZE,)),
                                                         num_classes=NUM_CATEG).to(device)
                    
                    z = torch.cat((labels, _z), dim=1)
                    #z = torch.cat((c, zeros, _z), dim=1)
                else:
                    raise NotImplementedError
                    z = _z

                fakes = G(z)

                # shuffle the reals so that the matched item for discrim is not necessarily from the same referent                
                shuffled_reals = reals[torch.randperm(reals.shape[0]),:,:]
                
                penalty = gradient_penalty(G, D, shuffled_reals, fakes, epsilon)
                D_loss = torch.mean(D(fakes) - D(shuffled_reals) + LAMBDA * penalty)
                writer.add_scalar('Loss/D', D_loss.detach().item(), step)
                D_loss.backward()
                if label_stages:
                    print('Discriminator update!')
                optimizer_D.step()            
                optimizer_D.zero_grad()

                if i % WAVEGAN_DISC_NUPDATES == 0:
                    optimizer_G.zero_grad()
                                                                
                    _z = torch.FloatTensor(BATCH_SIZE, 100 - (NUM_CATEG + 1)).uniform_(-1, 1).to(device)


                    if train_Q:
                        #optimizer_Q_to_QG.zero_grad()
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

                    G_z = G(z) # generate again using the same labels

                    # G Loss
                    G_loss = torch.mean(-D(G_z))
                    G_loss.backward(retain_graph=True)
                    # Update
                    optimizer_G.step()
                    optimizer_G.zero_grad()
                    if label_stages:
                        print('Generator update!')
                    writer.add_scalar('Loss/G', G_loss.detach().item(), step)


                    if (epoch % WAV_OUTPUT_N == 0) & (i <= 1):
                         
                        print('Sampling .wav outputs (but not running them through Q2)...')
                        write_out_wavs(G_z, labels, timit_words, epoch)                        
                        # but don't do anything with it; just let it write out all of the audio files

                    # Q2 Loss: Update G and Q to better imitate the Q2 model
                    
                        
                    if label_stages:
                        print('Q -> G update')
                    
                    optimizer_Q_to_QG.zero_grad()                        
                    Q_production_loss = criterion_Q(Q(G(z)), c[:,0:NUM_CATEG]) # Note we exclude the UNK label --  child never intends to produce unk
                    Q_production_loss.backward()
                    writer.add_scalar('Loss/Q to G', Q_production_loss.detach().item(), step)
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
            if train_Q2 and optimizer_Q2_to_QG is not None:
                torch.save(optimizer_Q2_to_QG.state_dict(), os.path.join(logdir, f'epoch{epoch}_step{step}_Q_to_Q2opt.pt'))
