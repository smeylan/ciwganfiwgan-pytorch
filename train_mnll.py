import argparse
import os
import re
import itertools as it
import sys

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from scipy.io.wavfile import read, write
import scipy.stats
from torch.utils.data import DataLoader
from tqdm import tqdm
import random

from infowavegan import WaveGANGenerator, WaveGANDiscriminator, WaveGANQNetwork

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
    def __init__(self, datadir, slice_len, NUM_CATEG, vocab, word_means=None, sigma=None):
        print("Loading data")
        dir = os.listdir(datadir)
        x = np.zeros((len(dir), 1, slice_len))
        y = np.zeros((len(dir), NUM_CATEG)) 
 
        i = 0
        files = []
        categ_labels = []
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
            categ_labels.append(j)
            i += 1

        categ_labels = np.array(categ_labels)

        if word_means is not None:
            # condition on the y values to choose real-valued positions in the semantic space associated with each of the words
            # implicitly, we could have generated these in the semantic space first, and run a classifier to get y values
            sem_vector_store = {} 
            for i in range(NUM_CATEG):
                sem_vector_store[i] = scipy.stats.multivariate_normal.rvs( mean=word_means[i], cov=sigma, size = len(files))

            # the category label i indexes the key in sem_vector_store, j indexes the row (ie many rows are not used)                 
            sem_vector =  [sem_vector_store[i][j,:] for i,j in zip(categ_labels, range(len(files)))]   
            self.sem_vector = torch.from_numpy(np.array(sem_vector, dtype=np.float32))            

        self.len = len(x)
        self.audio = torch.from_numpy(np.array(x, dtype=np.float32))
        self.labels = torch.from_numpy(np.array(y, dtype=np.float32))
        

    def __getitem__(self, index):
        if hasattr(self,'sem_vector'):
            return ((self.audio[index], self.labels[index], self.sem_vector[index]))
        else:
            return ((self.audio[index], self.labels[index]))

    def __len__(self):
        return self.len

def get_architecture_appropriate_c(architecture, num_categ, batch_size):
    if ARCHITECTURE == 'ciwgan':
        c = torch.nn.functional.one_hot(torch.randint(0, num_categ, (batch_size,)),
                                                num_classes=num_categ).to(device)
    elif ARCHITECTURE == 'fiwgan':
        c = torch.FloatTensor(batch_size, num_categ).bernoulli_().to(device)
    else:
        assert False, "Architecture not recognized."
    return c

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


def Q2_cnn(selected_candidate_wavs, Q2, architecture):
    print('in Q2_cnn')
    if ARCHITECTURE == "ciwgan":
        Q2_probs = torch.softmax(Q2(selected_candidate_wavs), dim=1)
        return Q2_probs
    elif ARCHITECTURE == "eiwgan":
        # this directly returns the embeddings (of the dimensionsionality NUM_DIMS)
        return(Q2(selected_candidate_wavs))
    elif ARCHITECTURE == "fiwgan":
        Q2_binary = torch.sigmoid(Q2(selected_candidate_wavs))
        return Q2_binary
    else:
        raise ValueError('architecture for Q2_cnn must be one of (ciwgan, eiwgan, fiwgan)')


def write_out_wavs(architecture, G_z_2d, labels, vocab, logdir, epoch):
    # returns probabilities and a set of indices; takes a smaller number of arguments
    files_for_asr = []
    epoch_path = os.path.join(logdir,'audio_files',str(epoch))
    if not os.path.exists(epoch_path):
        os.makedirs(epoch_path)    

    labels_local = labels.cpu().detach().numpy()
    # for each file in the batch, write out a wavfile
    for j in range(G_z_2d.shape[0]):
        audio_buffer = G_z_2d[j,:].detach().cpu().numpy()
        if architecture == 'ciwgan':
            true_word = vocab[np.argwhere(labels_local[j,:])[0][0]]
        else:
            true_word = ''
        tf = os.path.join(epoch_path,true_word + '_' + str(uuid.uuid4())+".wav")
        write(tf, 16000, audio_buffer[0])
        files_for_asr.append(copy.copy(tf))
    return(files_for_asr)


def one_hot_classify_sem_vector(Q2_sem_vecs, word_means):
    print('do a one_hot classification of the sem_vecs: find the closest word mean for each')

    word_means = torch.from_numpy(np.array(word_means)).to(device)     
    
    dists = [] 
    for x in range(word_means.shape[0]):
        dists.append(torch.sqrt(torch.sum((Q2_sem_vecs - word_means[x]) ** 2, dim=1)))    
    distances = torch.vstack(dists)
    
    best = torch.argmin(distances,0)
    return(best)

class EuclideanLoss(nn.Module):
    def __init__(self):
        super(EuclideanLoss, self).__init__()

    def forward(self, inputs, targets):
        return torch.sqrt(torch.sum((inputs - targets) ** 2, dim=1))


if __name__ == "__main__":
    # Training Arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--architecture',
        type=str,
        required=True,
        help='Architecure. Can be ciwgan for fiwgan'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        required=True,
        help='Log/Results Directory. Results will be stored by wandb_group / wandb_name / wandb_id (see below)'
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
        action='store_true',
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
        '--wandb_name',
        type=str,
        help='Name of this specific run',        
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

    parser.add_argument(
        '--q2_noise_probability',
        type=float,
        help="Probability that the action taken by Q2 is affected by noise and does not match Q's referent",
        default=0
    )

    args = parser.parse_args()
    train_Q = True
    track_Q2 = bool(args.track_q2)
    vocab = args.vocab.split()

    ARCHITECTURE = args.architecture
    
    if args.q2_noise_probability > 0:
        assert ARCHITECTURE != 'eiwgan', "Eiwgan does not have noise probability support."
    
    # Parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datadir = args.data_dir

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
   
    gpu_properties = torch.cuda.get_device_properties('cuda')
    kwargs = {
       'project' :  args.wandb_project,        
       'config' : args.__dict__,
       'group' : args.wandb_group,
       'name' : args.wandb_name,
       'config': {
            'slurm_job_id' : os.getenv('SLURM_JOB_ID'),
            'gpu_name' : gpu_properties.name,
            'gpu_memory' : gpu_properties.total_memory
        }
    }
    wandb.init(**kwargs)

    logdir = os.path.join(args.log_dir, args.wandb_group, args.wandb_name, wandb.run.id)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    if ARCHITECTURE == "eiwgan":
        if NUM_CATEG != 4:
            raise ValueError('NUM_CATEG must be 4 for hard-coded word means in Eiwgan. Make this variable later.')
        word_means = [[-.5,-.5],[-.5,.5],[.5,-.5],[.5,.5]]
        NUM_DIM = len(word_means[0])
        sigma = np.matrix([[.025,0],[0,.025]])
        dataset = AudioDataSet(datadir, SLICE_LEN, NUM_CATEG, vocab, word_means, sigma)
    else:
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

        Q, optimizer_Q_to_G, optimizer_Q_to_Q, criterion_Q, criterion_Q2 = (None, None, None, None, None)
        if train_Q:
            if args.architecture in ('ciwgan','fiwgan'):
                Q = WaveGANQNetwork(slice_len=SLICE_LEN, num_categ=NUM_CATEG).to(device).train()
            elif args.architecture == 'eiwgan':
                Q = WaveGANQNetwork(slice_len=SLICE_LEN, num_categ=NUM_DIM).to(device).train()
                # number of dimensions in the sematnic space, not number of words
            else:
                raise ValueError('Architecure not recognized! Must be fiwgan or ciwgan')

            #optimizer_Q_to_G = optim.RMSprop(G.parameters(), lr=LEARNING_RATE)
            optimizer_Q_to_QG = optim.RMSprop(it.chain(G.parameters(), Q.parameters()), lr=LEARNING_RATE)            
            # just update the G parameters            

            if args.architecture == 'fiwgan':
                print("Training a fiwGAN with ", NUM_CATEG, " categories.")
                criterion_Q = torch.nn.BCEWithLogitsLoss() # binary cross entropy                
            elif args.architecture == 'eiwgan':
                print("Training a eiwGAN with ", NUM_CATEG, " categories.")
                criterion_Q = EuclideanLoss()

            elif args.architecture == 'ciwgan':
                print("Training a ciwGAN with ", NUM_CATEG, " categories.")
                # NOTE: one hot -> category nr. transformation
                # CE loss needs logit, category -> loss
                criterion_Q = lambda inpt, target: torch.nn.CrossEntropyLoss()(inpt, target.max(dim=1)[1])
            else:
                raise ValueError('Architecure not recognized! Must be fiwgan or ciwgan')                


        if track_Q2:
            if ARCHITECTURE in ("ciwgan","fiwgan"):
                Q2 = WaveGANQNetwork(slice_len=SLICE_LEN, num_categ=NUM_CATEG).to(device).train()
            elif ARCHITECTURE == "eiwgan":
                Q2 = WaveGANQNetwork(slice_len=SLICE_LEN, num_categ=NUM_DIM).to(device).train()
            

            optimizer_Q2_to_QG = optim.RMSprop(it.chain(G.parameters(), Q.parameters()), lr=LEARNING_RATE)
            optimizer_Q2_to_Q2 = optim.RMSprop(Q2.parameters(), lr=LEARNING_RATE)
            
            if ARCHITECTURE == 'fiwgan':
                criterion_Q2 = torch.nn.BCEWithLogitsLoss() # binary cross entropy

            elif ARCHITECTURE == "eiwgan":
                criterion_Q2 = EuclideanLoss()            
            
            elif ARCHITECTURE == "ciwgan":
                criterion_Q2 = lambda inpt, target: torch.nn.CrossEntropyLoss()(inpt, target.max(dim=1)[1])

            else:
                raise NotImplementedError        

        return G, D, optimizer_G, optimizer_D, Q, Q2, optimizer_Q_to_QG, optimizer_Q2_to_QG, optimizer_Q2_to_Q2, criterion_Q, criterion_Q2

    # Load models
    G, D, optimizer_G, optimizer_D, Q, Q2, optimizer_Q_to_QG, optimizer_Q2_to_QG, optimizer_Q2_to_Q2, criterion_Q, criterion_Q2 = make_new()
        
    
    start_epoch = 0
    start_step = 0

    print("Starting a new training")


    step = start_step


    regenerate_Q2 = True
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
                if ARCHITECTURE == 'eiwgan':  
                    continuous_labels = trial[2].to(device)
                optimizer_Q2_to_Q2.zero_grad()
                Q2_logits = Q2(reals)    
                
                if ARCHITECTURE in ("ciwgan", "fiwgan"):
                    Q2_comprehension_loss = criterion_Q(Q2_logits, labels)
                elif ARCHITECTURE == "eiwgan":                    
                    Q2_comprehension_loss = torch.mean(criterion_Q(Q2_logits, continuous_labels))

                Q2_comprehension_loss.backward()

                wandb.log({"Loss/Q2 to Q2": Q2_comprehension_loss.detach().item()}, step=step)
                optimizer_Q2_to_Q2.step()
                step += 1
        torch.save(Q2, 'saved_networks/adult_pretrained_Q_network_'+str(NUM_CATEG)+'_'+ARCHITECTURE+'.torch')

    else:
        print('Loading a Previous Adult Q2 CNN Network')
        Q2 = torch.load('saved_networks/adult_pretrained_Q_network_'+str(NUM_CATEG)+'_'+ARCHITECTURE+'.torch')

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
                if ARCHITECTURE == 'eiwgan':
                    raise NotImplementedError
                
                if label_stages:
                    print('Updating Child Q network to identify referents')

                optimizer_Q_to_Q.zero_grad()
                child_recovers_from_adult = Q(reals)    
                Q_comprehension_loss = criterion_Q(child_recovers_from_adult, labels)
                Q_comprehension_loss.backward()
                wandb.log({"Loss/Q to Q": Q_comprehension_loss.detach().item()}, step=step)
                optimizer_Q_to_Q.step()
                step += 1
                            

            else:
                # Discriminator Update
                optimizer_D.zero_grad()                 

                epsilon = torch.rand(BATCH_SIZE, 1, 1).repeat(1, 1, SLICE_LEN).to(device)
                
                if ARCHITECTURE == "eiwgan":
                    # draw from the semantic space a c that will need to be encoded
                
                    words = torch.nn.functional.one_hot(torch.randint(0, NUM_CATEG, (BATCH_SIZE,)),
                             num_classes=NUM_CATEG).detach().numpy() # randomly generate a bunch of one-hots
                    word_indices = [x[1] for x in np.argwhere(words)]                                                                    

                    # pre-draw the semantic vectors
                    sem_vector_store = []
                    for categ_index in range(NUM_CATEG):                            
                        sem_vector_store.append(scipy.stats.multivariate_normal.rvs( mean=word_means[categ_index], cov=sigma, size=BATCH_SIZE))

                    # draw a c from the pre-drawn params
                    # draw the jth item from the ith word (many will not get used)
                    c =  torch.from_numpy(np.vstack([sem_vector_store[i][j,:] for i,j in zip(word_indices, range(BATCH_SIZE))]).astype(np.float32)).to(device)                         
                    _z = torch.FloatTensor(BATCH_SIZE, 100 - NUM_DIM).uniform_(-1, 1).to(device)                    
                    z = torch.cat((c, _z), dim=1)

                elif ARCHITECTURE in {'ciwgan', 'fiwgan'}: 
                    c = get_architecture_appropriate_c(ARCHITECTURE, NUM_CATEG, BATCH_SIZE)
                    _z = torch.FloatTensor(BATCH_SIZE, 100 - (NUM_CATEG)).uniform_(-1, 1).to(device)
                    z = torch.cat((c, _z), dim=1)
                    
                fakes = G(z)
             
                # shuffle the reals so that the matched item for discrim is not necessarily from the same referent                
                shuffled_reals = reals[torch.randperm(reals.shape[0]),:,:]
                
                penalty = gradient_penalty(G, D, shuffled_reals, fakes, epsilon)
                D_loss = torch.mean(D(fakes) - D(shuffled_reals) + LAMBDA * penalty)
                
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
                    
                    if ARCHITECTURE == "eiwgan":
                        # draw from the semantic space a c that will need to be encoded
                
                        words = torch.nn.functional.one_hot(torch.randint(0, NUM_CATEG, (BATCH_SIZE,)),
                             num_classes=NUM_CATEG).detach().numpy() # randomly generate a bunch of one-hots
                        word_indices = [x[1] for x in np.argwhere(words)]                                                                    

                        # pre-draw the semantic vectors
                        sem_vector_store = []
                        for categ_index in range(NUM_CATEG):                            
                            sem_vector_store.append(scipy.stats.multivariate_normal.rvs( mean=word_means[categ_index], cov=sigma, size=BATCH_SIZE))

                        # draw a c from the pre-drawn params
                        # draw the jth item from the ith word (many will not get used)
                        c =  torch.from_numpy(np.vstack([sem_vector_store[i][j,:] for i,j in zip(word_indices, range(BATCH_SIZE))]).astype(np.float32)).to(device)                         
                        _z = torch.FloatTensor(BATCH_SIZE, 100 - NUM_DIM).uniform_(-1, 1).to(device)                    
                        z = torch.cat((c, _z), dim=1)
                    elif ARCHITECTURE in {'ciwgan', 'fiwgan'}:
                        c = get_architecture_appropriate_c(ARCHITECTURE, NUM_CATEG, BATCH_SIZE)
                        _z = torch.FloatTensor(BATCH_SIZE, 100 - NUM_CATEG).uniform_(-1, 1).to(device)
                        z = torch.cat((c, _z), dim=1)

                    G_z_for_G_update = G(z) # generate again using the same labels

                    # G Loss
                    G_loss = torch.mean(-D(G_z_for_G_update))
                    G_loss.backward(retain_graph=True)
                    # Update
                    optimizer_G.step()
                    optimizer_G.zero_grad()
                    if label_stages:
                        print('Generator update!')
                    wandb.log({"Loss/G": G_loss.detach().item()}, step=step)


                    if (epoch % WAV_OUTPUT_N == 0) & (i <= 1):
                        
                        print('Sampling .wav outputs (but not running them through Q2)...')
                        as_words = torch.from_numpy(words).to(device) if ARCHITECTURE == 'eiwgan' else c
                        write_out_wavs(ARCHITECTURE, G_z_for_G_update, as_words, vocab, logdir, epoch)                        
                        # but don't do anything with it; just let it write out all of the audio files
                
                    # Q2 Loss: Update G and Q to better imitate the Q2 model
                    if (i != 0) and track_Q2 and (i % WAVEGAN_Q2_NUPDATES == 0) & (epoch >= Q2_EPOCH_START):
                        
                        if label_stages:
                            print('Starting Q2 evaluation...')                        

                        optimizer_Q2_to_QG.zero_grad() # clear the gradients for the Q update

                        selected_candidate_wavs = []  
                        mixed_selected_candidate_wavs = []                      
                        selected_Q_estimates = []
                        mixed_selected_Q_estimates = []

                        print('Choosing '+str(Q2_BATCH_SIZE)+' best candidates for each word...')

                        if ARCHITECTURE in {'ciwgan', 'fiwgan'}:
                            if ARCHITECTURE == 'ciwgan':
                                predicted_value_loss = torch.nn.CrossEntropyLoss()
                            else:
                                predicted_value_loss = torch.nn.BCEWithLogitsLoss()

                            selected_referents = []
                            for categ_index in range(NUM_CATEG):
                                
                                num_candidates_to_consider_per_word = 1 # increasing this breaks stuff. Results in considering a larger space

                                def mask_for_random_vectors(original_label, vocab_size, batch_size, q2_noise_probability):
                                    # Get random vectors and prepare for mask
                                    random_labels = torch.from_numpy(np.random.choice(np.arange(vocab_size), size = (batch_size,)))
                                    mask_by_batch_dim = torch.from_numpy(np.random.choice([0, 1], size = (batch_size,), p = [1 - q2_noise_probability, q2_noise_probability]))
                                    labels = torch.where(mask_by_batch_dim == 1, random_labels, original_label)
                                    onehot_per_word = F.one_hot(labels, num_classes = vocab_size)    
                                    assert len(onehot_per_word.shape) == 2
                                    candidate_referents = onehot_per_word.unsqueeze(1).repeat((1, num_candidates_to_consider_per_word, 1)).reshape(-1, onehot_per_word.shape[-1])
                                    return candidate_referents

                                # generate a large numver of possible candidates
                                candidate_referents = np.zeros([Q2_BATCH_SIZE*num_candidates_to_consider_per_word, NUM_CATEG], dtype=np.float32)
                                candidate_referents[:,categ_index] = 1     
                                candidate_referents = torch.Tensor(candidate_referents).to(device)
                                _z = torch.FloatTensor(Q2_BATCH_SIZE*num_candidates_to_consider_per_word, 100 - (NUM_CATEG)).uniform_(-1, 1).to(device)

                                # possible candidated but mixed
                                mixed_candidate_referents = mask_for_random_vectors(categ_index, len(vocab), Q2_BATCH_SIZE, args.q2_noise_probability).to(device)

                                # generate new candidate wavs
                                candidate_wavs = G(torch.cat((candidate_referents, _z), dim=1))
                                candidate_Q_estimates = Q(candidate_wavs)

                                # generate new mixed candidate wavs
                                mixed_candidate_wavs = G(torch.cat((mixed_candidate_referents, _z), dim=1))
                                mixed_candidate_Q_estimates = Q(mixed_candidate_wavs)


                                # select the Q2_BATCH_SIZE items that are most likely to produce the correct response
                                candidate_predicted_values = torch.Tensor([predicted_value_loss(candidate_Q_estimates[i], candidate_referents[i]) for i in range(candidate_referents.shape[0])])                                
                                # order by their predicted score
                                candidate_ordering = torch.argsort(candidate_predicted_values, dim=- 1, descending=False, stable=False)

                                # select a subset of the candidates
                                selected_candidate_wavs.append(torch.narrow(candidate_wavs[candidate_ordering,:], dim=0, start=0, length=Q2_BATCH_SIZE)[:,0].clone())
                                selected_referents.append(torch.narrow(candidate_referents[candidate_ordering,:], dim=0, start=0, length=Q2_BATCH_SIZE).clone())
                                selected_Q_estimates.append(torch.narrow(candidate_Q_estimates[candidate_ordering,:], dim=0, start=0, length=Q2_BATCH_SIZE).clone())

                                # select mixed candidates that correspond to that subset
                                mixed_selected_candidate_wavs.append(torch.narrow(mixed_candidate_wavs[candidate_ordering,:], dim=0, start=0, length=Q2_BATCH_SIZE)[:,0].clone())
                                mixed_selected_Q_estimates.append(torch.narrow(mixed_candidate_Q_estimates[candidate_ordering,:], dim=0, start=0, length=Q2_BATCH_SIZE).clone())

                                del candidate_referents
                                del candidate_wavs
                                del candidate_Q_estimates
                                gc.collect()
                                torch.cuda.empty_cache()
                            

                            print('collapsing candidates')
                            selected_candidate_wavs = torch.vstack(selected_candidate_wavs)
                            selected_referents =  torch.vstack(selected_referents)
                            selected_Q_estimates = torch.vstack(selected_Q_estimates)  

                            mixed_selected_candidate_wavs = torch.vstack(mixed_selected_candidate_wavs)
                            mixed_selected_Q_estimates = torch.vstack(mixed_selected_Q_estimates)
                        elif ARCHITECTURE == 'eiwgan':                            
                            
                            selected_meanings = []
                            selected_referents = []
                            for categ_index in range(NUM_CATEG):
                                
                                # increasing this breaks stuff. Results in considering a larger space
                                num_candidates_to_consider_per_word = 1 

                                # propagae the categorical label associated with the Gaussian for checking what Q2 infers
                                candidate_referents = np.zeros([Q2_BATCH_SIZE*num_candidates_to_consider_per_word, NUM_CATEG], dtype=np.float32)
                                candidate_referents[:,categ_index] = 1
                                candidate_referents = torch.Tensor(candidate_referents).to(device)

                                c = scipy.stats.multivariate_normal.rvs(mean=word_means[categ_index], cov=sigma, size=Q2_BATCH_SIZE * num_candidates_to_consider_per_word)
                                
                                #candidate_meanings rather than candidate references                                
                                candidate_meanings = torch.Tensor(c).to(device)
                                _z = torch.FloatTensor(Q2_BATCH_SIZE*num_candidates_to_consider_per_word, 100 - NUM_DIM).uniform_(-1, 1).to(device)

                                # generate new candidate wavs
                                candidate_wavs = G(torch.cat((candidate_meanings, _z), dim=1))
                                candidate_Q_estimates = Q(candidate_wavs)

                                # select the Q2_BATCH_SIZE items that are most likely to produce the correct response -- those that have the smallest distance under the model

                                # compute the distances
                                candidate_predicted_values = criterion_Q2(candidate_Q_estimates, candidate_meanings)

                                # order by their predicted score
                                candidate_ordering = torch.argsort(candidate_predicted_values, dim=- 1, descending=True, stable=False)

                                # select a subset of the candidates
                                selected_candidate_wavs.append(torch.narrow(candidate_wavs[candidate_ordering,:], dim=0, start=0, length=Q2_BATCH_SIZE)[:,0].clone())
                                selected_meanings.append(torch.narrow(candidate_meanings[candidate_ordering,:], dim=0, start=0, length=Q2_BATCH_SIZE).clone())
                                selected_referents.append(torch.narrow(candidate_referents[candidate_ordering,:], dim=0, start=0, length=Q2_BATCH_SIZE).clone())
                                selected_Q_estimates.append(torch.narrow(candidate_Q_estimates[candidate_ordering,:], dim=0, start=0, length=Q2_BATCH_SIZE).clone())


                                del candidate_meanings
                                del candidate_referents
                                del candidate_wavs
                                del candidate_Q_estimates
                                gc.collect()
                                torch.cuda.empty_cache()
                            

                            print('collapsing candidates')
                            selected_candidate_wavs = torch.vstack(selected_candidate_wavs)
                            selected_meanings =  torch.vstack(selected_meanings)
                            selected_referents = torch.vstack(selected_referents)
                            selected_Q_estimates = torch.vstack(selected_Q_estimates)  

                        print('Recognizing G output with Q2 model...')                        
                        if ARCHITECTURE in {"ciwgan", "fiwgan"}:
                            with torch.no_grad():
                                Q2_probs = Q2_cnn(selected_candidate_wavs.unsqueeze(1), Q2, ARCHITECTURE)
                            mixed_Q2_probs = Q2_cnn(mixed_selected_candidate_wavs.unsqueeze(1), Q2, ARCHITECTURE)
                        elif ARCHITECTURE == "eiwgan":

                            Q2_sem_vecs = Q2_cnn(selected_candidate_wavs.unsqueeze(1), Q2, ARCHITECTURE) 
                            
                        #assert(Q2_probs.shape[1] == NUM_CATEG+1)
                        
                        print('Comparing Q predictions to Q2 output')        
                        #Q2_output = torch.from_numpy(Q2_probs.astype(np.float32)).to(device) 

                        # Q_of_selected_candidates is the expected value of each utterance

                        Q_prediction = torch.softmax(selected_Q_estimates, dim=1)
                        if ARCHITECTURE in ('ciwgan', 'fiwgan'):
                            mixed_Q_prediction = torch.softmax(mixed_selected_Q_estimates, dim=1)
                                
                        # this is a one shot game for each reference, so implicitly the value before taking the action is 0. I might update this later, i.e., think about this in terms of sequences                   
                                                
                        # compute the cross entropy between the Q network and the Q2 outputs, which are class labels recovered by the adults                                                    
                        if ARCHITECTURE in {'ciwgan', 'fiwgan'}:    
                            augmented_Q_prediction = torch.log(Q_prediction + .0000001)   
                            mixed_augmented_Q_prediction = torch.log(mixed_Q_prediction + .0000001)                          
                            
                            mixed_Q2_loss = criterion_Q2(mixed_augmented_Q_prediction, mixed_Q2_probs)   
                            with torch.no_grad():
                                Q2_loss = criterion_Q2(augmented_Q_prediction, Q2_probs)   

                            if ARCHITECTURE == 'ciwgan':
                                # is the Q prediction the same as selected_referents?                            
                                if not torch.equal(torch.argmax(selected_referents, dim=1), torch.argmax(Q_prediction, dim =1)):
                                    print("Child model produced an utterance that they don't think will invoke the correct action. Consider choosing action from a larger set of actions. Disregard if this is early in training and the Q network is not trained yet.")

                                # count the number of words that Q recovers the same thing as Q2
                                Q_recovers_Q2 = torch.eq(torch.argmax(augmented_Q_prediction, dim=1), torch.argmax(Q2_probs, dim=1)).cpu().numpy().tolist()
                                Q2_recovers_child = torch.eq(torch.argmax(selected_referents, dim=1), torch.argmax(Q2_probs, dim=1)).cpu().numpy().tolist()
                            else:

                                def count_binary_vector_matches(raw_set1, raw_set2):
                                    matches_mask = []
                                    threshold = 0.5
                                    binarize = lambda vector : (vector >= threshold).int()
                                    set1, set2 = binarize(raw_set1), binarize(raw_set2)
                                    assert set1.shape == set2.shape
                                    for i in range(set1.shape[0]):
                                        match_int = 1 if torch.all(set1[i] == set2[i]) else 0
                                        matches_mask.append(match_int)
                                    return matches_mask

                                Q_recovers_Q2 = count_binary_vector_matches(augmented_Q_prediction, Q2_probs)
                                Q2_recovers_child = count_binary_vector_matches(selected_referents, Q2_probs)
                        
                        elif ARCHITECTURE == 'eiwgan':                                    
                            Q2_loss = torch.mean(criterion_Q2(Q_prediction, Q2_sem_vecs))                                

                            print('Check if we recover the one-hot that was used to draw the continuously valued vector')                                
                            Q2_recovers_child = torch.eq(torch.argmax(selected_referents, dim=1), one_hot_classify_sem_vector(Q2_sem_vecs, word_means)).cpu().numpy().tolist()                                
                                                                                                                
                        #this is where we would compute the loss
                        if args.backprop_from_Q2:
                            if ARCHITECTURE in ('ciwgan', 'fiwgan'):
                                mixed_Q2_loss.backward(retain_graph=True)
                            elif ARCHITECTURE == 'eiwgan':
                                Q2_loss.backward(retain_graph=True)
                            else:
                                raise NotImplementedError
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

                        total_Q2_recovers_child = np.sum(Q2_recovers_child)

                        if ARCHITECTURE in ('ciwgan', 'fiwgan'):
                            total_Q_recovers_Q2 = np.sum(Q_recovers_Q2)                                                            

                        wandb.log({"Loss/Q2 to Q": Q2_loss.detach().item()}, step=step)
                        
                        wandb.log({"Metric/Number of Referents Recovered by Q2": total_Q2_recovers_child}, step=step)

                        if ARCHITECTURE  in ('ciwgan', 'fiwgan'):
                            # How often does the Q network repliacte the Q2 network
                            wandb.log({"Metric/Number of Q2 references replicated by Q": total_Q_recovers_Q2}, step=step)
                        
                   
                    if label_stages:
                        print('Q -> G, Q update')

                    if ARCHITECTURE in ("ciwgan", "fiwgan"):
                        c = get_architecture_appropriate_c(ARCHITECTURE, NUM_CATEG, BATCH_SIZE)
                        _z = torch.FloatTensor(BATCH_SIZE, 100 - NUM_CATEG).uniform_(-1, 1).to(device)
                        z = torch.cat((c, _z), dim=1)
                    
                    elif ARCHITECTURE == "eiwgan":
                        # draw from the semantic space a c that will need to be encoded
                
                        words = torch.nn.functional.one_hot(torch.randint(0, NUM_CATEG, (BATCH_SIZE,)),
                             num_classes=NUM_CATEG).detach().numpy() # randomly generate a bunch of one-hots
                        word_indices = [x[1] for x in np.argwhere(words)]                                                                    

                        # pre-draw the semantic vectors
                        sem_vector_store = []
                        for categ_index in range(NUM_CATEG):                            
                            sem_vector_store.append(scipy.stats.multivariate_normal.rvs( mean=word_means[categ_index], cov=sigma, size=BATCH_SIZE))

                        # draw a c from the pre-drawn params
                        # draw the jth item from the ith word (many will not get used)
                        c =  torch.from_numpy(np.vstack([sem_vector_store[i][j,:] for i,j in zip(word_indices, range(BATCH_SIZE))]).astype(np.float32)).to(device)                         
                        _z = torch.FloatTensor(BATCH_SIZE, 100 - NUM_DIM).uniform_(-1, 1).to(device)                    
                        z = torch.cat((c, _z), dim=1)                            
                    G_z_for_Q_update = G(z) # generate again using the same labels

                    optimizer_Q_to_QG.zero_grad()
                    if ARCHITECTURE == "eiwgan":
                        
                        Q_production_loss = torch.mean(criterion_Q(Q(G_z_for_Q_update), c))
                        # distance in the semantic space between what the child expects the adult to revover and what the child actually does

                    elif ARCHITECTURE in {"ciwgan", "fiwgan"}:
                                                
                        Q_production_loss = criterion_Q(Q(G_z_for_Q_update), c)

                    Q_production_loss.backward()
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
            if train_Q and track_Q2 and optimizer_Q2_to_QG is not None:
                torch.save(optimizer_Q2_to_QG.state_dict(), os.path.join(logdir, f'epoch{epoch}_step{step}_Q_to_Q2opt.pt'))

            if ('last_path_prefix' in locals()) or ('last_path_prefix' in globals()):
                os.system('rm '+last_path_prefix)

            last_path_prefix = os.path.join(logdir, 'epoch'+str(epoch)+'_step'+str(step)+'*')
