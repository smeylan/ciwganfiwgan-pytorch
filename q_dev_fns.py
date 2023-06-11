import numpy as np
import torch
import pandas as pd
import librosa
from IPython.display import Audio, display
import glob
import os
from tqdm import tqdm
from scipy.io.wavfile import read

def load_wavs_and_labels(datadir, slice_len, NUM_CATEG, device, timit_words):
    dir = glob.glob(os.path.join(datadir,'*.wav'))
    x = np.zeros((len(dir), 1, slice_len))
    y = np.zeros((len(dir), NUM_CATEG+1)) # +1 for UNK
    
    i = 0
    filenames = []
    for file in tqdm(dir):
        filenames.append(file)
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
        word = os.path.basename(file).split('_')[0]
        j = timit_words.index(word)
        y[i, j] = 1            
        i += 1
        
    audio = torch.from_numpy(np.array(x, dtype=np.float32)).to(device)
    labels = torch.from_numpy(np.array(y, dtype=np.float32)).to(device)
    
    return(audio, labels, filenames)
        
def evaluate_asr_system(ASR_probs, Y, device, timit_words, filenames):
    
    timit_words = np.array(timit_words)
    highest_prob_from_asr = torch.argmax(ASR_probs, dim=1)    
    matches = torch.eq(highest_prob_from_asr.to(device), torch.argmax(Y, dim=1))
    failures = torch.argwhere(~matches)

    # compute entropy
    log_probs = torch.log(ASR_probs)
    prod = ASR_probs * log_probs
    entropy = -torch.sum(prod, dim =1)    

    # entropy classifier goes here

    rdf = pd.DataFrame({
        "human label":[timit_words[x] for x in torch.argmax(Y, dim=1).detach().cpu().numpy()],
        "asr system label":[timit_words[x] for x in highest_prob_from_asr.detach().cpu().numpy()],
        'matches': matches.detach().cpu().numpy(),
        "filenames" : np.array(filenames),
        'entropy': entropy.detach().cpu().numpy(),
        'recognized': matches.detach().cpu().numpy()
    })
    
    full_performance = np.mean(matches.detach().cpu().numpy())
    
    rdict = {}
    rdict['df'] = rdf
    rdict['incorrect'] = rdf.loc[~rdf.matches]
    
    # full_performance: how good is the performance, treating UNK as a word
    rdict['full_performance'] = full_performance 
    
    if rdf.loc[rdf['asr system label'] == 'UNK'].shape[0] > 0:
        # how good is the system at recognizing UNKs
        rdict['unk_precision'] =  rdf.loc[(rdf['asr system label'] == 'UNK') & (rdf['human label'] == 'UNK')].shape[0] / rdf.loc[rdf['asr system label'] == 'UNK'].shape[0]
        # did we get all of the human-labeled unks
        rdict['unk_recall'] = rdf.loc[(rdf['human label'] == 'UNK') & (rdf['asr system label'] == 'UNK')].shape[0] / rdf.loc[rdf['human label'] == 'UNK'].shape[0]
    else: 
        rdict['unk_precision'] = np.nan
        rdict['unk_recall'] = np.nan
    

    # among items the model does not think are UNK, how many matches?
    rdict['word_rec_performance'] = np.mean(rdf.loc[rdf['asr system label'] != 'UNK'].matches)
    


    rdict['failures'] = failures
    
    return(rdict)            

def inpsect_failure(network_results, index):
    print('Mistmatch in: '+network_results['incorrect'].iloc[index]['filenames'])
    print('Q index expects: '+network_results['incorrect'].iloc[index]['asr system label'])
    print('Human labeled: '+network_results['incorrect'].iloc[index]['human label'])
    signal, sample_rate = librosa.load(network_results['incorrect']['filenames'][index], sr=None)
    return(display(Audio(data=signal, rate=sample_rate)))   

def mark_unks_in_Q(Q_network_probs, threshold, device):
    # need a little prob mass on the UNKs to avoid numerical errors if this is the Q network 
    zeros = torch.zeros([Q_network_probs.shape[0],1], device = device)
    zeros += .0001
    Q_network_probs_with_unk = torch.hstack((Q_network_probs, zeros))
    unk_tensor = torch.zeros([1, Q_network_probs_with_unk.shape[1]], device = device)
    unk_tensor += .0001
    unk_tensor[0,-1] = .999999

    # compute entropies
    log_probs = torch.log(Q_network_probs_with_unk)
    prod = Q_network_probs_with_unk * log_probs
    entropy = -torch.sum(prod, dim =1)        

    unks_to_mark = torch.argwhere( entropy > torch.Tensor([threshold]).to(device))
    Q_network_probs_with_unk[unks_to_mark,] = unk_tensor
    return(Q_network_probs_with_unk)
