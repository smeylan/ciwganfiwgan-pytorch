# -*- coding: utf-8 -*-
"""
Author: Andrej Leban
Created on Sun May 29 13:05:27 2022
"""

import argparse
import os
import time
import numpy as np

from scipy.io.wavfile import read, write
# import soundfile as sf
import torch

from infowavegan import WaveGANGenerator
from utils import get_continuation_fname

# cf: https://github.com/pytorch/pytorch/issues/16797
# class CPU_Unpickler(pk.Unpickler):
#     def find_class(self, module, name):
#         if module == 'torch.storage' and name == '_load_from_bytes':
#             return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
#         else:
#             return super().find_class(module, name)

if __name__ == "__main__":
    # generator = CPU_Unpickler(open("generator.pkl", 'rb')).load()
    # discriminator = CPU_Unpickler(open("discriminator.pkl", 'rb')).load()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dir',
        type=str,
        required=True,
        help='Directory where checkpoints are saved'
    )
    parser.add_argument(
        '--epoch',
        type=str,
        required=True,
        help='Training Directory'
    )
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=16000,
        help='Q-net categories'
    )
    parser.add_argument(
        '--slice_len',
        type=int,
        default=16384,
    )
    parser.add_argument(
        '--num_categ',
        type=int,
        default=0,
        help='Q-net categories'
    )

    parser.add_argument(
        '--num_sentences',
        type=int,
        default=10,
        help='Number of sentences to sample'
    )

    parser.add_argument(
        '--max_batch_size',
        type=int,
        default=64,
        help='Max batch size. If num_sentences * num_categories exceeds max_batch_size, bail!'
    )

    args = parser.parse_args()
    if (args.num_sentences * args.num_categ) > args.max_batch_size:
        raise ValueError('num_sentences * num_categories exceeds max_batch_size')
    
    epoch = args.epoch
    dir = args.dir
    sample_rate = args.sample_rate
    slice_len = args.slice_len

    # Load generator from checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fname, _ = get_continuation_fname(epoch, dir)
    G = WaveGANGenerator(slice_len=slice_len)
    G.load_state_dict(torch.load(os.path.join(dir, fname + "_G.pt"),
                                 map_location = device))
    G.to(device)
    G.eval()

    # Generate 10 sentences from one hots
    num_words = args.num_sentences * args.num_categ
    
    _z = torch.FloatTensor(num_words, 100 - (args.num_categ + 1)).uniform_(-1., 1.).to(device)
    
    sent = np.zeros([args.num_categ, args.num_categ], dtype=np.float32)
    np.fill_diagonal(sent, 1., wrap=False)    
    
    zeros = torch.from_numpy(np.zeros([num_words,1], dtype=np.float32)).to(device)    
    c = torch.from_numpy(np.vstack([sent for x in range(args.num_sentences)])).to(device)
        
    z = torch.cat((c, zeros ,_z), dim=1)

    print('Generating...')
    genData = G(z).detach().cpu().numpy()
    # take num_sentences cuts that are each num_categ long
    
    print('Slicing...')
    sentences =  [np.hstack(genData[x:x+args.num_categ]) for x in  [10*x for x in range(args.num_sentences+1)]]
    
    print('Adding silences...')
    silence = np.zeros([1,4000], dtype=np.float32)
    output_sentences = []
    for i in range(len(sentences)):
        output_sentences.append(sentences[i])
        output_sentences.append(silence)
    final = np.hstack(output_sentences)[0]
    
    test_sentence_dir  = 'test_sentences'
    if not os.path.exists(test_sentence_dir):
        os.makedirs(test_sentence_dir)
    
    print('Writing out the wav...')
    output_path = os.path.join(test_sentence_dir, fname+".wav")    
    write(output_path, 16000, final)
    
    # slice them up, concatenate, add some space in between., write them out
    # write(f'out.wav', sample_rate, (genData * 32767).astype(np.int16))
    #sd.play(genData, sample_rate)
        
