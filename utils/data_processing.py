###
#
# Data processing
# Support functions
#
###

import numpy as np
import time
import re

import torch

from .funcs import npy_to_tsv
from .model_paths import get_model_path, which_files_exist, get_vocab_for_name, get_correl_dict, get_base_embeds, get_correl_dict
from .spatial import pairwise_dists, explained_variance, get_embedding_norms, calculate_singular_vals

from .validation import calc_correl_data


def calculate_histograms(npy_file, save_file, num_bins=100, min_val=None, max_val=None):
    t = time.time()
    
    val_array = np.load(npy_file, mmap_mode='r')
    
    if min_val == None: min_val = val_array.min()
    if max_val == None: max_val = val_array.max()
    
    print('Processing histogram from ', npy_file)
    hist, bins = np.histogram(val_array, num_bins, (min_val, max_val))
    
    print('Saving histogram to ', save_file)
    np.save(save_file, np.dstack((bins[:-1], hist)))
    
    elapsed_time = time.time() - t
    print('\n\tElapsed time: \t\t %f' % (elapsed_time))


def calculate_missing_values(val_name, ratio_models=False):
    embs_dict = get_model_path('embeds', ratio_models=ratio_models)
    vals_dict = get_model_path(val_name, ratio_models=ratio_models)
    embs_exist = which_files_exist(embs_dict)
    vals_exist = which_files_exist(vals_dict)
    
    for name, file in vals_exist.items():
        print(name, '\t', file)
    
    for name, file in embs_dict.items():
        if embs_exist[name]:
            if not vals_exist[name]:
                print('Calculating ', val_name, ' for ', name)
                if val_name == 'norms':
                    get_embedding_norms(embs_dict[name], vals_dict[name])
                
                elif val_name == 'cos_dist':
                    pairwise_dists(embs_dict[name], vals_dict[name])
                
                elif val_name == 'sing_vals':
                    calculate_singular_vals(embs_dict[name], vals_dict[name])
                
                elif val_name == 'embeds_tsv':
                    vocab_file = get_vocab_for_name(name)
                    print('Voc file for %r: \t %r' % (name, vocab_file))
                    npy_to_tsv(embs_dict[name], vocab_file, vals_dict[name])
                
                elif val_name == 'dist_hists':
                    dists = get_model_path('cos_dist', ratio_models=ratio_models)
                    num_bins = 100
                    min_val = 0.
                    max_val = 2.
                    
                    calculate_histograms(dists[name], vals_dict[name], num_bins=num_bins, min_val=min_val, max_val=max_val)
                elif re.search('correl', val_name):#val_name == 'correl':
                    correl_dict = get_correl_dict()
                    
                    calc_correl_data(embs_dict[name], correl_dict[val_name], vals_dict[name], incl_header=True, data_has_header=False, score_index=2)
                else:
                    raise ValueError('val_name=%r is not a valid option' % (val_name))
            else:
                print(val_name, ' file exists for ', name)
        else:
            print('%r embeddings file does not exist in %r' % (name, embs_dict[name]))