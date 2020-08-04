###
#
# Embedding space analysis
#
###

import time
import numpy as np
import torch

import csv
import re

# from sklearn.decomposition import TruncatedSVD
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

# from utils.funcs import npy_to_tsv
from utils.model_paths import get_model_path, get_vocab_for_name, get_vocab_num, get_correl_dict, get_base_embeds, get_word_pair_datasets, get_syn_dists_basedir, get_base_embeds#, which_files_exist
# from utils.spatial import pairwise_dists, explained_variance, get_embedding_norms, calculate_singular_vals
from utils.plot import plot_exp_variance, plot_dists_dict_histogram, plot_histogram, plot_exp_variances, plot_dists_dict_histo_subplot, plot_dists_histogram_csv, word_pair_dists_mean_var, plot_mean_vars
from utils.data_processing import calculate_histograms, calculate_missing_values
from utils.validation import word_pair_distances, word_pair_dists_dict, correl_coefficients_for_vocab



if __name__ == '__main__':
    
    start_time = time.time()
    
    ratio_models = False
    
    # Calculate pairwise distance histograms
    # calculate_missing_values('dist_hists', ratio_models=ratio_models)
    '''
    dist_hist_dict = get_model_path('dist_hists', ratio_models=ratio_models)
    save_file = 'plots/dist_hists/w2vi_s_v7_ratios_norm_dists_hist_subp.png'
    # plot_dists_dict_histogram(dist_hist_dict, save_file=save_file, title='W2V init. syns ratios voc-7 distances histogram', model_filter='', norm=True)
    plot_dists_dict_histo_subplot(dist_hist_dict, save_file=save_file, title='W2V init. voc-7 syn. ratios pairwise distances', model_filter='', norm=True)
    '''
    # norms_dict = get_model_path('norms')
    # save_file = 'plots/norm_hists/w2v_init_ns_l2_hist.png'
    # plot_histogram(norms_dict, num_bins=100, save_file=save_file, title='Word2Vec in. no-syns L2 norms hists', model_filter='w2vi_ns', norm=False, x_label='Norms (bins)')
    
    # S_dict = get_model_path('sing_vals')
    # save_file = 'plots/explained_var/w2v_init_s_exp_vals.png'
    # plot_exp_variances(S_dict, save_file=save_file, model_filter='w2vi_s', title='Word2Vec init syns explained variances', log_scale=False, plot=False)
    
    '''
    # CALCULATE CORRELATION SCORES
    ratio_models = False
    vocab_num = 20
    
    # correlation_data2(correl_dataset_dict, embs_dict, incl_header=True, score_type='similarity', data_has_header=False, score_index=2)
    # correlation_data2(correl_dataset_dict, save_file, vocab_path, embs_dict, incl_header=True, score_type='similarity', data_has_header=False, score_index=2)
    # calculate_missing_values('correl_ws_rel')
    correl_coeffs = correl_coefficients_for_vocab(vocab_num, correl_metric='spearmanr', ratio_models=ratio_models)
    print('Correl coeffs:', correl_coeffs)
    '''
    
    '''
    Calculate word pair distances
    
    include_base = True
    word_pair_file_dict = get_word_pair_datasets()
    embs_dict = get_model_path('embeds', ratio_models=ratio_models, include_base=include_base)
    vocab_file = get_vocab_num(7)
    save_file = get_syn_dists_basedir() + 'ratios_syn_dists_base.csv'
    
    print('word_pair_dict', word_pair_file_dict)
    print('embds', embs_dict)
    
    # word_pair_dists_dict(word_pair_file_dict['synonyms'], vocab_file, embs_dict, save_file, focus_col='focus_word', target_col='synonym')
    
    # PLOT WORD PAIR DISTANCES
    dist_file_dict = get_syn_dists_basedir() + 'all_syn_dists.csv',
    # save_fig_file = 'plots/word_pair_dists/w2vi_syn_dists_hist.png'
    # plot_dists_histogram_csv(dist_file, save_file=save_fig_file, plot=False, title='W2V init - Synonym pairs dists. histogram', model_filter='w2vi', dist_type='_cos')
    
    vocab_num = 7
    # correl_coeffs = correl_coefficients_for_vocab(vocab_num, correl_metric='spearmanr', ratio_models=ratio_models)
    # print('correl_coeffs', correl_coeffs)
    '''
    
    
    '''
    PLOT DISTANCE MEAN AND STD
    models = 'all' # 'all'
    dist_file_dict = {  'syn'  : get_syn_dists_basedir() + models + '_syn_dists.csv',
                        'nat'  : get_syn_dists_basedir() + models + '_nat_dists.csv',
                        'rand' : get_syn_dists_basedir() + models + '_rand_dists.csv'}
    model_filter = '_euc'
    mean_var_dict = word_pair_dists_mean_var(dist_file_dict, model_filter=model_filter)
    save_fig_file = 'plots/word_pair_dists/mean_std_' + models + '_dists_euc.png'
    
    print('mean_var_dict', mean_var_dict)
    
    plot_mean_vars(mean_var_dict, save_fig_file, model_filter=model_filter, plot=False)
    '''
    
    '''
    # MATRIX NORMS
    ratio_models = False
    sing_vals_dict = get_model_path('sing_vals', ratio_models=ratio_models)
    print('sing_vals_dict', sing_vals_dict)
    
    for name, sing_val_file in sing_vals_dict.items():
        sing_vals = np.load(sing_val_file)
        print(name, sing_vals[0])
    '''
    
    elapsed_time = time.time() - start_time
    print('\nTotal elapsed time: ', elapsed_time)
    
    # plt.show()