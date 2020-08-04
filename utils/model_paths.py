###
#
# Model paths
#
###

import os.path
import re

def get_model_root_names(ratio_models=False):
    """
    Returns a dictionary of all available
    trained models
    
    TODO: Convert into config file
    
    Parameters
    ----------
    ratio_models : bool, optional
        whether to return the names of the ratio
        variation models (defualt=False)
    
    Returns
    -------
    embs_dict : {str : str}
        Dictionary of trained models, where the
        keys are shorthand names and the values
        are long names corresponding to the
        model locations. E.g.:
            model/rand_init-10e-voc3-emb300/
                rand_init-10e-voc3-emb300.npy
    """
    
    if not ratio_models:
        models_dict = {
                    'rand_s_v3'    :   'rand_init-10e-voc3-emb300',
                    'rand_s_v7'     :   'rand_init-syns-10e-voc7-emb300',
                    'rand_s_v20'    :   'rand_init-syns-10e-voc20-emb300',
                    
                    'rand_ns_v3'    :   'rand_init-no_syns-10e-voc3-emb300',
                    'rand_ns_v7'    :   'rand_init-no_syns-10e-voc7-emb300',
                    'rand_ns_v20'   :   'rand_init-no_syns-10e-voc20-emb300',
                    
                    'w2vi_s_v3'     :   'w2v_init-10e-voc3-emb300',
                    'w2vi_s_v7'     :   'w2v_init-syns-10e-voc7-emb300',
                    'w2vi_s_v20'    :   'w2v_init-syns-10e-voc20-emb300',
                    
                    'w2vi_ns_v3'    :   'w2v_init-nosyns-10e-voc3-emb300',
                    'w2vi_ns_v7'    :   'w2v_init-no_syns-10e-voc7-emb300',
                    'w2vi_ns_v20'   :   'w2v_init-no_syns-10e-voc20-emb300',
                    }
    else:
        models_dict = {
                'w2vi_r64_v7'     :   'w2v_init-syns-64r-10e-voc7-emb300',
                'w2vi_r50_v7'     :   'w2v_init-syns-50r-10e-voc7-emb300',
                'w2vi_r37_v7'     :   'w2v_init-syns-37r-10e-voc7-emb300',
                'w2vi_r25_v7'     :   'w2v_init-syns-10e-voc7-emb300',
                'w2vi_r16_v7'     :   'w2v_init-syns-16r-10e-voc7-emb300',
                'w2vi_r10_v7'     :   'w2v_init-syns-10r-10e-voc7-emb300',
                'w2vi_r06_v7'     :   'w2v_init-syns-06r-10e-voc7-emb300',
                'w2vi_r035_v7'    :   'w2v_init-syns-035r-10e-voc7-emb300',
                'w2vi_r02_v7'     :   'w2v_init-syns-02r-10e-voc7-emb300',
                'w2vi_r00_v7'     :   'w2v_init-no_syns-10e-voc7-emb300',
                }
    
    return models_dict


def get_models_basedir():
    return 'model/'
    

def get_vocab_basedir():
    return 'data/vocabulary/'


def get_model_path(mode='embeds', ratio_models=False, include_base=False):
    """
    Returns a dictionary of paths to model
    embeddings files
    
    Parameters
    ----------
    ratio_models : bool, optional
        whether to get the model paths for the
        ratio variation models (defualt=False)
    """
    basedir = get_models_basedir()
    
    if mode == 'embeds': file_extension = '.npy'
    elif mode == 'norms': file_extension = '-norms.npy'
    elif mode == 'sing_vals': file_extension = '_SVD_S.npy'
    elif mode == 'cos_dist': file_extension = '_cos_dists.npy'
    elif mode == 'embeds_tsv': file_extension = '.tsv'
    elif mode == 'dist_hists': file_extension = '_cos_dists_hist.npy'
    elif mode == 'correl_ws_sim': file_extension = '_correl_ws_sim.npy'
    elif mode == 'correl_ws_rel': file_extension = '_correl_ws_rel.npy'
    elif mode == 'correl_simlex': file_extension = '_correl_simlex.npy'
    
    else: raise ValueError('The option mode=%r is not defined' % (mode))
    
    embeds_dict = {k: (basedir + v + '/' + v + file_extension) for k, v in get_model_root_names(ratio_models).items()}
    
    if mode == 'embeds' and include_base:
        for k,v in get_base_embeds().items():
            embeds_dict[k] = v
    
    return embeds_dict


def get_vocab_for_name(name):
    if re.search("_v3", name):
        vocab_file = 'vocabulary.csv'
    elif re.search("_v7", name):
        vocab_file = 'vocabulary-7.csv'
    elif re.search("_v20", name):
        vocab_file = 'vocabulary-20.csv'
    else:
        raise ValueError('No vocabulary defined for name=%r' % (name))
    
    return get_vocab_basedir() + vocab_file


def get_vocab_num(num):
    return get_vocab_for_name('_v' + str(num))


def get_base_embeds_basedir():
    return 'data/word_embeddings/'


def get_base_embeds():
    base_embeds_name = 'word2vec-google-news-300'
    
    base_embeds_dict = {
            'w2v_v3': get_base_embeds_basedir() + base_embeds_name + '_voc3.npy',
            'w2v_v7': get_base_embeds_basedir() + base_embeds_name + '_voc7.npy',
            'w2v_v20': get_base_embeds_basedir() + base_embeds_name + '_voc20.npy',
    }
    return base_embeds_dict


def get_embeds_for_vocab(vocab_num, ratio_models=False):
    base_embeds = get_base_embeds()
    embeds_dict = get_model_path('embeds', ratio_models)
    
    embeds = {}
    vocab_re = '_v' + str(vocab_num)
    
    for emb_name, emb_path in base_embeds.items():
        if re.search(vocab_re, emb_name):
            embeds[emb_name] = emb_path
    
    for emb_name, emb_path in embeds_dict.items():
        if re.search(vocab_re, emb_name):
            embeds[emb_name] = emb_path
    
    return embeds
 

def which_files_exist(path_dict):
    file_exists_dict = {k: os.path.exists(v) for k, v in path_dict.items()}
    
    return file_exists_dict


def get_correl_basedir():
    return 'data/validate/'


def get_correl_datasets():
    correl_data_dict = {
        'SimLex' : get_correl_basedir() + 'SimLex-999.txt',
        'WordSim-sim' : get_correl_basedir() + 'wordsim_similarity_goldstandard.txt',
        'WordSim-rel' : get_correl_basedir() + 'wordsim_relatedness_goldstandard.txt'
    }
    return correl_data_dict


def get_correl_dict(vocab_num):
    vocab_num = str(vocab_num)
    correl_dict = {
        'correl_ws_rel': get_correl_basedir() + 'wordsim_relatedness_vocab' + vocab_num + '.csv',
        'correl_ws_sim': get_correl_basedir() + 'wordsim_similarity_vocab' + vocab_num + '.csv',
        'correl_simlex': get_correl_basedir() + 'SimLex-999_vocab' + vocab_num + '.csv'
        }
    return correl_dict


def correl_paths_for_vocab(vocab_num, ratio_models=False):
    correl_results_dir = get_correl_basedir() + 'correlation/'
    
    vocab_re = '_v' + str(vocab_num)
    ratios = '' if not ratio_models else '_ratios'
    result_files = {
        'SimLex' : correl_results_dir + 'SimLex' + ratios + vocab_re + '.csv',
        'WordSim-sim' : correl_results_dir + 'WordSim-sim' + ratios + vocab_re + '.csv',
        'WordSim-rel' : correl_results_dir + 'WordSim-rel' + ratios + vocab_re + '.csv'
    }
    return result_files


def get_knn_results(model_name, knn=11):
    results_dir = 'data/wmd/results/'
    file_name = model_name + '_wmd-knn' + str(knn) + '-results.csv'
    
    return results_dir + file_name


def get_syn_dists_basedir():
    return 'data/syn_dists/'


def get_word_pair_datasets():
    word_pair_data_dict = {
        'synonyms' : get_syn_dists_basedir() + 'synonym_counts_vocab3_sw.csv',
        'random' : get_syn_dists_basedir() + 'rand_w_pairs.csv',
        'sampled' : get_syn_dists_basedir() + 'sampled_w_pairs.csv'
    }
    return word_pair_data_dict