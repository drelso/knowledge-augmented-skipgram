###
#
# Knowledge-Augmented Skip gram configuration file
#
###
import os
from pathlib import Path

home = str(Path.home())

parameters = {}

parameters['general_data_dir'] = home + '/data/'
parameters['data_dir'] = 'data/'
parameters['train_dataset_dir'] = parameters['data_dir'] + 'train/'
parameters['val_dataset_dir'] = parameters['data_dir'] + 'validate/'
parameters['vocabulary_dir'] = parameters['data_dir'] + 'vocabulary/'
parameters['word_embeddings_dir'] = parameters['general_data_dir'] + 'word_embeddings/'

parameters['vocab_cutoff'] = 5

parameters['vocab_file'] = os.path.abspath(parameters['vocabulary_dir'] + 'vocabulary-' + str(parameters['vocab_cutoff']) + '.csv')
parameters['dataset_file'] = os.path.abspath(parameters['train_dataset_dir'] + 'dataset_vocab' + str(parameters['vocab_cutoff']) + '.csv')
parameters['syn_file'] = os.path.abspath(parameters['train_dataset_dir'] + 'syn_dataset_vocab' + str(parameters['vocab_cutoff']) + '.csv')
parameters['validation_file'] = os.path.abspath(parameters['val_dataset_dir'] + 'dataset_vocab' + str(parameters['vocab_cutoff']) + '.csv')
# parameters['w2v_path'] = os.path.abspath(parameters['word_embeddings_dir'] + 'word2vec-google-news-300_voc' + str(parameters['vocab_cutoff']) + '.csv')
parameters['w2v_path'] = None

parameters['data_augmentation_ratio'] = .25
parameters['w2v_init'] = False
parameters['syn_augm'] = True

parameters['split_ratio'] = .9

# WORD2VEC VARIABLES
parameters['embedding_size'] = 300
parameters['epochs'] = 1 #10
parameters['batch_size'] = 10
parameters['ctx_size'] = 5
parameters['num_neg_samples'] = 5
parameters['learning_rate'] = 0.01

parameters['model_name'] =  ('w2v_init' if parameters['w2v_init'] else 'rand_init') + '-' + \
                            ('syns' if parameters['syn_augm'] else 'no_syns') + '-' + \
                            str(parameters['data_augmentation_ratio']).strip("0").strip(".") + 'r-' + \
                            str(parameters['epochs']) + 'e-' + \
                            'voc' + str(parameters['vocab_cutoff']) + \
                            '-emb' + str(parameters['embedding_size'])

parameters['all_models_dir'] = 'model/'
parameters['model_dir'] = parameters['all_models_dir'] + parameters['model_name'] + '/'
parameters['model_file'] =  os.path.abspath(parameters['model_dir'] + parameters['model_name'] + '.pth')
parameters['checkpoints_dir'] =  os.path.abspath(parameters['model_dir'] + 'checkpoints/')
parameters['input_emb_file'] =  os.path.abspath(parameters['model_dir'] + parameters['model_name'])

# GUTENBERG PARAMETERS
parameters['gutenberg_dir'] = parameters['data_dir'] + 'Gutenberg/'
parameters['processed_books_path'] = parameters['data_dir'] + 'list_processed_books.txt'

parameters['vocab_file'] = parameters['vocabulary_dir'] + 'vocabulary.csv'
parameters['syn_sel_file'] = parameters['train_dataset_dir'] + 'synonyms_vocab' + str(parameters['vocab_cutoff']) + '.csv'


# BNC DATA
parameters['bnc_data_name'] = 'bnc_full_proc_data'
# parameters['bnc_data_name'] = 'bnc_baby_proc_data'
parameters['bnc_data_dir'] = parameters['general_data_dir'] + 'British_National_Corpus/bnc_full_processed_data/'
parameters['bnc_data'] = parameters['bnc_data_dir'] + parameters['bnc_data_name'] + '.txt'
parameters['bnc_tags'] = parameters['bnc_data_dir'] + parameters['bnc_data_name'] + '_tags.txt'

parameters['use_data_subset'] = True
parameters['data_subset_size'] = 0.5
parameters['bnc_subset_data_name'] = parameters['bnc_data_name'] + '_shffl_sub-' + str(parameters['data_subset_size']).strip("0").strip(".")
parameters['bnc_subset_data'] = parameters['bnc_data_dir'] + parameters['bnc_subset_data_name'] + '.txt'
parameters['bnc_subset_tags'] = parameters['bnc_data_dir'] + parameters['bnc_subset_data_name'] + '_tags.txt'

parameters['train_data'] = parameters['data_dir'] + parameters['bnc_data_name'] + '_train.txt'
parameters['train_tags'] = parameters['data_dir'] + parameters['bnc_data_name'] + '_train_tags.txt'
parameters['val_data'] = parameters['data_dir'] + parameters['bnc_data_name'] + '_val.txt'
parameters['val_tags'] = parameters['data_dir'] + parameters['bnc_data_name'] + '_val_tags.txt'

parameters['bnc_skipgram_data'] = parameters['bnc_data_dir'] + 'skipgram_' + parameters['bnc_data_name'] + '.csv'
parameters['bnc_skipgram_augm_data'] = parameters['bnc_data_dir'] + 'skipgram_augm_' + parameters['bnc_data_name'] + '.csv'

parameters['train_skipgram_data'] = parameters['data_dir'] + 'skipgram_' + parameters['bnc_data_name'] + '_train.csv'
parameters['train_skipgram_augm_data'] = parameters['data_dir'] + 'skipgram_augm_' + parameters['bnc_data_name'] + '_train.csv'
parameters['val_skipgram_data'] = parameters['data_dir'] + 'skipgram_' + parameters['bnc_data_name'] + '_val.csv'
parameters['val_skipgram_augm_data'] = parameters['data_dir'] + 'skipgram_augm_' + parameters['bnc_data_name'] + '_val.csv'

# parameters['bnc_skipgram_data'] = parameters['data_dir'] + 'skipgram_bnc_baby_proc_data_1.csv'
# parameters['bnc_skipgram_augm_data'] = parameters['data_dir'] + 'skipgram_augm_bnc_baby_proc_data_1.csv'

parameters['num_train_skipgram_data'] = parameters['data_dir'] + 'skipgram_' + parameters['bnc_data_name'] + '_voc-' + str(parameters['vocab_cutoff']) + '_train.csv'
parameters['num_train_skipgram_augm_data'] = parameters['data_dir'] + 'skipgram_augm_' + parameters['bnc_data_name'] + '_voc-' + str(parameters['vocab_cutoff']) + '_train.csv'
parameters['num_val_skipgram_data'] = parameters['data_dir'] + 'skipgram_' + parameters['bnc_data_name'] + '_voc-' + str(parameters['vocab_cutoff']) + '_val.csv'
parameters['num_val_skipgram_augm_data'] = parameters['data_dir'] + 'skipgram_augm_' + parameters['bnc_data_name'] + '_voc-' + str(parameters['vocab_cutoff']) + '_val.csv'

parameters['bnc_counts'] = parameters['bnc_data_dir'] + 'counts_bnc_full_seqlist_deptree.csv'