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
parameters['word_embeddings_dir'] = parameters['data_dir'] + 'word_embeddings/'

parameters['vocab_cutoff'] = 5

parameters['vocab_file'] = os.path.abspath(parameters['vocabulary_dir'] + 'vocabulary-' + str(parameters['vocab_cutoff']) + '.csv')
parameters['dataset_file'] = os.path.abspath(parameters['train_dataset_dir'] + 'dataset_vocab' + str(parameters['vocab_cutoff']) + '.csv')
parameters['syn_file'] = os.path.abspath(parameters['train_dataset_dir'] + 'syn_dataset_vocab' + str(parameters['vocab_cutoff']) + '.csv')
parameters['validation_file'] = os.path.abspath(parameters['val_dataset_dir'] + 'dataset_vocab' + str(parameters['vocab_cutoff']) + '.csv')
parameters['w2v_path'] = os.path.abspath(parameters['word_embeddings_dir'] + 'word2vec-google-news-300_voc' + str(parameters['vocab_cutoff']) + '.csv')

parameters['data_augmentation_ratio'] = .02
parameters['w2v_init'] = True
parameters['syn_augm'] = True

parameters['embedding_size'] = 300
parameters['epochs'] = 10
parameters['batch_size'] = 10
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
parameters['bnc_data'] = parameters['general_data_dir'] + 'British_National_Corpus/bnc_full_processed_data/' + parameters['bnc_data_name'] + '.txt'
parameters['bnc_data_tags'] = parameters['general_data_dir'] + 'British_National_Corpus/bnc_full_processed_data/' + parameters['bnc_data_name'] + '_tags.txt'
# parameters['bnc_data'] = parameters['data_dir'] + parameters['bnc_data_name'] + '_1.txt'
# parameters['bnc_data_tags'] = parameters['data_dir'] + parameters['bnc_data_name'] + '_tags_1.txt'
parameters['bnc_skipgram_data'] = parameters['data_dir'] + 'skipgram_' + parameters['bnc_data_name'] + '.txt'
parameters['bnc_skipgram_augm_data'] = parameters['data_dir'] + 'skipgram_augm_' + parameters['bnc_data_name'] + '.txt'

def print_parameters():
    global parameters
    
    # PRINT PARAMETERS
    print('\n=================== MODEL PARAMETERS: =================== \n')
    for name, value in parameters.items():
        # num_tabs = int((32 - len(name))/8) + 1
        # tabs = '\t' * num_tabs
        num_spaces = 30 - len(name)
        spaces = ' ' * num_spaces
        print(f'{name}: {spaces} {value}')
    print('\n=================== / MODEL PARAMETERS: =================== \n')