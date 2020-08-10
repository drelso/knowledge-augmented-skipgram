###
#
# Construct SkipGram dataset
#
###v

import os
from config import parameters
from utils.dataset_utils import shuffle_and_subset_dataset, process_bnc_data, numericalise_dataset, train_validate_split
from utils.funcs import dir_validation, dir_validation
import time
import datetime




if __name__ == '__main__':
    # GET THE RIGHT DATASET SIZE
    DATA_PATH = parameters['bnc_subset_data'] if parameters['use_data_subset'] else parameters['bnc_data']
    TAGS_PATH = parameters['bnc_subset_tags'] if parameters['use_data_subset'] else parameters['bnc_tags']

    ## SHUFFLE AND SUBSET DATASET
    if parameters['use_data_subset']:
        print(f'Using data subset: {parameters["data_subset_size"] * 100}% of full dataset')
        if not os.path.exists(parameters['bnc_subset_data']) or not os.path.exists(parameters['bnc_subset_tags']):
            shuffle_and_subset_dataset(
                parameters['bnc_data'],
                parameters['bnc_tags'],
                parameters['bnc_subset_data'],
                parameters['bnc_subset_tags'],
                data_size=parameters['data_subset_size'])
        else:
            print(f'Found existing dataset subset at {parameters["bnc_subset_data"]} and {parameters["bnc_subset_tags"]} \n')

    # SPLIT DATASET
    if not os.path.exists(parameters['train_data']) or not os.path.exists(parameters['train_tags']) \
        or not os.path.exists(parameters['val_data']) or not os.path.exists(parameters['val_tags']):
        
        train_validate_split(
            DATA_PATH,
            parameters['train_data'],
            parameters['val_data'],
            tags_data_file=TAGS_PATH,
            train_tags_savefile=parameters['train_tags'],
            val_tags_savefile=parameters['val_tags'],
            proportion=parameters['split_ratio'])
    else:
        print(f'Found existing train/validation datasets at: \n - {parameters["train_data"]} \n - {parameters["train_tags"]} \n - {parameters["val_data"]} \n - {parameters["val_tags"]}')
    
    # CONSTRUCT AUGMENTED DATASET
    if parameters['syn_augm']:
        print('Processing with synonym augmented data')

    if not os.path.exists(parameters['train_skipgram_data']):
        TRAIN_AUGM_DATA = parameters['train_skipgram_augm_data'] if parameters['syn_augm'] else None
        print(f'Constructing Skip gram training dataset at {parameters["train_skipgram_data"]}')
        process_bnc_data(parameters['train_data'], parameters['train_skipgram_data'], tags_file=parameters['train_tags'], augm_dataset_file=TRAIN_AUGM_DATA, ctx_size=parameters['ctx_size'])
    else:
        print(f'Skip gram training data file found at {parameters["train_skipgram_data"]}')
        if parameters['syn_augm'] and not os.path.exists(parameters['train_skipgram_augm_data']):
            raise Exception(f'Skip gram file exists, but no augmented training data file found at {parameters["train_skipgram_augm_data"]}')
    
    if not os.path.exists(parameters['val_skipgram_data']):
        VAL_AUGM_DATA = parameters['val_skipgram_augm_data'] if parameters['syn_augm'] else None
        print(f'Constructing augmented validation dataset at {parameters["val_skipgram_augm_data"]}')
        process_bnc_data(parameters['val_data'], parameters['val_skipgram_data'], tags_file=parameters['val_tags'], augm_dataset_file=VAL_AUGM_DATA, ctx_size=parameters['ctx_size'])
    else:
        print(f'Skipgram validation data file found at {parameters["val_skipgram_data"]}')
        if parameters['syn_augm'] and not os.path.exists(parameters['val_skipgram_augm_data']):
            raise Exception(f'Skip gram file exists, but no augmented validation data file found at {parameters["val_skipgram_augm_data"]}')
    
    '''
    TODO: CONSTRUCT NEW COUNTS FILE FOR SUBSET OF DATASET ***BEFORE*** NUMERICALISING
    ## ONLY NUMERICALISE THE DATA RIGHT IF NO EXISTING FILE IS FOUND
    if not os.path.exists(parameters['num_train_skipgram_data']):
        print(f'No numericalised file found at {parameters["num_train_skipgram_data"]}, creating numericalised file from dataset at {parameters["train_skipgram_data"]}')
        numericalise_dataset(parameters['train_skipgram_data'], parameters['num_train_skipgram_data'], vocabulary)
    else:
        print(f'Numericalised Skip gram training data file found at {parameters["num_train_skipgram_data"]}')
    
    if not os.path.exists(parameters['num_val_skipgram_augm_data']):
        print(f'No numericalised file found at {parameters["num_train_skipgram_augm_data"]}, creating numericalised file from dataset at {parameters["train_skipgram_augm_data"]}')
        numericalise_dataset(parameters['train_skipgram_augm_data'], parameters['num_train_skipgram_augm_data'], vocabulary)
    else:
        print(f'Numericalised Skip gram training data file found at {parameters["num_train_skipgram_augm_data"]}')
    '''
    
    '''
    dataset_sampling(data_file, augm_data_file, dataset_file, augm_dataset_file, max_context=5)

    # skipgram_data_from_gutenberg(parameters['gutenberg_dir'], processed_books_path, vocab_file, syn_sel_file, num_books=10)
    '''