###
#
# Construct SkipGram dataset
#
###v

import os
from config import parameters
from utils.dataset_utils import shuffle_and_subset_dataset, process_bnc_data, numericalise_dataset, train_validate_split, basic_tokenise, word_counts, dataset_sampling, select_synonyms
from utils.funcs import dir_validation, print_parameters
from utils.training_utils import build_vocabulary
import time
import datetime




if __name__ == '__main__':
    
    # DATA_PATH = parameters['bnc_subset_data'] if parameters['use_data_subset'] else parameters['bnc_data']
    # TAGS_PATH = parameters['bnc_subset_tags'] if parameters['use_data_subset'] else parameters['bnc_tags']
    # COUNTS_FILE = parameters['bnc_subset_counts'] if parameters['use_data_subset'] else parameters['bnc_counts']
    # print_parameters(parameters)
    # exit()

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

    ## DATASET WORD COUNTS
    if not os.path.exists(parameters['counts_file']):
        print(f'Calculating word counts for dataset at {parameters["raw_data"]}')
        tokenised_data = basic_tokenise(parameters['raw_data'], preserve_sents=True)
        word_counts(tokenised_data, parameters['counts_file'])
    else:
        print(f'Found existing word counts file at {parameters["counts_file"]}\n')


    ## SPLIT DATASET
    if not os.path.exists(parameters['train_data']) or not os.path.exists(parameters['train_tags']) \
        or not os.path.exists(parameters['val_data']) or not os.path.exists(parameters['val_tags']):
        
        train_validate_split(
            parameters['raw_data'],
            parameters['train_data'],
            parameters['val_data'],
            tags_data_file=parameters['raw_tags'],
            train_tags_savefile=parameters['train_tags'],
            val_tags_savefile=parameters['val_tags'],
            proportion=parameters['split_ratio'])
    else:
        print(f'Found existing train/validation datasets at: \n - {parameters["train_data"]} \n - {parameters["train_tags"]} \n - {parameters["val_data"]} \n - {parameters["val_tags"]}')
    
    # CONSTRUCT AUGMENTED DATASETS
    # NOTE: augmented validation dataset might not be necessary
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
        print(f'Skip gram validation data file found at {parameters["val_skipgram_data"]}')
        if parameters['syn_augm'] and not os.path.exists(parameters['val_skipgram_augm_data']):
            raise Exception(f'Skip gram file exists, but no augmented validation data file found at {parameters["val_skipgram_augm_data"]}')
    
    ## SAMPLE CONTEXT WORDS AND SAVE TO FILE
    if not os.path.exists(parameters['train_skipgram_sampled_data']):
        print(f'Constructing context-sampled skip gram training dataset at {parameters["train_skipgram_data"]}')
        dataset_sampling(
            parameters['train_skipgram_data'],
            parameters['train_skipgram_augm_data'],
            parameters['train_skipgram_sampled_data'],
            parameters['train_skipgram_augm_sampled_data'],
            max_context=parameters['ctx_size'])
    else:
        print(f'Context-sampled skip gram training data file found at {parameters["train_skipgram_sampled_data"]}')
        if parameters['syn_augm'] and not os.path.exists(parameters['train_skipgram_augm_sampled_data']):
            raise Exception(f'Context-sampled skip gram file exists, but no corresponding augmented training data file found at {parameters["train_skipgram_augm_data"]}')
    
    if not os.path.exists(parameters['val_skipgram_sampled_data']):
        print(f'Constructing context-sampled skip gram validation dataset at {parameters["val_skipgram_sampled_data"]}')
        dataset_sampling(
            parameters['val_skipgram_data'],
            parameters['val_skipgram_augm_data'],
            parameters['val_skipgram_sampled_data'],
            parameters['val_skipgram_augm_sampled_data'],
            max_context=parameters['ctx_size'])
    else:
        print(f'Context-sampled skip gram validation data file found at {parameters["val_skipgram_sampled_data"]}')
        if parameters['syn_augm'] and not os.path.exists(parameters['val_skipgram_augm_sampled_data']):
            raise Exception(f'Context-sampled skip gram file exists, but no corresponding augmented validation data file found at {parameters["val_skipgram_augm_data"]}')
    
    ## CONSTRUCT VOCABULARY OBJECT FROM COUNTS FILE
    VOCABULARY = build_vocabulary(parameters['counts_file'], min_freq=parameters['vocab_cutoff'])

    ## SELECT A SINGLE SYNONYM PER FOCUS WORD AND SAVE TO FILE
    if not os.path.exists(parameters['train_skipgram_syns_data']):
        print(f'Constructing single synonym skip gram training dataset at {parameters["train_skipgram_syns_data"]}')
        select_synonyms(
            parameters['train_skipgram_augm_sampled_data'],
            parameters["train_skipgram_syns_data"],
            VOCABULARY,
            syn_selection=parameters['synonym_selection'])
    else:
        print(f'Single synonym skip gram training data file found at {parameters["train_skipgram_syns_data"]}')
    
    if not os.path.exists(parameters['val_skipgram_syns_data']):
        print(f'Constructing single synonym skip gram validation dataset at {parameters["val_skipgram_syns_data"]}')
        select_synonyms(
            parameters['val_skipgram_augm_sampled_data'],
            parameters["val_skipgram_syns_data"],
            VOCABULARY,
            syn_selection=parameters['synonym_selection'])
    else:
        print(f'Single synonym skip gram validation data file found at {parameters["val_skipgram_syns_data"]}')
        
    ## NUMERICALISE DATASETS
    # NUMERICALISING TRAINING SKIPGRAM DATA
    if not os.path.exists(parameters['num_train_skipgram_data']):
        print(f'No numericalised Skip gram training data file found at {parameters["num_train_skipgram_data"]}, creating numericalised file from dataset at {parameters["train_skipgram_data"]} and counts file at {parameters["counts_file"]}')
        numericalise_dataset(parameters['train_skipgram_data'], parameters['num_train_skipgram_data'], VOCABULARY)
    else:
        print(f'Numericalised Skip gram training data file found at {parameters["num_train_skipgram_data"]}')
    
    # NUMERICALISING AUGMENTED, CONTEXT-SAMPLED, SINGLE-SYNONYM TRAINING SKIPGRAM DATA
    if not os.path.exists(parameters['num_train_skipgram_syns_data']):
        print(f'No numericalised Skip gram training augmented data file found at {parameters["num_train_skipgram_syns_data"]}, creating numericalised file from dataset at {parameters["train_skipgram_syns_data"]} and counts file at {parameters["counts_file"]}')
        numericalise_dataset(parameters['train_skipgram_syns_data'], parameters['num_train_skipgram_syns_data'], VOCABULARY)
    else:
        print(f'Numericalised Skip gram training augmented data file found at {parameters["num_train_skipgram_syns_data"]}')
    
    # NUMERICALISING VALIDATION SKIPGRAM DATA
    if not os.path.exists(parameters['num_val_skipgram_data']):
        print(f'No numericalised Skip gram validation data file found at {parameters["num_val_skipgram_data"]}, creating numericalised file from dataset at {parameters["val_skipgram_data"]} and counts file at {parameters["counts_file"]}')
        numericalise_dataset(parameters['val_skipgram_data'], parameters['num_val_skipgram_data'], VOCABULARY)
    else:
        print(f'Numericalised Skip gram validation data file found at {parameters["num_val_skipgram_data"]}')
    
    # NUMERICALISE VALIDATION AUGMENTED SKIPGRAM DATA
    if not os.path.exists(parameters['num_val_skipgram_syns_data']):
        print(f'No numericalised Skip gram validation augmented data file found at {parameters["num_val_skipgram_syns_data"]}, creating numericalised file from dataset at {parameters["val_skipgram_syns_data"]} and counts file at {parameters["counts_file"]}')
        numericalise_dataset(parameters['val_skipgram_syns_data'], parameters['num_val_skipgram_syns_data'], VOCABULARY)
    else:
        print(f'Numericalised Skip gram validation augmented data file found at {parameters["num_val_skipgram_syns_data"]}')