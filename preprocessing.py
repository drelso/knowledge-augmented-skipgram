###
#
# Construct SkipGram dataset
#
###v

import os
import time
import datetime

from config import parameters

from utils.dataset_utils import shuffle_and_subset_dataset, process_bnc_data, numericalise_dataset, train_validate_split, basic_tokenise, word_counts, dataset_sampling, select_synonyms, numeric_csv_to_npy, process_all_datafiles, seqlist_from_raw_text, seqlist_to_skipgram_data, skipgram_sampling, augment_npy_skipgram
from utils.funcs import dir_validation, print_parameters
from utils.training_utils import build_vocabulary

import numpy as np
import torch


if __name__ == '__main__':
    print_parameters(parameters)

    # DATA_PATH = parameters['bnc_subset_data'] if parameters['use_data_subset'] else parameters['bnc_data']
    # TAGS_PATH = parameters['bnc_subset_tags'] if parameters['use_data_subset'] else parameters['bnc_tags']
    # COUNTS_FILE = parameters['bnc_subset_counts'] if parameters['use_data_subset'] else parameters['bnc_counts']
    # print_parameters(parameters)
    # exit()

    # PROCESS ALL TEXT FILES AND SAVE TO A SINGLE
    # RAW TEXT FILE
    if not os.path.exists(parameters['bnc_data']):
        print(f'No processed file found at {parameters["bnc_data"]}, creating single simple text dataset file from XML files at {parameters["bnc_texts_dir"]}')
        process_all_datafiles(
            parameters['bnc_texts_dir'],
            parameters['bnc_data'],
            tags_savefile=parameters['bnc_tags'],
            use_headwords=False,
            replace_nums=False,
            replace_unclass=False)
    else:
        print(f'Processed simple text file found at {parameters["bnc_data"]}')

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

    ## CONVERT RAW TEXT SENTENCES TO SEQUENCE LIST
    if not os.path.exists(parameters['tokenised_data']):
        bnc_data = parameters['bnc_subset_data'] if parameters['use_data_subset'] else parameters["bnc_data"]
        print(f'Processing sequence lists for dataset at {bnc_data}, saving to {parameters["tokenised_data"]}')
        seqlist_from_raw_text(
            bnc_data,
            parameters['tokenised_data'],
            to_lower=parameters['to_lower'],
            replace_num=parameters['replace_num'],
            remove_punct=parameters['remove_punct'])
    else:
        print(f'Found existing dependency trees and sequence lists dataset file at {parameters["tokenised_data"]}\n')
    

    ## DATASET WORD COUNTS
    if not os.path.exists(parameters['counts_file']):
        print(f'Calculating word counts for tokenised dataset at {parameters["tokenised_data"]}')
        # tokenised_data = basic_tokenise(parameters['raw_data'], preserve_sents=True)
        # Load tokenised data, without POSl k tags
        tokenised_data = [[w[0] for w in sent] for sent in np.load(parameters['tokenised_data'])]
        word_counts(tokenised_data, parameters['counts_file'])
    else:
        print(f'Found existing word counts file at {parameters["counts_file"]}\n')
    
    ## SPLIT DATASET
    if not os.path.exists(parameters['train_data']) \
        or not os.path.exists(parameters['val_data']):
        train_validate_split(
            parameters['tokenised_data'],
            parameters['train_data'],
            parameters['val_data'],
            proportion=parameters['split_ratio'])
    else:
        print(f'Found existing train/validation datasets at: \n - {parameters["train_data"]} \n - {parameters["val_data"]}')
    

    # CONSTRUCT SKIP-GRAM DATASET
    if not os.path.exists(parameters['train_skipgram_data']):
        print(f'Constructing Skip gram training dataset at {parameters["train_skipgram_data"]} from tokenised training data at {parameters["train_data"]}')

        seqlist_to_skipgram_data(
            parameters["train_data"],
            parameters["train_skipgram_data"],
            ctx_size=parameters['ctx_size'],
            write_batch=50000)

    if not os.path.exists(parameters['val_skipgram_data']):
        print(f'Constructing Skip gram validation dataset at {parameters["val_skipgram_data"]} from tokenised validation data at {parameters["val_data"]}')

        seqlist_to_skipgram_data(
            parameters["val_data"],
            parameters["val_skipgram_data"],
            ctx_size=parameters['ctx_size'],
            write_batch=50000)
    

    ## SAMPLE CONTEXT WORDS AND SAVE TO FILE
    if not os.path.exists(parameters['train_skipgram_sampled_data']):
        print(f'Constructing context-sampled skip gram training dataset at {parameters["train_skipgram_sampled_data"]} from {parameters["train_skipgram_data"]}')

        skipgram_sampling(
            parameters["train_skipgram_data"],
            parameters["train_skipgram_sampled_data"],
            max_context=parameters['ctx_size'])
    else:
        print(f'Context-sampled skip gram training data file found at {parameters["train_skipgram_sampled_data"]}')
    
    if not os.path.exists(parameters['val_skipgram_sampled_data']):
        print(f'Constructing context-sampled skip gram validation dataset at {parameters["val_skipgram_sampled_data"]} from {parameters["val_skipgram_data"]}')

        skipgram_sampling(
            parameters["val_skipgram_data"],
            parameters["val_skipgram_sampled_data"],
            max_context=parameters['ctx_size'])
    else:
        print(f'Context-sampled Skip-gram validation data file found at {parameters["val_skipgram_sampled_data"]}')

    
    ## CONSTRUCT VOCABULARY OBJECT FROM COUNTS FILE
    VOCABULARY = build_vocabulary(parameters['counts_file'], min_freq=parameters['vocab_cutoff'])


    # CONSTRUCT AUGMENTED DATASETS
    # NOTE: augmented validation dataset might not be necessary
    if parameters['syn_augm']:
        if not os.path.exists(parameters['train_skipgram_augm_data']):
            print(f'Augmenting Skip-gram training dataset at {parameters["train_skipgram_augm_data"]} from sampled training data at {parameters["train_skipgram_sampled_data"]}')
            
            augment_npy_skipgram(
                parameters["train_skipgram_sampled_data"],
                parameters['train_skipgram_augm_data'],
                VOCABULARY,
                ctx_size=parameters['ctx_size'],
                syn_selection=parameters['synonym_selection'])
        else:
            print(f'Augmented Skip-gram training data file found at {parameters["train_skipgram_augm_data"]}')
        
        if not os.path.exists(parameters['val_skipgram_augm_data']):
            print(f'Augmenting Skip-gram validation dataset at {parameters["val_skipgram_augm_data"]} from sampled validationg data at {parameters["val_skipgram_sampled_data"]}')
            
            augment_npy_skipgram(
                parameters["val_skipgram_sampled_data"],
                parameters['val_skipgram_augm_data'],
                VOCABULARY,
                ctx_size=parameters['ctx_size'],
                syn_selection=parameters['synonym_selection'])
        else:
            print(f'Augmented Skip-gram validation data file found at {parameters["val_skipgram_augm_data"]}')

    # NUMERICALISING SKIPGRAM DATA
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if parameters['num_to_tensor']:
        print(f'Numericalised files as PyTorch tensors for device {DEVICE}')

    # TRAINING
    if not os.path.exists(parameters['num_train_skipgram_sampled_data']):
        print(f'No numericalised Skip-gram training sampled data file found at {parameters["num_train_skipgram_sampled_data"]}, creating numericalised file from dataset at {parameters["train_skipgram_sampled_data"]}')

        numericalise_dataset(
            parameters['train_skipgram_sampled_data'],
            parameters['num_train_skipgram_sampled_data'],
            VOCABULARY,
            to_tensor=parameters['num_to_tensor'],
            device=DEVICE)
    else:
        print(f'Numericalised Skip gram training sampled data file found at {parameters["num_train_skipgram_sampled_data"]}')
    
    # TRAINING AUGMENTED
    if not os.path.exists(parameters['num_train_skipgram_augm_data']):
        print(f'No numericalised Skip-gram training augmented data file found at {parameters["num_train_skipgram_augm_data"]}, creating numericalised file from dataset at {parameters["train_skipgram_augm_data"]}')

        numericalise_dataset(
            parameters['train_skipgram_augm_data'],
            parameters['num_train_skipgram_augm_data'],
            VOCABULARY,
            to_tensor=parameters['num_to_tensor'],
            device=DEVICE)
    else:
        print(f'Numericalised Skip-gram training augmented data file found at {parameters["num_train_skipgram_augm_data"]}')

    # VALIDATION
    if not os.path.exists(parameters['num_val_skipgram_sampled_data']):
        print(f'No numericalised Skip-gram validation sampled data file found at {parameters["num_val_skipgram_sampled_data"]}, creating numericalised file from dataset at {parameters["val_skipgram_sampled_data"]}')

        numericalise_dataset(
            parameters['val_skipgram_sampled_data'],
            parameters['num_val_skipgram_sampled_data'],
            VOCABULARY,
            to_tensor=parameters['num_to_tensor'],
            device=DEVICE)
    else:
        print(f'Numericalised Skip gram training sampled data file found at {parameters["num_val_skipgram_sampled_data"]}')
    
    # VALIDATION AUGMENTED
    if not os.path.exists(parameters['num_val_skipgram_augm_data']):
        print(f'No numericalised Skip-gram validation augmented data file found at {parameters["num_val_skipgram_augm_data"]}, creating numericalised file from dataset at {parameters["val_skipgram_augm_data"]}')

        numericalise_dataset(
            parameters['val_skipgram_augm_data'],
            parameters['num_val_skipgram_augm_data'],
            VOCABULARY,
            to_tensor=parameters['num_to_tensor'],
            device=DEVICE)
    else:
        print(f'Numericalised Skip-gram validation augmented data file found at {parameters["num_val_skipgram_augm_data"]}')
