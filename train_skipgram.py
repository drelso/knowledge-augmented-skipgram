###
#
# Model validation process
#
###

import os
import sys
import shutil
from pathlib import Path
import time
import csv
import random
import numpy as np

from config import parameters
from utils.funcs import print_parameters, dir_validation, mem_check
from utils.dataset_utils import build_vocabulary, csv_reader_check_header

import torch
import torch.nn as nn
import torch.optim as optim

from skipgram.train import train_augm_w2v
from skipgram.nn import SkipGram
from skipgram.utils import init_sample_table, new_neg_sampling, save_param_to_npy, process_word_pair_batch

import gc



if __name__ == '__main__':

    parameters['all_models_dir'] = dir_validation(parameters['all_models_dir'])
    parameters['model_dir'] = dir_validation(parameters['model_dir'])
    parameters['checkpoints_dir'] = dir_validation(parameters['checkpoints_dir'])

    home = str(Path.home())
    # CONFIG_FILE_PATH = home + '/Scratch/knowledge-augmented-skipgram/config.py' # TODO: CHANGE FOR DIS FILESYSTEM
    CONFIG_FILE_PATH = home + '/knowledge-augmented-skipgram/config.py' # TODO: CHANGE FOR DIS FILESYSTEM
    shutil.copy(CONFIG_FILE_PATH, parameters['model_dir'])
    print(f'Copied config file {CONFIG_FILE_PATH} to {parameters["model_dir"]}')

    start_time = time.time()
    
    print_parameters(parameters)
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {DEVICE}")
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    
    # MEMORY-MAPPED DATASET READING 
    # data        = np.load(parameters['num_train_skipgram_npy'], mmap_mode='r')
    # syns        = np.load(parameters['num_train_skipgram_syns_npy'], mmap_mode='r')
    # val_data    = np.load(parameters['num_val_skipgram_npy'], mmap_mode='r')

    data        = torch.load(parameters['num_train_skipgram_sampled_data'])
    syns        = torch.load(parameters['num_train_skipgram_augm_data'])
    val_data    = torch.load(parameters['num_val_skipgram_sampled_data'])

    num_data = data.shape[0]
    num_syns = syns.shape[0]
    num_val_data = val_data.shape[0]
    print("Data dims:", data.shape)
    print("Syns dims:", syns.shape)
    print("Val:", val_data.shape)
    
    VOCABULARY = build_vocabulary(parameters['counts_file'], parameters['vocabulary_indices'], min_freq=parameters['vocab_cutoff'])
    print(f'Constructed vocabulary with {len(VOCABULARY)} distinct tokens from file at {parameters["counts_file"]}')

    vocab_words = VOCABULARY.itos
    print('vocab_words[:10]:', vocab_words[:10])
    
    vocab_counts = [VOCABULARY.freqs[w] for w in vocab_words]
    print('vocab_counts:', vocab_counts[:10])

    print(f'checking frequencies: freqs["and"] ---> {VOCABULARY.freqs["and"]} == {vocab_counts[vocab_words.index("and")]}')

    # Calculate the vocabulary ratios
    # Elevate counts to the 3/4th power
    pow_counts = np.array(vocab_counts)**0.75
    normaliser = sum(pow_counts)
    # Normalise the counts
    vocab_ratios = pow_counts / normaliser
    
    sample_table = init_sample_table(vocab_counts)
    
    print('Size of sample table: ', sample_table.size)
    print('Total distinct words: ', len(VOCABULARY))

    # SYNONYM SWITCH LIST: BOOLEAN LIST TO RANDOMLY DETERMINE
    # WHEN TO PROCESS SYNONYMS AND WHEN TO PROCESS NATURAL
    # WORD PAIRS
    # Rough ratio of "natural" vs. augmented examples
    # Current dataset sizes:
    #   2.2G num_voc-5_skipgram_bnc_full_proc_data_shffl_sub-5_train.csv
    #   278,999,805 word pairs (69.715% of the combined dataset)
    #   970M num_voc-5_syns-sw_sampled_skipgram_bnc_full_proc_data_shffl_sub-5_train.csv
    #   112,200,002 word pairs (30.285% of the combined dataset)
    natural_data_ratio = 1 - parameters['data_augmentation_ratio']

    augmented_dataset_size = int(num_data + (num_data * parameters['data_augmentation_ratio']))
    augmentation_selector = np.random.choice([True,False], augmented_dataset_size, p=[parameters['data_augmentation_ratio'], natural_data_ratio])
    
    augmented_dataset_num_batches = augmented_dataset_size / parameters['batch_size']

    FOCUS_COL = 0
    CONTEXT_COL = 1

    model = SkipGram(
                len(VOCABULARY),
                parameters['embedding_size'],
                w2v_init=parameters['w2v_init'],
                w2v_path=parameters['w2v_path'])    
    
    if DEVICE == torch.device('cuda'): model.cuda()
        
    optimiser = optim.SGD(model.parameters(),lr=parameters['learning_rate'])
    
    ## LOADING MODELS
    if parameters['load_model']:
        if parameters['load_model'].find('checkpoints') > -1:
            # LOAD CHECKPOINT
            print(f'Loading checkpoint file from: {parameters["load_model"]} \n')
            checkpoint = torch.load(parameters['load_model'])
            model_sd = checkpoint['model_state_dict']
            # optim_sd = checkpoint['optimizer_state_dict']
        else:
            print(f'Loading model file from: {parameters["load_model"]} \n')
            model_sd = torch.load(parameters['model_file'])

        model.load_state_dict(model_sd)

    losses = []
    val_losses = []
    times = []
    PRINT_INTERVAL = 20
    
    elapsed_time = time.time() - start_time
    print(f'Elapsed time before training: {elapsed_time}', flush=True)
    
    for epoch in range(parameters['epochs']):
        print(f'\n {"#" * 24} \n \t\t EPOCH NUMBER {epoch} \n {"#" * 24}')

        ix_data = 0
        ix_syn = 0
        batch = []
        focus_ixs = []
        context_ixs = []
        
        # RE-CALCULATE AT THE BEGINNING OF EACH EPOCH
        data_ixs = np.random.choice(num_data, num_data, replace=False)
        syns_ixs = np.random.choice(num_syns, num_syns, replace=False)
        val_data_ixs = np.random.choice(num_val_data, num_val_data, replace=False)
        
        ## TRAINING PHASE
        print(f'Training ({augmented_dataset_size} word pairs)...')
        model.train()
        num_batches = 0
        batches_loss = 0.
        for i, select_syn in enumerate(augmentation_selector):
            if select_syn:
                datapoint = syns[syns_ixs[ix_syn]]
                # RESTART THE INDEX IF DATASET IS EXHAUSTED
                ix_syn = ix_syn + 1 if ix_syn < num_syns-1 else 0
            else:
                datapoint = data[data_ixs[ix_data]]
                # RESTART THE INDEX IF DATASET IS EXHAUSTED
                ix_data = ix_data + 1 if ix_data < num_data-1 else 0
            
            focus_ixs.append(datapoint[FOCUS_COL])
            context_ixs.append(datapoint[CONTEXT_COL])
            
            if len(focus_ixs) == parameters['batch_size']:
                batches_loss += process_word_pair_batch(focus_ixs, context_ixs, model, optimiser, sample_table, parameters['num_neg_samples'], parameters['batch_size'], phase='train')
                
                if not num_batches % int(augmented_dataset_num_batches / PRINT_INTERVAL):
                    elapsed_time = time.time() - start_time
                    print(f'{num_batches}/{augmented_dataset_num_batches} batches processed (elapsed time: {elapsed_time})', flush=True)

                num_batches += 1
                focus_ixs = []
                context_ixs = []
                
        epoch_loss = batches_loss / num_batches
        losses.append(epoch_loss)
        
        ## VALIDATION PHASE
        print(f'Validation ({num_val_data} word pairs)...')
        model.eval()
        num_batches = 0
        batches_loss = 0.
        with torch.no_grad():
            for i, ix in enumerate(val_data_ixs):
                datapoint = val_data[ix]
                
                focus_ixs.append(datapoint[FOCUS_COL])
                context_ixs.append(datapoint[CONTEXT_COL])
                
                if not i % int(num_val_data / PRINT_INTERVAL):
                    print(f'{i}/{num_val_data} lines processed', flush=True)
                
                if len(focus_ixs) == parameters['batch_size']:
                    batches_loss += process_word_pair_batch(focus_ixs, context_ixs, model, optimiser, sample_table, parameters['num_neg_samples'], parameters['batch_size'], phase='validate')

                    num_batches += 1
                    focus_ixs = []
                    context_ixs = []
        val_loss = batches_loss / num_batches
        val_losses.append(val_loss)
        print(f'\n {">" * 16} \t EPOCH LOSS: {epoch_loss} \t VAL LOSS: {val_loss} {"<" * 16} \n')
        
        # FREE-UP SOME MEMORY
        del data_ixs
        del syns_ixs
        del val_data_ixs
        gc.collect() # GARBAGE COLLECT THE DELETED VARIABLES

        elapsed_time = time.time() - start_time
        times.append(elapsed_time)
        print(f'Elapsed time after epoch {epoch}: {elapsed_time}', flush=True)

        # SAVING A CHECKPOINT
        checkpoints_file = parameters['checkpoints_dir'] + str(epoch) + '-epoch-chkpt.tar'
        print(f'Saving checkpoint file: {checkpoints_file} \n')
        
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimiser.state_dict(),
                'loss': epoch_loss,
                'val_loss': val_loss
                }, checkpoints_file)
      
    print(f'\n\nSaving model to {parameters["model_file"]}')
    # A common PyTorch convention is to save models using
    # either a .pt or .pth file extension.
    torch.save(model.state_dict(), parameters['model_file'])
    #model.load_state_dict(torch.load(parameters['model_file']))
    
    if parameters['input_emb_file'] is not None:
        # Save input embeddings to a NPY file
        param_name = 'i_embedding'
        save_param_to_npy(model, param_name, parameters['input_emb_file'])
    avg_time = np.mean(times)
    
    print('Train losses:')
    print(losses)
    
    print('Validation losses:')
    print(val_losses)
    
    print('Average run time per epoch: ', avg_time)

    elapsed_time = time.time() - start_time
    print(f' {"=" * 40} \n\t Total elapsed time: {elapsed_time} \n {"=" * 40} \n')