###
#
# Model validation process
#
###

import os
import sys
import time
import csv
import random
import numpy as np

from config import parameters
from utils.funcs import print_parameters, dir_validation, memory_usage
from utils.dataset_utils import csv_reader_check_header

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

    start_time = time.time()
    
    print_parameters(parameters)
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {DEVICE}")
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    start_time = time.time()
    
    memory_usage(legend=0) # MEMORY DEBUGGING

    # MEMORY-MAPPED DATASET READING 
    data        = np.load(parameters['num_train_skipgram_npy'], mmap_mode='r')
    syns        = np.load(parameters['num_train_skipgram_syns_npy'], mmap_mode='r')
    val_data    = np.load(parameters['num_val_skipgram_npy'], mmap_mode='r')

    memory_usage(legend='Loading data (memmap)') # MEMORY DEBUGGING

    num_data = data.shape[0]
    num_syns = syns.shape[0]
    num_val_data = val_data.shape[0]
    print("Data dims:", data.shape)
    print("Syns dims:", syns.shape)
    print("Val:", val_data.shape)

    with open(parameters['counts_file'], 'r', encoding='utf-8', errors='replace') as v:
        vocab_reader    = csv_reader_check_header(v)
        
        vocabulary = [w for w in vocab_reader]
    vocab_words = [w[0] for w in vocabulary]
    vocab_counts = [int(w[1]) for w in vocabulary]
    
    memory_usage(legend='Reading vocabulary') # MEMORY DEBUGGING

    # Calculate the vocabulary ratios
    # Elevate counts to the 3/4th power
    pow_counts = np.array(vocab_counts)**0.75
    normaliser = sum(pow_counts)
    # Normalise the counts
    vocab_ratios = pow_counts / normaliser
    
    memory_usage(legend='Vocab counts and ratios') # MEMORY DEBUGGING

    sample_table = init_sample_table(vocab_counts)
    memory_usage(legend='Sample table') # MEMORY DEBUGGING
    
    print('Size of sample table: ', sample_table.size)
    print('Total distinct words: ', len(vocabulary))
    print('Samples from vocab: ', vocabulary[:5])

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
    memory_usage(legend=5) # MEMORY DEBUGGING
    
    FOCUS_COL = 0
    CONTEXT_COL = 1

    ix_data = 0
    ix_syn = 0
    batch = []
    focus_ixs = []
    context_ixs = []

    model = SkipGram(len(vocabulary), parameters['embedding_size'], w2v_init=parameters['w2v_init'], w2v_path=parameters['w2v_path'])    
    if DEVICE == torch.device('cuda'):
        model.cuda()
        
    optimiser = optim.SGD(model.parameters(),lr=parameters['learning_rate'])
    
    losses = []
    val_losses = []
    times = []

    for epoch in range(parameters['epochs']):
        print(f'\n {"#" * 24} \n \t\t EPOCH NUMBER {epoch} \n {"#" * 24}')

        # RE-CALCULATE AT THE BEGINNING OF EACH EPOCH
        data_ixs = np.random.choice(num_data, num_data, replace=False)
        memory_usage(legend='data ixs') # MEMORY DEBUGGING
        syns_ixs = np.random.choice(num_syns, num_syns, replace=False)
        memory_usage(legend='syns ixs') # MEMORY DEBUGGING
        val_data_ixs = np.random.choice(num_val_data, num_val_data, replace=False)
        memory_usage(legend='val ixs') # MEMORY DEBUGGING
        
        ## TRAINING PHASE
        print('Training...')
        model.train()
        num_batches = 0
        batches_loss = 0.
        for select_syn in augmentation_selector:
            if select_syn:
                datapoint = syns[syns_ixs[ix_syn]]
                # RESTART THE INDEX IF DATASET IS EXHAUSTED
                ix_syn = ix_syn + 1 if ix_syn < num_syns else 0
            else:
                datapoint = data[data_ixs[ix_data]]
                # RESTART THE INDEX IF DATASET IS EXHAUSTED
                ix_data = ix_data + 1 if ix_data < num_data else 0
            
            focus_ixs.append(datapoint[FOCUS_COL])
            context_ixs.append(datapoint[CONTEXT_COL])
            
            if len(focus_ixs) == parameters['batch_size']:
                batches_loss += process_word_pair_batch(focus_ixs, context_ixs, model, optimiser, sample_table, parameters['num_neg_samples'], parameters['batch_size'], phase='train')
                
                num_batches += 1
                focus_ixs = []
                context_ixs = []

                if ix_syn > 10: break # REMOVE
        epoch_loss = batches_loss / num_batches
        losses.append(epoch_loss)
        
        memory_usage(legend='After training') # MEMORY DEBUGGING

        ## VALIDATION PHASE
        print('Validation...')
        model.eval()
        num_batches = 0
        batches_loss = 0.
        with torch.no_grad():
            i_break = 0 # REMOVE
            for ix in val_data_ixs:
                i_break += 1
                datapoint = val_data[ix]
                
                focus_ixs.append(datapoint[FOCUS_COL])
                context_ixs.append(datapoint[CONTEXT_COL])
                
                if len(focus_ixs) == parameters['batch_size']:
                    batches_loss += process_word_pair_batch(focus_ixs, context_ixs, model, optimiser, sample_table, parameters['num_neg_samples'], parameters['batch_size'], phase='validate')

                    num_batches += 1
                    focus_ixs = []
                    context_ixs = []
                if i_break > 10: break # REMOVE
        val_loss = batches_loss / num_batches
        val_losses.append(val_loss)
        print(f'\n {">" * 16} \t EPOCH LOSS: {epoch_loss} \t VAL LOSS: {val_loss} {"<" * 16} \n')
        
        memory_usage(legend='After validation') # MEMORY DEBUGGING
        
        # FREE-UP SOME MEMORY
        memory_usage(legend='before freeing memory') # MEMORY DEBUGGING
        del data_ixs
        del syns_ixs
        del val_data_ixs
        gc.collect()
        memory_usage(legend='after freeing memory') # MEMORY DEBUGGING

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

        memory_usage(legend='After saving checkpoint') # MEMORY DEBUGGING
      
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