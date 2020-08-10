###
#
# Model validation process
#
###

import os
import time

from config import parameters
from utils.funcs import print_parameters, dir_validation
from utils.training_utils import build_vocabulary, construct_dataset_splits

import torch
import torch.nn as nn
import torch.optim as optim

from skipgram.train import train_augm_w2v
from skipgram.nn import SkipGram
from skipgram.utils import init_sample_table, new_neg_sampling, save_param_to_npy

import torchtext
from torchtext.data import Dataset, Field, Iterator, BucketIterator


if __name__ == '__main__':

    parameters['all_models_dir'] = dir_validation(parameters['all_models_dir'])
    parameters['model_dir'] = dir_validation(parameters['model_dir'])
    parameters['checkpoints_dir'] = dir_validation(parameters['checkpoints_dir'])

    start_time = time.time()
    
    print_parameters(parameters)
    
    # DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DEVICE = torch.device('cpu') # TODO REMOVE THIS LINE, ONLY FOR DEBUGGING
    print(f"Running on device: {DEVICE}")
    # if torch.cuda.is_available():
    #     torch.set_default_tensor_type(torch.cuda.FloatTensor)

    ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ## VOCABULARY CONSTRUCTION AND DATASET NUMERICALISATION
    ##
    # CONSTRUCT VOCABULARY
    vocabulary = build_vocabulary(parameters['bnc_counts'], min_freq=parameters['vocab_cutoff'])
    # parameters['input_dim'] = len(vocabulary)
    input_dim = len(vocabulary)

    # ## ONLY NUMERICALISE THE DATA RIGHT IF NO EXISTING FILE IS FOUND
    # if not os.path.exists(parameters['bnc_num_skipgram_data']):
    #     print(f'No numericalised file found at {parameters["bnc_num_skipgram_data"]}, creating numericalised file from dataset at {parameters["bnc_skipgram_data"]}')
    #     numericalise_dataset(parameters['bnc_skipgram_data'], parameters['bnc_num_skipgram_data'], vocabulary)
    # else:
    #     print(f'Numericalised file found at {parameters["bnc_num_skipgram_data"]}')
    # ## NUMERICALISE AUGMENTED DATA
    # if not os.path.exists(parameters['bnc_num_skipgram_augm_data']):
    #     print(f'No numericalised file found at {parameters["bnc_num_skipgram_augm_data"]}, creating numericalised file from dataset at {parameters["bnc_skipgram_data"]}')
    #     numericalise_dataset(parameters['bnc_skipgram_data'], parameters['bnc_num_skipgram_augm_data'], vocabulary)
    # else:
    #     print(f'Numericalised file found at {parameters["bnc_num_skipgram_augm_data"]}')
    
    ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ## MODEL AND TRAINING INITIALISATION
    model = SkipGram(input_dim, parameters['embedding_size'], w2v_init=parameters['w2v_init'], w2v_path=parameters['w2v_path'])
    
    print(f'\n {"=" * 25} \n TRAINABLE PARAMETERS \n {"=" * 25} \n ')
    for name, param in model.named_parameters():
        print(name)
        if param.requires_grad:
            print('\t -> requires grad')
        else:
            print('\t -> NO grad')
    
    optimiser = optim.SGD(model.parameters(), lr=parameters['learning_rate'])

    print(f'vocabulary length: {len(vocabulary)} \t first word: {vocabulary.itos[0]}')
    
    ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ## LOAD AND SPLIT DATASET
    # 'SAMPLE_bnc_full_seqlist_deptree_numeric_voc-1.json'
    # train_data, test_data, val_data = construct_dataset_splits(parameters['num_data_save_path'], vocabulary, split_ratios=parameters['split_ratios'])
    # train_data, val_data = construct_dataset_splits(parameters['bnc_skipgram_data'], vocabulary, split_ratio=parameters['split_ratio'])
    
    # print(f'\nFirst example train: {train_data.examples[0].focus_word} \t {train_data.examples[0].context_word}')

    # train_iter, val_iter = Iterator(
    #     (train_data, val_data),
    #     # sort=False,
    #     sort_key=lambda x: len(x.focus_word),
    #     shuffle=True,
    #     batch_size=(parameters['batch_size'], parameters['batch_size']),
    #     device=DEVICE
    # )

    # for epoch in range(parameters['epochs']):
    #     for i, sample in enumerate(train_data):
    #         print(sample)
    #         print(sample.focus_word, sample.context_word)
    #         if i > 4: break


    '''
    ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ## BUILD DATA BATCHES
    train_iter, val_iter = BucketIterator.splits(
        (train_data, val_data),
        batch_sizes=(parameters['batch_size'], parameters['batch_size']),
        # device=parameters['device'],
        device=DEVICE,
        sort=parameters['sort_train_val_data'],
        # sort_within_batch=True,
        sort_key=lambda x: len(x.seq),
        shuffle=parameters['shuffle_train_val_data'],
        repeat=parameters['repeat_train_val_iter']
    )

    test_iter = Iterator(
        test_data,
        batch_size=parameters['batch_size'],
        # device=parameters['device'],
        device=DEVICE,
        repeat=parameters['repeat_train_val_iter']
    )
    
    for epoch in range(parameters['num_epochs']):
        print(f'\n\n &&&&&&&&&&&&& \n ############# \n \t\t\t EPOCH ======> {epoch} \n &&&&&&&&&&&&& \n ############# \n\n')

        print_epoch = not epoch % math.ceil(parameters['num_epochs'] / 10)
        
        epoch_start_time = time.time()

        # print(f'{"-" *30} \n MEMORY STATS EPOCH {epoch} **PRE RUN** \n {"-" *30} \n ')
        # memory_stats(device=DEVICE)

        print(f'\n Epoch {epoch} training... \n')
        epoch_loss = run_model(train_iter, model, optimizer, criterion, vocabulary, device=DEVICE, phase='train', print_epoch=print_epoch)
        
        checkpoints_file = parameters['checkpoints_path'] + '_epoch' + str(epoch) + '-chkpt.tar'
        print('Saving checkpoint file: %r \n' % (checkpoints_file))
        
        # print(f'{"-" *30} \n MEMORY STATS EPOCH {epoch} **PRE SAVE** \n {"-" *30} \n ')
        # memory_stats(device=DEVICE)

        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss
                }, checkpoints_file)
        
        # print(f'{"-" *30} \n MEMORY STATS EPOCH {epoch} **POST TRAIN** \n {"-" *30} \n ')
        # memory_stats(device=DEVICE)

        print(f'\n Epoch {epoch} validation... \n')
        val_epoch_loss = run_model(val_iter, model, optimizer, criterion, vocabulary, device=DEVICE, phase='val', print_epoch=print_epoch)

        # print(f'{"-" *30} \n MEMORY STATS EPOCH {epoch} **POST VAL** \n {"-" *30} \n ')
        # memory_stats(device=DEVICE)

        if print_epoch:
            elapsed_time = time.time() - epoch_start_time
            print(f'Elapsed time in epoch {epoch}: {elapsed_time}' )
            print(f'Iteration {epoch} \t Loss: {epoch_loss} \t Validation loss: {val_epoch_loss}')
    
    print('\n\nSaving model to ', parameters['model_path'] )
    # A common PyTorch convention is to save models using
    # either a .pt or .pth file extension.
    torch.save(model.state_dict(), parameters['model_path'] )
    #model.load_state_dict(torch.load(parameters['model_path'] ))

    param_name = 'encoder.word_embedding'
    save_param_to_npy(model, param_name, parameters['word_embs_path'])
    
    '''


    elapsed_time = time.time() - start_time
    print(f'{"=" * 20} \n\t Total elapsed time: {elapsed_time} \n {"=" * 20} \n')












    '''
    train_augm_w2v( parameters['dataset_file'],
                    parameters['vocab_file'], 
                    parameters['syn_file'], 
                    parameters['model_file'], 
                    parameters['checkpoints_folder'], 
                    parameters['validation_file'], 
                    embedding_size=parameters['embedding_size'], 
                    epochs=parameters['epochs'],
                    batch_size=parameters['batch_size'], 
                    num_neg_samples=parameters['num_neg_samples', 
                    learning_rate=parameters['learning_rate'], 
                    w2v_init=parameters['w2v_init'], 
                    w2v_path=parameters['w2v_path'], 
                    syn_augm=parameters['syn_augm'], 
                    emb_npy_file=parameters['input_emb_file'], 
                    data_augmentation_ratio=parameters['data_augmentation_ratio'])

    elapsed_time = time.time() - start_time
    print('\nTotal elapsed time: ', elapsed_time)
    '''