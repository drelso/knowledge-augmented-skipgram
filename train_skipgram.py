###
#
# Model validation process
#
###

import time

from skipgram.train import train_augm_w2v
from config import parameters, print_parameters

import os

if __name__ == '__main__':

    start_time = time.time()
    
    print_parameters()

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