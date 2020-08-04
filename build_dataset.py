###
#
# Construct SkipGram dataset
# from Gutenberg books
#
###

from config import parameters, print_parameters
from utils.dataset_utils import process_bnc_data
import time
import datetime




if __name__ == '__main__':
    
    process_bnc_data(parameters['bnc_data'], parameters['bnc_skipgram_data'], tags_file=parameters['bnc_data_tags'], augm_dataset_file=parameters['bnc_skipgram_augm_data'], ctx_size=5, write_batch=10000)

    # skipgram_data_from_gutenberg(parameters['gutenberg_dir'], processed_books_path, vocab_file, syn_sel_file, num_books=10)