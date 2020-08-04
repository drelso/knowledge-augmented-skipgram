###
#
# Word2Vec utility functions
#
###

import gensim.downloader as api
import csv
import numpy as np

def get_word_vector(word, model):
    try:
        model[word]
    except:
        #print('Word not found')
        return False
    return model[word]
    
    
def word2vec_with_vocab(vocab_file, save_file, save_npy_file):
    
    model = api.load("word2vec-google-news-300")
    
    missing_words = 0
    
    with open(vocab_file,'r') as v, \
        open(save_file, 'w+') as f:
        
        word_vec_dim = 300
        
        vocab_data = csv.reader(v)
        wr = csv.writer(f)
        
        # i = 0
        
        word_vec_arr = []
        word_vectors = []
        
        for row in vocab_data:
            word_vec = get_word_vector(row[0], model)
            
            if isinstance(word_vec, bool):
                missing_words += 1
                # print(row[0])
                word_vec = [0] * word_vec_dim
            
            vec = [d for d in word_vec]
            temp = [row[0]] + vec
            word_vec_arr.append(temp)
            word_vectors.append(vec)
            
            # i += 1
            # if i > 100: break
        
        wr.writerows(word_vec_arr)
        np.save(save_npy_file, word_vectors)
        
    print('Missing %d words' % (missing_words))
    