###
#
# WMD-Relax + spaCy pipeline on 20 News Groups
#
# Multiprocessing tested for HPC cluster
#
# NOTE: This solves the problem produced by
#       quoting the words in the vocabulary ("")
#
# Resources:
# https://spacy.io/universe/project/wmd-relax
# https://spacy.io/usage/vectors-similarity#custom
#
###

import wmd
import spacy
from collections import Counter
import numpy as np

from sklearn.datasets import fetch_20newsgroups

import time
import csv
import sys
import os
import math

import threading
import multiprocessing
from multiprocessing import Pool, Process

import subprocess
import shlex
import psutil


# Loading model for our own word embeddings
# To load new embeddings make sure they are
# in a text file, columns separated by a single
# space, where the first line contains two values:
#   - Number of embeddings
#   - Embedding dimensionality
# and the first column corresponds to the 
# unquoted word (watch out for white space
# and special characters which might cause
# problems). Load this file into temp to
# access it from spaCy:

# > python3 -m spacy init-model en /tmp/w2v_nosyns_voc20 --vectors-loc model/w2v_init-no_syns-10e-voc20-emb300/w2v_init-no_syns-10e-voc20-emb300.tsv

# Then, load the saved model with spaCy, within a
# WMD pipeline
# embs_model = '/tmp/w2v_nosyns_voc20'


def init_spacy(model_name, is_base=False):
    
    embs_dir = '/tmp/' + model_name
    
    if not os.path.exists(embs_dir):
        spacy_init_model(embs_dir, model_name, is_base=is_base)
    
    nlp = spacy.load(embs_dir, create_pipeline=wmd.WMD.create_spacy_pipeline)
    # nlp = spacy.load('en_core_web_md', create_pipeline=wmd.WMD.create_spacy_pipeline)
    nlp.add_pipe(wmd.WMD.SpacySimilarityHook(nlp), last=True)
    
    print('Loaded model into spaCy: \t', embs_dir)
    
    spacy_embs = SpacyEmbeddings(nlp)
    print('Initialised spaCy embeddings')
    
    return spacy_embs, nlp
    

# Hook in WMD
class SpacyEmbeddings(object):
    def __init__(self, nlp):
        self.nlp = nlp
        
    def __getitem__(self, item):
        return self.nlp.vocab[item].vector


def docs_to_nbow_wmd(nlp, doc_list, add_quotations=True):
    documents = {}
    for i, doc in enumerate(doc_list):
        text = nlp(doc)
        tokens = [t for t in text if t.is_alpha and not t.is_stop]
        words = Counter(t.text for t in tokens)
        # orths = {t.text: t.orth for t in tokens}
        # This change 
        if add_quotations:
            orths = {t.text: nlp.vocab['"' + t.text + '"'].orth for t in tokens}
        else:
            orths = {t.text: nlp.vocab[t.text].orth for t in tokens}
        sorted_words = sorted(words)
        documents[i] = (i, [orths[t] for t in sorted_words],
                            np.array([words[t] for t in sorted_words],
                                        dtype=np.float32))
    return documents


def print_process():
    bash_cmd = 'ps -T -p %d' % (os.getpid())
    sprocess = subprocess.Popen(shlex.split(bash_cmd), stdout=subprocess.PIPE)
    output, error = sprocess.communicate()
    # print('CONSOLE OUTPUT: ', bash_cmd.split(), shlex.split(bash_cmd))
    print(output.decode('unicode_escape'))
    
    p = psutil.Process(os.getpid())
    print('Process num %d: number of threads: %d\n' % (os.getpid(), p.num_threads()))
    # print('Thread information: ', p.threads())

    
def get_nearest_neighbours(batch_start, batch_end,calc, num_neighbors, results):
    j = 0    
    print_after = math.ceil((batch_end - batch_start) / 2)
    
    print('####### Start of batch %d - %d #######' % (batch_start, batch_end))
    
    for i in range(batch_start, batch_end):#num_docs):
        class_votes = []
        neighbor_nums = []
        neighbor_wmds = []
        
        for doc_num, relevance in calc.nearest_neighbors(i, k=num_neighbors):
            if doc_num != i:
                class_votes.append(y[doc_num])
                neighbor_nums.append(doc_num)
                neighbor_wmds.append(relevance)
                
        top_class = Counter(class_votes).most_common(1)[0]
        predictions.append(top_class)
        votes_labels = [class_names[j] for j in class_votes]
        # print('Target: %r \t Prediction: %r \t Votes: %d/%d' % (class_names[y[i]], class_names[top_class[0]], top_class[1], (num_neighbors-1)))
        # print('Class votes: ', votes_labels)
        # print('\n')
        
        # ['doc_num', 'class_num', 'top_pred', 'all_preds', 'neighbor_nums', 'neighbor_wmds']
        all_preds_str = ','.join(str(p) for p in class_votes)
        neighbor_nums_str = ','.join(str(n) for n in neighbor_nums)
        neighbor_wmds_str = ','.join(str(d) for d in neighbor_wmds)
        temp = [i, y[i], top_class[0], all_preds_str, neighbor_nums_str, neighbor_wmds_str]
        # results.append(temp)
        results[i] = temp
        
        if j % print_after == 0:
            elapsed_time = time.time() - temp_time
            print('\tBatch %d-%d -- time after %d documents: \t %f\n' % (batch_start, batch_end, j, elapsed_time))
            # print_process()
            sys.stdout.flush()
        
        j += 1
    
    print('******* End of batch %d - %d *******' % (batch_start, batch_end))


def spacy_init_model(embs_dir, model_name, is_base=False):
    if not is_base:
        model_path = os.path.abspath('model/%s/%s.tsv' % (model_name, model_name))
    else:
        model_path = os.path.abspath('data/word_embeddings/%s.tsv' % (model_name))
    bash_cmd = 'python3 -m spacy init-model en %s --vectors-loc %s' % (embs_dir, model_path)
    
    print('Running command: \t', bash_cmd)
    
    sprocess = subprocess.Popen(shlex.split(bash_cmd), stdout=subprocess.PIPE)
    output, error = sprocess.communicate()
    # print('CONSOLE OUTPUT: ', bash_cmd.split(), shlex.split(bash_cmd))
    print(output.decode('unicode_escape'))
    

if __name__ == '__main__':
    print('Start time: ', time.asctime())
    t = time.time()
    
    # To run with profiler sorted by tottime
    # (tottime is the total runtime for a
    # single process)
    # python3 -m cProfile -s tottime wmd-10threads.py -o wmd-relax_spacy_LUKE.profile
    
    # model_name = 'word2vec-google-news-300_voc3'
    # model_name = 'rand_init-syns-10e-voc7-emb300'
    model_name = 'GoogleNews-vectors-negative300'
    wmd_dir = 'data/wmd/results/'
    is_base = True
    add_quotations = False
    
    # Initialising spaCy embeddings
    spacy_embs, nlp = init_spacy(model_name, is_base=is_base)
    
    cpu_count = multiprocessing.cpu_count()
    print('Available CPUs:', cpu_count)
    
    newsgroups = fetch_20newsgroups()#subset='test')
    docs, y, class_names = newsgroups.data, newsgroups.target, newsgroups.target_names

    # docs = docs[:100]
    num_docs = len(docs)
    print('Number of docs:', num_docs)
    
    nbow_docs = docs_to_nbow_wmd(nlp, docs, add_quotations=add_quotations)
    
    elapsed_time = time.time() - t
    print('\n\tTotal loading and preprocessing time: \t %f\n' % (elapsed_time))
    
    temp_time = time.time()
    
    print("Calculating...")
    # calc = wmd.WMD(SpacyEmbeddings(), nbow_docs, vocabulary_min=2)
    calc = wmd.WMD(spacy_embs, nbow_docs, vocabulary_min=2)
    
    elapsed_time = time.time() - temp_time
    print('\n\tTotal calculation time: \t %f\n' % (elapsed_time))

    ############## CACHING CENTROIDS
    temp_time = time.time()

    print('Caching centroids')
    calc.cache_centroids()
    # print('New centroids: ', calc._centroid_cache)

    elapsed_time = time.time() - temp_time
    print('\n\tTotal cache centroids time: \t %f\n' % (elapsed_time))

    sys.stdout.flush()
    
    ############## NEAREST NEIGHBORS
    temp_time = time.time()

    # After cache_centroids(), the nearest neighbors
    # function always includes the origin in the list
    # of neighbors, so to get a K of 10, num_neighbors
    # must be K+1=11
    num_neighbors = 11
    
    predictions = []
    # results = [['doc_num', 'class_num', 'top_pred', 'all_preds', 'neighbor_nums', 'neighbor_wmds']]
    
    # Sweet spot obtained from timing tests,
    # i.e. create three processes per available
    # core
    batch_size = math.ceil(num_docs / ((cpu_count*2)-1)) #40
    batches = {}
    results = {}
    jobs = []
    
    # Shared variable to communicate between
    # processes
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    
    for b in range(math.ceil(num_docs/batch_size)):#num_docs%batch_size):
        batch_start = b*batch_size
        batch_end = min(batch_start+batch_size, num_docs)
        print('Batch start: %d \t Batch end: %d' % (batch_start, batch_end))
        
        ######### MULTI-PROCESSING
        p = Process(target=get_nearest_neighbours, args=(batch_start, batch_end, calc, num_neighbors, return_dict,))
        p.start()
        jobs.append(p)
        
    # This part ensures that all threads finish
    # processing before going ahead
    for job in jobs:
        job.join()
    
    # print('Results dict: ', return_dict)
    
    results_file = os.path.abspath(wmd_dir + model_name + '_wmd-knn' + str(num_neighbors) + '-results.csv')
    print('Save path:', results_file)
    with open(results_file, 'w+') as f:
        writer = csv.writer(f)
        
        header = ['doc_num', 'class_num', 'top_pred', 'all_preds', 'neighbor_nums', 'neighbor_wmds']
        
        writer.writerow(header)
        
        for k, v in return_dict.items():
            # row = [k]
            # row.extend(v)
            # print('ROW: ',row)
            writer.writerow(v)#row)
    
    elapsed_time = time.time() - temp_time
    print('\n\tTotal nearest neighbors time: \t %f\n' % (elapsed_time))

    elapsed_time = time.time() - t
    print('\n\tElapsed time: \t\t %f' % (elapsed_time))

    print('End time: ', time.asctime())

    # print('\n\nvector for "that": ', SpacyEmbeddings()['the'])
    print('\n\nvector for "the": ', spacy_embs['"the"'])

