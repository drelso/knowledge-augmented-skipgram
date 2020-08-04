###
#
# KNN results
#
###

import csv
from collections import Counter
import numpy as np

def calculate_results(csv_file):
    with open(csv_file, 'r') as f:
        data = csv.reader(f)
        
        header = next(data)
        cols = {name: i for i, name in enumerate(header)}
        
        num_correct = 0
        total = 0
        
        targets = []
        preds = []
        
        for row in data:
            targets.append(row[cols['class_num']])
            preds.append(row[cols['top_pred']])
            
            if row[cols['class_num']] == row[cols['top_pred']]:
                num_correct += 1
        
            total += 1
    
    conf_interval = normal_approx_interval(num_correct, total)
    
    print('Accuracy: %d/%d \t %.4f (+/- %.4f)' % (num_correct, total, (num_correct/total), conf_interval))
    # print('Target counts:', Counter(targets).most_common())
    print('Preds counts:', Counter(preds).most_common())
    

def normal_approx_interval(num_correct, num_total, z=1.96):#confidence_level=0.95):
    # From https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Wilson_score_interval
    # Values for z:
    # 1.64 (90%)
    # 1.96 (95%)
    # 2.33 (98%)
    # 2.58 (99%)
    # alpha = 1 - confidence_level
    confidence_interval = (z/num_total) * np.sqrt((num_correct * (num_total - num_correct)) / num_total)
    return confidence_interval


if __name__ == '__main__':
    csv_file = 'data/wmd/results/word2vec-google-news-300_voc20_wmd-knn11-results.csv'
    print('\nCalculating file: ', csv_file)
    calculate_results(csv_file)
    
    csv_file = 'data/wmd/results/w2v_init-syns-10e-voc20-emb300_wmd-knn11-results.csv'
    print('\nCalculating file: ', csv_file)
    calculate_results(csv_file)
    
    csv_file = 'data/wmd/results/w2v_init-no_syns-10e-voc20-emb300_wmd-knn11-results.csv'
    print('\nCalculating file: ', csv_file)
    
    calculate_results(csv_file)
    csv_file = 'data/wmd/results/rand_init-syns-10e-voc20-emb300_wmd-knn11-results.csv'
    print('\nCalculating file: ', csv_file)
    calculate_results(csv_file)
    
    csv_file = 'data/wmd/results/rand_init-no_syns-10e-voc20-emb300_wmd-knn11-results.csv'
    print('\nCalculating file: ', csv_file)
    calculate_results(csv_file)
    
    csv_file = 'data/wmd/results/w2v_init-syns-10e-voc7-emb300_wmd-knn11-results.csv'
    print('\nCalculating file: ', csv_file)
    calculate_results(csv_file)
    
    csv_file = 'data/wmd/results/w2v_init-no_syns-10e-voc7-emb300_wmd-knn11-results.csv'
    print('\nCalculating file: ', csv_file)
    calculate_results(csv_file)
    
    csv_file = 'data/wmd/results/word2vec-google-news-300_voc7_wmd-knn11-results.csv'
    print('\nCalculating file: ', csv_file)
    calculate_results(csv_file)
    
    csv_file = 'data/wmd/results/w2v_init-10e-voc3-emb300_wmd-knn11-results.csv'
    print('\nCalculating file: ', csv_file)
    calculate_results(csv_file)
    
    csv_file = 'data/wmd/results/w2v_init-nosyns-10e-voc3-emb300_wmd-knn11-results.csv'
    print('\nCalculating file: ', csv_file)
    calculate_results(csv_file)
    
    csv_file = 'data/wmd/results/word2vec-google-news-300_voc3_wmd-knn11-results.csv'
    print('\nCalculating file: ', csv_file)
    calculate_results(csv_file)
    
    csv_file = 'data/wmd/results/rand_init-no_syns-10e-voc3-emb300_wmd-knn11-results.csv'
    print('\nCalculating file: ', csv_file)
    calculate_results(csv_file)
    
    csv_file = 'data/wmd/results/rand_init-10e-voc3-emb300_wmd-knn11-results.csv'
    print('\nCalculating file: ', csv_file)
    calculate_results(csv_file)
    
    csv_file = 'data/wmd/results/rand_init-no_syns-10e-voc7-emb300_wmd-knn11-results.csv'
    print('\nCalculating file: ', csv_file)
    calculate_results(csv_file)
    
    csv_file = 'data/wmd/results/rand_init-syns-10e-voc7-emb300_wmd-knn11-results.csv'
    print('\nCalculating file: ', csv_file)
    calculate_results(csv_file)
    
    csv_file = 'data/wmd/results/GoogleNews-vectors-negative300_wmd-knn11-results.csv'
    print('\nCalculating file: ', csv_file)
    calculate_results(csv_file)