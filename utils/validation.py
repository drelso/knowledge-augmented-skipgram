###
#
# Validation functions
#
###

import re
import csv
import numpy as np
import scipy.spatial
from  scipy.stats import pearsonr
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

from .model_paths import get_model_path, get_correl_dict, get_base_embeds, get_correl_datasets, get_vocab_num, get_embeds_for_vocab, correl_paths_for_vocab


def correl_all_data_min_vocab(min_vocab_num=20, distance='cos', vocab_sizes=[3,7,20], ratio_models=False):#dataset_path, save_file, vocab_path, embs_list, incl_header=True, score_type='similarity', data_has_header=False, score_index=2):
    """
    Read a similarity dataset and keep
    only the examples that appear in the
    vocabulary.
    
    Requirements
    ------------
    import re
    import csv
    import numpy as np
    import scipy.spatial
    
    Parameters
    ----------
    dataset_path : str
        path to the similarity data to process
    save_file : str
        path to save the correlation data to
    vocab_path : str
        path to the vocabulary file
    embs_list : list[[str,str]]
        list of tuples where the first element
        is the name of the embeddings and the
        second is the path to the word embeddings
        (requires NPY format)
    incl_header : bool, optional
        whether to include a header in
        the file or not (default: True)
    
    NOTE: Make sure all the embeddings on the
          list can fit in memory at the same time
    """
    
    correl_datasets = get_correl_datasets()
    
    correl_results_dir = 'data/validate/correlation/'
    
    min_vocab_path = get_vocab_num(min_vocab_num)
    print('Minimum vocabulary path:', min_vocab_path)
    with open(min_vocab_path, 'r') as v:
        min_vocab = [row[0] for row in csv.reader(v)]
    
    # embs_dict = get_model_path('embeds', ratio_models)
    # print(embs_dict)
    
    base_embs = get_base_embeds()
    print('\nBase embeds: ', base_embs)
    
    print('Vocab length:', len(min_vocab))
    # print('Vocab first 10: ', min_vocab[:10])
    
    for vocab_size in vocab_sizes:
        print('Vocabulary %d' % (vocab_size))
        vocab_path = get_vocab_num(vocab_size)
        with open(vocab_path, 'r') as v:
            # vocab = [row[0] for row in csv.reader(v)]
            vocab = {w[0]: i for i, w in enumerate(csv.reader(v))}
            
        embs_dict = get_embeds_for_vocab(vocab_size, ratio_models)
        print('Models for vocab: ', embs_dict)
        vocab_re = '_v' + str(vocab_size)
        emb_names = [name for name in embs_dict.keys()]
        emb_col_names = [re.sub(vocab_re, '', name) for name in emb_names]
        col_names = ['word_1', 'word_2', 'score'] + emb_col_names
        
        embs = {name: np.load(path) for name, path in embs_dict.items()}
        for correl_name, correl_data_path in correl_datasets.items():
            print('%s correl dataset path: \t %s' % (correl_name, correl_data_path))
            
            print('Col names: ', col_names)
            if not ratio_models:
                save_file = correl_results_dir + correl_name + '_v' + str(vocab_size) + '.csv'
            else:
                save_file = correl_results_dir + correl_name + '_ratios_v' + str(vocab_size) + '.csv'
            
            with open(correl_data_path, 'r') as f, \
                open(save_file, 'w+') as s:
                data = csv.reader(f, delimiter='\t')
                writer = csv.writer(s)
                if re.search('SimLex', correl_name):
                    # SimLex dataset has a header
                    header = next(data)
                    score_index = header.index('SimLex999')
                else:
                    # WordSim353 datasets scores
                    # appear in the third column
                    score_index = 2
                valid_pairs = 0
                total_pairs = 0
                results = [col_names]
                
                for row in data:
                    word_1 = row[0].lower()
                    word_2 = row[1].lower()
                    score = row[score_index]
                    
                    
                    if word_1 in min_vocab and word_2 in min_vocab:
                        temp_row = [None] * len(col_names)
                        temp_row[col_names.index('word_1')] = word_1
                        temp_row[col_names.index('word_2')] = word_2
                        temp_row[col_names.index('score')] = score
                        valid_pairs += 1
                        
                        for emb_name in emb_names:
                            emb_1 = embs[emb_name][vocab[word_1]]
                            emb_2 = embs[emb_name][vocab[word_2]]
                            if distance == 'cos':
                                dist = scipy.spatial.distance.cosine(emb_1, emb_2)
                            elif distance == 'euc':
                                dist = scipy.spatial.distance.cosine(emb_1, emb_2)
                            else:
                                raise ValueError('Invalid value for distance metric: %s' % (distance))
                            emb_col = re.sub(vocab_re, '', emb_name)
                            temp_row[col_names.index(emb_col)] = dist
                        
                        results.append(temp_row)
                    total_pairs += 1
                
                print('Valid word pairs in %s: %d/%d' % (correl_name, valid_pairs, total_pairs))
                print('Writing to ', save_file)
                writer.writerows(results)


def correlation_data(dataset_path, save_file, vocab_path, embs_list, incl_header=True, score_type='similarity', data_has_header=False, score_index=2):
    """
    Read a similarity dataset and keep
    only the examples that appear in the
    vocabulary.
    
    Requirements
    ------------
    import csv
    import numpy as np
    import scipy.spatial
    
    Parameters
    ----------
    dataset_path : str
        path to the similarity data to process
    save_file : str
        path to save the correlation data to
    vocab_path : str
        path to the vocabulary file
    embs_list : list[[str,str]]
        list of tuples where the first element
        is the name of the embeddings and the
        second is the path to the word embeddings
        (requires NPY format)
    incl_header : bool, optional
        whether to include a header in
        the file or not (default: True)
    
    NOTE: Make sure all the embeddings on the
          list can fit in memory at the same time
    """
    
    with open(dataset_path, 'r') as f, \
        open(vocab_path, 'r') as v, \
        open(save_file, 'w+') as w:
        data = csv.reader(f, delimiter='\t')
        vocab = csv.reader(v)
        writer = csv.writer(w)
        
        # NOTE: requires ~120MB in RAM per
        # embedding matrix
        embs = {name: np.load(path) for name, path in embs_list}
        
        # w2v_embs = np.load(w2v_embs)
        # w2v_syns_embs = np.load(w2v_syns_embs)
        # rand_embs = np.load(rand_embs)
        
        validation_set = []
        
        if data_has_header:
            header = next(data)
            score_index = header.index(score_type)
        
        if incl_header:
            # col_names = ['word_1', 'word_2', score_type, 'cos_w2v', 'euc_w2v', 'cos_w2v_syns', 'euc_w2v_syns', 'cos_rand', 'euc_rand']
            col_names = ['word_1', 'word_2', score_type]
            
            for emb_name in embs.keys():
                col_names.extend([emb_name + '_cos', emb_name + '_euc'])
            
            validation_set.append(col_names)
        
        voc = {w[0]: i for i, w in enumerate(vocab)}
        
        i = 0
        
        for row in data:
            word_1 = row[0].lower()
            word_2 = row[1].lower()
            
            dists_row = [word_1, word_2, row[score_index]]
            
            if word_1 in voc.keys() and word_2 in voc.keys():
                for emb_name in embs.keys():
                    emb_1 = embs[emb_name][voc[word_1]]
                    emb_2 = embs[emb_name][voc[word_2]]
                
                    cos_dist = scipy.spatial.distance.cosine(emb_1, emb_2)
                    euc_dist = scipy.spatial.distance.euclidean(emb_1, emb_2)
                    
                    dists_row.extend([cos_dist, euc_dist])
                # dists_row = [word_1, word_2, row[score_index], w2v_cos, w2v_euc, w2v_syns_cos, w2v_syns_euc, rand_cos, rand_euc]
                
                if len(col_names) == len(dists_row):
                    validation_set.append(dists_row)
                else:
                    raise ValueError('Size mismatch: attempting to add a list of values that does not match the number of columns: %r and values %r' % (col_names, dists_row))
                
                i+=1
        
        print('Saving %d rows to %r' % (i, save_file))
        # np.save(save_file, validation_set)
        writer.writerows(validation_set)


def calc_correl_data(embs_file, correl_file, save_file, incl_header=True, data_has_header=False, score_index=2):
    """
    Read a similarity dataset and keep
    only the examples that appear in the
    vocabulary.
    
    Requirements
    ------------
    import csv
    import numpy as np
    import scipy.spatial
    
    from .model_paths import get_model_path,
                            get_correl_dict,
                            get_base_embeds
    
    Parameters
    ----------
    embs_file : str
        path to the embeddings file to process
        (requires NPY format)
    correl_dataset_dict : str
        dictionary of similarity data paths to
        process, keys are similarity names and 
    save_file : str
        path to save the correlation data to
    vocab_path : str
        path to the vocabulary file
    embs_dict : dict{str: str}
        dictionary where keys are the names of
        the embeddings and values are the paths
        to the word embedding files (requires NPY
        format)
    incl_header : bool, optional
        whether to include a header in
        the file or not (default: True)
    
    NOTE: Make sure all the embeddings on the
          list can fit in memory at the same time
    """
    
    print(embs_file)
    
    print('Embs file:', embs_file)
    print('Correl file:', correl_file)
    print('Save file:', save_file)
    
    # extension = '_correl_' + score_type
    
    '''
    with open(dataset_path, 'r') as f, \
        open(vocab_path, 'r') as v, \
        open(save_file, 'w+') as w:
        data = csv.reader(f, delimiter='\t')
        vocab = csv.reader(v)
        writer = csv.writer(w)
        
        # NOTE: requires ~120MB in RAM per
        # embedding matrix
        embs = {name: np.load(path) for name, path in embs_list}
        
        # w2v_embs = np.load(w2v_embs)
        # w2v_syns_embs = np.load(w2v_syns_embs)
        # rand_embs = np.load(rand_embs)
        
        validation_set = []
        
        if data_has_header:
            header = next(data)
            score_index = header.index(score_type)
        
        if incl_header:
            # col_names = ['word_1', 'word_2', score_type, 'cos_w2v', 'euc_w2v', 'cos_w2v_syns', 'euc_w2v_syns', 'cos_rand', 'euc_rand']
            col_names = ['word_1', 'word_2', score_type]
            
            for emb_name in embs.keys():
                col_names.extend([emb_name + '_cos', emb_name + '_euc'])
            
            validation_set.append(col_names)
        
        voc = {w[0]: i for i, w in enumerate(vocab)}
        
        i = 0
        
        for row in data:
            word_1 = row[0].lower()
            word_2 = row[1].lower()
            
            dists_row = [word_1, word_2, row[score_index]]
            
            if word_1 in voc.keys() and word_2 in voc.keys():
                for emb_name in embs.keys():
                    emb_1 = embs[emb_name][voc[word_1]]
                    emb_2 = embs[emb_name][voc[word_2]]
                
                    cos_dist = scipy.spatial.distance.cosine(emb_1, emb_2)
                    euc_dist = scipy.spatial.distance.euclidean(emb_1, emb_2)
                    
                    dists_row.extend([cos_dist, euc_dist])
                # dists_row = [word_1, word_2, row[score_index], w2v_cos, w2v_euc, w2v_syns_cos, w2v_syns_euc, rand_cos, rand_euc]
                
                if len(col_names) == len(dists_row):
                    validation_set.append(dists_row)
                else:
                    raise ValueError('Size mismatch: attempting to add a list of values that does not match the number of columns: %r and values %r' % (col_names, dists_row))
                
                i+=1
        
        print('Saving %d rows to %r' % (i, save_file))
        # np.save(save_file, validation_set)
        writer.writerows(validation_set)
    '''
    
def correl_coefficients_for_vocab(vocab_num, correl_metric='spearmanr', ratio_models=False):
    """
    Create a plot from a correlation
    data file in the format of
    correlation_data()
    
    Requirements
    ------------
    import csv
    scipy.stats.pearsonr
    scipy.stats.spearmanr
    correl_paths_for_vocab
    
    Parameters
    ----------
    data_file : str
        path to the correlation data
        file with the following columns:
        - word_1
        - word_2
        - relatedness / similarity
        - cos_w2v
        - euc_w2v
        - cos_w2v_syns
        - euc_w2v_syns
        - cos_rand
        - euc_rand
    score_type : str, optional
        type of score being ploted. Options are:
        - similarity
        - relatedness
        - SimLex999
    """
    correl_result_files = correl_paths_for_vocab(vocab_num, ratio_models)
    print('Correl result files: ', correl_result_files)
    
    full_correl_results = {}
    
    for correl_name, correl_file in correl_result_files.items():
        print('Running %s for vocabulary %d' % (correl_name, vocab_num))
        with open(correl_file, 'r') as f:
            data = csv.reader(f)
            header = next(data)
            
            # Creates a dictionary of column names
            cols = {w: i for i, w in enumerate(header)}
            correls_dict = {name: [] for name in header if name != 'word_1' and name != 'word_2'}
            # print('Columns: ', cols)
            
            for row in data:
                for key in correls_dict.keys():
                    correls_dict[key].append(float(row[cols[key]]))
            
        results = {}
        for model_name in correls_dict.keys():
            if model_name != 'score':
                if correl_metric == 'spearmanr':
                    correl_score = scipy.stats.spearmanr(correls_dict['score'], correls_dict[model_name])
                elif correl_score == 'pearsonr':
                    correl_score = scipy.stats.pearsonr(correls_dict['score'], correls_dict[model_name])
                else:
                    raise ValueError('Correlation metric "%s" not defined.' % (correl_metric))
                results[model_name] = correl_score[0]
                # print('%s for %s: \t %r' % (correl_metric, model_name, correl_score[0]))
        
        full_correl_results[correl_name] = results
    
    return full_correl_results


def correlation_coefficients(data_file, score_type='similarity'):
    """
    Create a plot from a correlation
    data file in the format of
    correlation_data()
    
    Requirements
    ------------
    import csv
    scipy.stats.pearsonr
    scipy.stats.spearmanr
    
    Parameters
    ----------
    data_file : str
        path to the correlation data
        file with the following columns:
        - word_1
        - word_2
        - relatedness / similarity
        - cos_w2v
        - euc_w2v
        - cos_w2v_syns
        - euc_w2v_syns
        - cos_rand
        - euc_rand
    score_type : str, optional
        type of score being ploted. Options are:
        - similarity
        - relatedness
        - SimLex999
    """
    with open(data_file, 'r') as f:
        data = csv.reader(f)
        header = next(data)
        
        # Creates a dictionary of column names
        cols = {w: i for i, w in enumerate(header)}
        
        scores = []
        w2v_cos = []
        w2v_euc = []
        w2v_syns_cos = []
        w2v_syns_euc = []
        rand_cos = []
        rand_euc = []
        
        for row in data:
            scores.append(float(row[cols[score_type]]))
            w2v_cos.append(float(row[cols['cos_w2v']]))
            w2v_euc.append(float(row[cols['euc_w2v']]))
            w2v_syns_cos.append(float(row[cols['cos_w2v_syns']]))
            w2v_syns_euc.append(float(row[cols['euc_w2v_syns']]))
            rand_cos.append(float(row[cols['cos_rand']]))
            rand_euc.append(float(row[cols['euc_rand']]))

    w2v_cos_correls = [scipy.stats.pearsonr(scores, w2v_cos), scipy.stats.spearmanr(scores, w2v_cos)]
    w2v_euc_correls = [scipy.stats.pearsonr(scores, w2v_euc), scipy.stats.spearmanr(scores, w2v_euc)]
    w2v_syns_cos_correls = [scipy.stats.pearsonr(scores, w2v_syns_cos), scipy.stats.spearmanr(scores, w2v_syns_cos)]
    w2v_syns_euc_correls = [scipy.stats.pearsonr(scores, w2v_syns_euc), scipy.stats.spearmanr(scores, w2v_syns_euc)]
    rand_cos_correls = [scipy.stats.pearsonr(scores, rand_cos),scipy.stats.spearmanr(scores, rand_cos)] 
    rand_euc_correls = [scipy.stats.pearsonr(scores, rand_euc), scipy.stats.spearmanr(scores, rand_euc)]
    
    return w2v_cos_correls, w2v_euc_correls, w2v_syns_cos_correls, w2v_syns_euc_correls, rand_cos_correls, rand_euc_correls


def embedding_distance_table(embs_path, save_file, similarity='cosine'):
    """
    Calculate the full pairwise distances
    between embeddings. This is an expensive
    operation, Nx(N-1)/2 cycles and an NxN
    matrix
    
    Requirements
    ------------
    import numpy as np
    
    Parameters
    ----------
    embs_path : str
        path to the embeddings file, which
        is an NPY file where every row is an
        embedding (NxD matrix)
    save_file : str
        path to the file to save the table to
    similarity : str
        type of similarity to use use. Options:
        - 'cosine'
        - 'euclidean'
    
    """
    embs = np.load(embs_path)
    
    N, dims = embs.shape
    
    print(N, dims)
    print(type(embs))
    
    dist_matrix = np.empty([N, N])
    # i = 0
    
    for i, vec in enumerate(embs):
        for j in range(i, N):
            if i == j:
                dist_matrix[i,j] = 0.
            else:
                dist = scipy.spatial.distance.cosine(vec, embs[j])
                dist_matrix[i,j] = dist
                dist_matrix[j,i] = dist
        
        if i % 1000 == 0:
            print('%d vectors processed')
        # i += 1
        # if i > 2: break
        
    np.save(save_file, dist_matrix)
    print('Distance matrix (%dx%d) saved to file %r' % (N, N, save_file))


def get_syn_pairs(data_file, save_file, randomly_sampled=0, max_datapoints=100000, target_col='synonym'):
    """
    Get a set of synonym pairs together with
    their counts and frequencies. The data file
    is expected in the following format:
    - synonym
    - context_word
    - sent_num
    - focus_index
    - context_position
    - focus_word
    - book_number
    
    Requirements
    ------------
    import csv
    import numpy as np
    
    Parameters
    ----------
    data_file : str
        path to the data file
    save_file : str
        path to the file to save the synonyms to
    randomly_sampled : int, optional
        if non-zero, number of samples to randomly
        select from the dataset (default: 0)
    max_datapoints : int, optional
        if randomly sampling datapoints, this
        value defines the range to sample from,
        if this is lower than the number of
        datapoints the last points will not be
        sampled, if it is larger than the number
        of datapoints the returned samples will
        be fewer than the value specified in
        randomly_sampled (default: 100000)
    target_col : str, optional
        name of the target word column
        (default: 'synonym')
    """
    
    keep_samples = np.array([])
    max_index = 0
    
    if int(randomly_sampled) > 0:
        keep_samples = np.random.choice(max_datapoints, randomly_sampled, replace=False)
        max_index = max(keep_samples)
        print('Keep samples: ', keep_samples)
        
    with open(data_file, 'r') as f, \
        open(save_file, 'w+') as s:
        data = csv.reader(f)
        writer = csv.writer(s)
        
        header = next(data)
        cols = {w:i for i,w in enumerate(header)}
        
        # i = 0
        
        syns = []
        syn_counts = []
        count_index = 1
        
        # syn_counts.append(['focus_word', 'synonym', 'counts'])
        
        for i, row in enumerate(data):
            if not keep_samples.any() or i in keep_samples:
                syn_pair = [row[cols['focus_word']], row[cols[target_col]]]
                if syn_pair in syns:
                    # Add 1 to account for the header row
                    # syn_index = syns.index(syn_pair) + 1
                    syn_index = syns.index(syn_pair)
                    count = syn_counts[syn_index][2]
                    syn_counts[syn_index][2] = int(count) + 1
                else:
                    syns.append(syn_pair)
                    syn_counts.append([syn_pair[0], syn_pair[1], 1])
                # i+=1
                # if i > 20: break
            if i % 100000 == 0:
                print('%d word pairs processed' % (i))
            
            if max_index > 0 and i > max_index: break
            
        syn_freqs = [[s[0], s[1], s[2], s[2]/i] for s in syn_counts]
        
        print('Processed %d word pairs (%d unique)' % (i, len(syn_freqs)))
        print('Saving file to ', save_file)
        
        writer.writerow(['focus_word', target_col, 'counts', 'freqs'])
        writer.writerows(syn_freqs)


def word_pair_distances(word_pair_file, vocab_file, embs_list, save_file, focus_col='focus_word', target_col='synonym'):
    """
    Given a file of word pairs, calculate the
    distances between their different embeddings
    
    Requirements
    ------------
    import csv
    import numpy as np
    import scipy.spatial
    
    Parameters
    ----------
    word_pair_file : str
        path to the file containing the
        (unique) word pairs. This file should
        have (at least) the following named
        columns:
        - focus_col
        - target_col
    vocab_file : str
        path to the vocabulary file
    embs_list : list[[str,str]]
        list of tuples where the first element
        is the name of the embeddings and the
        second is the path to the word embeddings
        (requires NPY format)
    save_file : str
        path to the file to save the calculated
        distances to
    focus_col : str, optional
        name of the column containing the
        'focus' word (default: 'focus_word')
    target_col : str, optional
        name of the column containing the
        'target' word (default: 'synonym')
    
    NOTE: if there is a zero vector we ignore
    the word-synonym distance row. This can be
    changed later
    """
    with open(word_pair_file, 'r') as f, \
        open(vocab_file, 'r') as v, \
        open(save_file, 'w+') as s:
        
        data = csv.reader(f)
        header = next(data)
        vocabulary = {w[0]: i for i,w in enumerate(csv.reader(v))}
        writer = csv.writer(s)
        
        try:
            focus_index = header.index(focus_col)
        except:
            raise ValueError('Focus column %r not in header: %r' % (focus_col, header)) from None
        
        try:
            target_index = header.index(target_col)
        except:
            raise ValueError('Target column %r not in header: %r' % (target_col, header)) from None
        
        embs = {name: np.load(path) for name, path in embs_list}
        emb_names = list(embs.keys())
        cols = [focus_col, target_col]
        
        for name in emb_names:
            cols.append(name + '_cos')
            cols.append(name + '_euc')
        
        print(cols)
        dist_matrix = [cols]
        
        missing_focus = 0
        missing_targets = 0
        zero_vecs = 0
        
        i = 0
        
        for row in data:
            try: focus = vocabulary[row[focus_index]]
            except:
                missing_focus += 1
                continue
            try: target = vocabulary[row[target_index]]
            except:
                missing_targets += 1
                continue
            
            dists = [row[focus_index], row[target_index]]
            is_zero = False
            
            for name in embs.keys():
                emb_1 = embs[name][focus]
                emb_2 = embs[name][target]
                cos_dist = scipy.spatial.distance.cosine(emb_1, emb_2)
                euc_dist = scipy.spatial.distance.euclidean(emb_1, emb_2)        
                
                dists.append(cos_dist)
                dists.append(euc_dist)
            
                if np.isnan(cos_dist):
                    zero_vecs += 1
                    is_zero = True
            
            if not is_zero: dist_matrix.append(dists)
            
            i += 1
        
        print('Total words: %d \t Missing words: focus=%d \t synonyms=%d \t zero vecs=%d' % (i, missing_focus, missing_targets, zero_vecs))
        
        print('Saving distances to %r' % (save_file))
        writer.writerows(dist_matrix)


def word_pair_dists_dict(word_pair_file, vocab_file, embs_dict, save_file, focus_col='focus_word', target_col='synonym'):
    """
    Given a file of word pairs, calculate the
    distances between their different embeddings
    
    Requirements
    ------------
    import csv
    import numpy as np
    import scipy.spatial
    
    Parameters
    ----------
    word_pair_file : str
        path to the file containing the
        (unique) word pairs. This file should
        have (at least) the following named
        columns:
        - focus_col
        - target_col
    vocab_file : str
        path to the vocabulary file
    embs_list : dict{str: str}
        dictionary of embedding paths where the
        key is the name of the embeddings and the
        value is the path to the word embeddings
        (requires NPY format)
    save_file : str
        path to the file to save the calculated
        distances to
    focus_col : str, optional
        name of the column containing the
        'focus' word (default: 'focus_word')
    target_col : str, optional
        name of the column containing the
        'target' word (default: 'synonym')
    
    NOTE: if there is a zero vector we ignore
    the word-synonym distance row. This can be
    changed later
    """
    with open(word_pair_file, 'r') as f, \
        open(vocab_file, 'r') as v, \
        open(save_file, 'w+') as s:
        
        data = csv.reader(f)
        header = next(data)
        vocabulary = {w[0]: i for i,w in enumerate(csv.reader(v))}
        writer = csv.writer(s)
        
        try:
            focus_index = header.index(focus_col)
        except:
            raise ValueError('Focus column %r not in header: %r' % (focus_col, header)) from None
        
        try:
            target_index = header.index(target_col)
        except:
            raise ValueError('Target column %r not in header: %r' % (target_col, header)) from None
        
        embs = {name: np.load(path) for name, path in embs_dict.items()}
        emb_names = list(embs.keys())
        cols = [focus_col, target_col]
        
        for name in emb_names:
            cols.append(name + '_cos')
            cols.append(name + '_euc')
        
        print(cols)
        dist_matrix = [cols]
        
        missing_focus = 0
        missing_targets = 0
        zero_vecs = 0
        
        i = 0
        
        for row in data:
            try: focus = vocabulary[row[focus_index]]
            except:
                missing_focus += 1
                continue
            try: target = vocabulary[row[target_index]]
            except:
                missing_targets += 1
                continue
            
            dists = [row[focus_index], row[target_index]]
            is_zero = False
            
            for name in embs.keys():
                emb_1 = embs[name][focus]
                emb_2 = embs[name][target]
                cos_dist = scipy.spatial.distance.cosine(emb_1, emb_2)
                euc_dist = scipy.spatial.distance.euclidean(emb_1, emb_2)        
                
                dists.append(cos_dist)
                dists.append(euc_dist)
            
                if np.isnan(cos_dist):
                    zero_vecs += 1
                    is_zero = True
            
            if not is_zero: dist_matrix.append(dists)
            
            i += 1
        
        print('Total words: %d \t Missing words: focus=%d \t synonyms=%d \t zero vecs=%d' % (i, missing_focus, missing_targets, zero_vecs))
        
        print('Saving distances to %r' % (save_file))
        writer.writerows(dist_matrix)
        
    
def calc_dist_changes(dist_file, emb_source, emb_target, title='', focus_col='focus_word', target_col='synonym'):
    """
    Given a file of distances, calculate
    the changes in distance between source
    and target embeddings in the following
    way:
    
        dist_change = source_dist - target_dist
    
    Such that a positive value implies that
    the source distance is larger than the
    target distance.
    
    Requirements
    ------------
    import csv
    import numpy as np
    
    Parameters
    ----------
    dist_file : str
        path to the file containing the
        distances. The file must be a CSV
        file with, at least, the following
        columns:
        - focus_col
        - target_col
        - @emb_source
        - @emb_target
    emb_source : str
        name of the source embedding distances
        column in the distances file
    emb_target : str
        name of the target embedding distances
        column in the distances file
    title : str, optional
        title to print to console
    focus_col : str, optional
        name of focus word column
        (default: 'focus_word')
    target_col : str, optional
        name of target word column
        (default: 'synonym')
    
    Returns
    -------
    list[float]
        list of distances sorted in ascending
        order
    """
    with open(dist_file, 'r') as f:
        data = csv.reader(f)
        header = next(data)
        
        # Creates a dictionary of column names
        cols = {w: i for i, w in enumerate(header)}
        
        dists = []
        dist_changes = []
        
        for row in data:
            word_pair = row[cols[focus_col]] + '-' + row[cols[target_col]]
            dist_change = float(row[cols[emb_source]]) - float(row[cols[emb_target]])
            
            dists.append([word_pair, dist_change])
            dist_changes.append(dist_change)
            
        sorted_dists = sorted(dists, key=lambda x: x[1])
        
        total_change = np.sum(dist_changes)
        average_change = total_change / len(dist_changes)
        
        print('Distance changes ', title)
        print('Total change: ', total_change)
        print('Average change: ', average_change)
        
        return sorted_dists


def rand_word_pairs(vocab_path, save_file, num_word_pairs=100, incl_header=True, focus_col='focus_word', target_col='rand_word'):
    """
    Construct a set of randomly generated
    word pairs from a vocabulary
    
    Requirements
    ------------
    import csv
    import numpy as np
    
    Parameters
    ----------
    vocab_path : str
        path to the vocabulary file, which
        is assumed to be a CSV file with a
        header where the first column corresponds
        to the words
    save_file : str
        file to save the random word pairs to
    num_word_pairs : int, optional
        number of word pairs to generate
        (default: 100)
    """
    
    with open(vocab_path, 'r') as v, \
        open(save_file, 'w+') as s:
        
        voc_data = csv.reader(v)
        writer = csv.writer(s)
        
        if incl_header: header = next(voc_data)
        vocab = {i: w[0] for i, w in enumerate(voc_data)}
        
        # num_word_pairs x 2 matrix of random indeces
        rand_ixs = np.random.choice(len(vocab), [num_word_pairs, 2])
        
        word_pairs = [[focus_col, target_col]]
        
        for w1, w2 in rand_ixs:
            word_pairs.append([vocab[w1],vocab[w2]])
        
        print('Saving %d randomly generated word pairs to %r' % (len(word_pairs), save_file))
        writer.writerows(word_pairs)