###
#
# Dataset processing and sampling utilities
#
###

import time
import datetime
import random
import csv
import re
from collections import Counter

import contextlib

import numpy as np
from numpy.core.numeric import False_

import spacy
from nltk.corpus import wordnet as wn
import gensim.downloader

import torch
import torchtext

from .funcs import sample_files, process_gutenberg_data, lightweight_dataset


def shuffle_and_subset_dataset(data_path, tags_path, subset_data_path, subset_tags_path, data_size=0.5):
    '''
    Shuffle the dataset and save a subset to file
    while keeping the POS tag alignment.
    This function is written to work with a pair of
    files where the first is raw text files with sentences
    in each line and the corresponding POS tags (space separated)
    on the same line in the second file

    NOTE: high memory usage, loads the two full datasets into memory

    Requirements
    ------------
    import numpy as np

    Parameters
    ----------
    data_path : str
        path to the file containing the raw text sentences
    tags_path : str
        path to the file containing the POS tags
    subset_data_path : str
        path to the file containing to save the raw text
        sentences to
    subset_tags_path : str
        path to the file containing to save the POS tags to
    data_size : float, optional
        percentage of the original dataset to keep in the 
        subset, a value of 1.0 saves a shuffled version of the
        full dataset (default: 0.5)
    '''
    print(f'Shuffling and subsetting {data_size * 100}% of text data at {data_path} and POS tags at {tags_path}')
    print(f'Saving datsets at {subset_data_path} and POS tags at {subset_tags_path}')

    with open(data_path, 'r', encoding='utf-8') as d, \
        open(tags_path, 'r') as td, \
        open(subset_data_path, 'w+') as sd, \
        open(subset_tags_path, 'w+') as std:

        text_data = d.readlines()
        tags_data = td.readlines()

        text_size = len(text_data)
        tags_size = len(tags_data)

        if text_size != tags_size:
            raise ValueError(f'Text file size ({text_size}) and POS tags file size ({tags_size}) must be the same')

        print(f'Lines in text data: {len(text_data)} \t Lines in tag data: {len(tags_data)}')
        num_datapoints = int(len(text_data) * data_size)

        print(f'{num_datapoints} in subset dataset')
        datapoint_ixs = np.random.choice(text_size, num_datapoints, replace=False)

        random_ix = datapoint_ixs[np.random.randint(num_datapoints)]
        
        print('Writing shuffled text data')
        subset_data = ''
        for i in datapoint_ixs:
            subset_data += text_data[i]
        sd.write(subset_data)
        verification_text = text_data[random_ix]
        del subset_data
        del text_data

        print('Writing shuffled POS tag data')
        subset_tags = ''
        for i in datapoint_ixs:
            subset_tags += tags_data[i]
        std.write(subset_tags)
        verification_tags = tags_data[random_ix]

        verif_txt_size = len(verification_text.split(' '))
        verif_tags_size = len(verification_tags.split(' '))
        print(f'Verification datapoint at line {random_ix}: \n words in text: {verif_txt_size} \t tags in POS tag data: {verif_tags_size}')
        print(f'Text at line {random_ix}: \n {verification_text}')
        print(f'POS tags at line {random_ix}: \n {verification_tags}')


def process_all_datafiles(data_dir, dataset_savefile, tags_savefile=None, use_headwords=False, replace_nums=False, replace_unclass=False):
    """
    Given a data directory (for BNC XML files)
    go through all files in the directory, process
    them, and append them to the processed data
    save file
    
    Parameters
    ----------
    data_dir : str
        directory containing all subdirectories
        and BNC XML files
    dataset_savefile : str
        filepath to the processed data file to
        save to
    use_headwords : bool, optional
        whether to use headwords or the raw
        version of the words (default: False)
    replace_nums : bool, optional
        whether to replace numbers with a default
        tag <NUM> (default: False)
    replace_unclass : bool, optional
        whether to replace "unclassified" (i.e.) with a
        default tag <UNC> (default: False)
    """
    for path in os.listdir(data_dir):
        full_path = data_dir + path
        if os.path.isdir(full_path):
            print('Opening ', full_path, 'directory')
            items = os.listdir(full_path)
            print(f'Directory contains {len(items)} files')
            for item in items:
                dir_path = full_path + '/' + item
                if item.endswith('.xml'):
                    print('Processing file', dir_path)
                    construct_dataset(dir_path, dataset_savefile, tags_savefile=tags_savefile, use_headwords=use_headwords, replace_nums=replace_nums, replace_unclass=replace_unclass)
                elif os.path.isdir(dir_path):
                    files = os.listdir(dir_path)
                    for file in files:
                        filepath = dir_path + '/' + file
                        print('Processing file', filepath)
                        construct_dataset(filepath, dataset_savefile, tags_savefile=tags_savefile, use_headwords=use_headwords, replace_nums=replace_nums, replace_unclass=replace_unclass)


def construct_dataset(bnc_xml_filename, dataset_savefile, tags_savefile=None, use_headwords=False, include_heads=False, replace_nums=False, replace_unclass=False):
    """
    Opens a BNC XML data file, processes the text
    and appends it to a save file
    
    Parameters
    ----------
    bnc_xml_filename : str
        filepath to the BNC XML file to process
    dataset_savefile : str
        filepath to the file to save the processed
        data to
    tags_savefile : str, optional
        filepath to the file to save the POS tag
        data to (default: None)
    use_headwords : bool, optional
        whether to use headwords or the raw
        version of the words (default: False)
    include_heads : bool, optional
        whether to include the headings (e.g.
        titles) or only process the body of the
        text (default: True)
    replace_nums : bool, optional
        whether to replace numbers with a default
        tag <NUM> (default: False)
    replace_unclass : bool, optional
        whether to replace "unclassified" (i.e.) with a
        default tag <UNC> (default: False)
    """
    
    if tags_savefile:
        open_tags_file = open(tags_savefile, 'a+')
    else:
        print('No tags file, skipping')
        open_tags_file = dummy_context_mgr()
    
    with open(dataset_savefile, 'a+', encoding='utf-8') as savefile, \
        open_tags_file as tags_file:
        print('Reading data from', bnc_xml_filename)
        text, tags = raw_text_from_elem_tree(bnc_xml_filename, use_headwords=use_headwords, replace_nums=replace_nums, replace_unclass=replace_unclass)
        print('Writing text data to', dataset_savefile)
        savefile.write(text)
        if tags_savefile:
            print('Writing POS tag data to', tags_savefile)
            tags_file.write(tags)
        print('Done writing')


def build_vocabulary(counts_file, vocab_ixs_file=None, min_freq=1):
    ''''
    Builds a torchtext.vocab object from a CSV file of word
    counts and an optionally specified frequency threshold

    Requirements
    ------------
    import csv
    from collections import Counter
    import torchtext
    
    Parameters
    ----------
    counts_file : str
        path to counts CSV file
    vocab_ixs_file : str, optional
        path to save the vocabulary indices
        CSV file to (default: None)
    min_freq : int, optional
        frequency threshold, words with counts lower
        than this will not be included in the vocabulary
        (default: 1)
    
    Returns
    -------
    torchtext.vocab.Vocab
        torchtext Vocab object
    '''
    counts_dict = {}

    print(f'{"@" * 30}\nVOCABULARY CONSTRUCTION\n{"@" * 30}')
    print(f'Constructing vocabulary from counts file in {counts_file}')

    num_inc = 0
    num_exc = 0

    with open(counts_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            # FIRST COLUMN IS ASSUMED TO BE THE WORD AND
            # THE SECOND COLUMN IS ASSUMED TO BE THE COUNT
            w_count = int(row[1])
            counts_dict[row[0]] = w_count
            if w_count < min_freq:
                num_exc += w_count
            else:
                num_inc += w_count

    counts = Counter(counts_dict)
    del counts_dict
    
    vocabulary = torchtext.vocab.Vocab(counts, min_freq=min_freq, specials=['<unk>', '<sos>', '<eos>', '<pad>'])
    perc_toks = "{:.2f}".format((len(vocabulary) / len(counts)) * 100) + '%'
    print(f'{len(vocabulary)} unique tokens in vocabulary with minimum frequency {min_freq} ({perc_toks} of {len(counts)} unique tokens in full dataset)')

    perc_inc = "{:.2f}".format((num_inc / (num_inc + num_exc)) * 100) + '%'
    print(f'{num_inc} of {(num_inc + num_exc)} words, vocabulary coverage of {perc_inc}')
    
    if vocab_ixs_file:
        # SAVE LIST OF VOCABULARY ITEMS AND INDICES TO FILE
        with open(vocab_ixs_file, 'w+', encoding='utf-8') as v:
            vocabulary_indices = [[i, w] for i,w in enumerate(vocabulary.itos)]
            print(f'Writing vocabulary indices to {vocab_ixs_file}')
            csv.writer(v).writerows(vocabulary_indices)

    return vocabulary


def get_word_embs_for_vocab(vocabulary, word_embs_file, emb_name='word2vec-google-news-300', to_tensor=False, device=torch.device('cpu')):
    """
    Get pre-trained word embeddings for a
    given vocabulary

    Requirements
    ------------
    import numpy as np (if saving as .npy)
    import torch (if saving as .pt)
    import gensim.downloader
    
    Parameters
    ----------
    vocabulary : torchtext.vocab
        vocabulary object to embed
    word_embs_file : str
        path to save the embeddings to
    emb_name : string, optional
        pre-trained embedding model name
        (default: 'word2vec-google-news-300')
        (a list of available model names can be
        accessed with
        print(list(gensim.downloader.info()['models'].keys())))
    to_tensor : bool, optional
        whether to save numericalised data as tensors
        (default: False)
    device : torch.device, optional
        device for tensor construction
        (default: torch.device('cpu'))
    
    Returns
    -------
    torch.tensor
        Pre-trained embeddings for every element
        in the vocabulary
    """
    print(f'Loading pre-trained embeddings for model {emb_name}')
    embs = gensim.downloader.load(emb_name)
    data = {}

    print(f'Constructing embeddings file ({len(vocabulary)} tokens)')
    for ix in range(len(vocabulary)):
        word = vocabulary.itos[ix]
        if word in embs.vocab:
            data[word] = torch.tensor(embs[word], device=device) if to_tensor else embs[word]
        else:
            data[word] = torch.zeros(embs.vector_size, device=device) if to_tensor else np.zeros(embs.vector_size)

    emb_format = 'PyTorch tensor' if to_tensor else 'NumPy array'
    print(f'Saving {len(data)} word embeddings as {emb_format} to file at {word_embs_file}')
    if to_tensor:
        torch.save(data, word_embs_file)
    else:
        np.save(word_embs_file, data)
    
    #return data
    return torch.stack(list(data.values()))
    

def raw_text_from_elem_tree(filename, use_headwords=False, replace_nums=False, replace_unclass=False):
    """
    This function only processes the words in
    a sentence, skips punctuation, and can
    optionally use headwords instead of raw text.
    Headwords are the (lowercase) root form of
    the word, e.g. the word "Said" has headword
    "say".
    
    Requirements
    ------------
    import xml.etree.ElementTree as ET
    
    Parameters
    ----------
    filename : str
        path to XML file to parse
    use_headwords : bool, optional
        whether to use headwords or the raw
        version of the words (default: False)
    replace_nums : bool, optional
        whether to replace numbers with a default
        tag <NUM> (default: False)
    replace_unclass : bool, optional
        whether to replace "unclassified" (i.e.) with a
        default tag <UNC> (default: False)
    
    Returns
    -------
    str
        full processed text
    """
    
    root = ET.parse(filename).getroot()
    text = ''
    tags = ''

    for sent in root.iter(tag='s'):
        # Gets all text, ignores all the rest
        # print([w for w in sent.itertext()])
        
        tokens = []
        sent_text = ''
        sent_tags = ''

        for token in sent:
            if token.tag == 'mw':
                for w in token:
                    tokens.append(w)
            else:
                tokens.append(token)
        
        word = ''
        tag = ''

        for token in tokens:
            # Skip empty words
            if token.text is None: continue 
            word = token.text.strip()
            
            if token.tag == 'w':
                tag = token.attrib['pos']
                # Replace word with head word (lemma)
                if use_headwords:
                    word = token.attrib['hw']

                # Replace numbers with <NUM>
                if replace_nums and token.attrib['c5'] == 'CRD':
                    word = '<NUM>'
                
                # Replace "unclassified" words with <UNC>
                if replace_unclass and token.attrib['c5'] == 'UNC':
                    word = '<UNC>'
            elif token.tag == 'c':
                # For punctuation use C5 tag
                tag = token.attrib['c5']
            
            if word and tag:
                sent_text += word + ' '
                sent_tags += tag + ' '
        
        if sent_text and sent_tags:
            text += sent_text + '\n'
            tags += sent_tags + '\n'

    # @DEBUG:
    # with open(filename, 'r') as f:
    #     xml_lines = f.readlines()
    #     print(f'Lines in XML:{len(xml_lines)}')

    text_lines = text.count("\n")+1
    tag_lines = tags.count("\n")+1
    if text_lines != tag_lines:
        print(f'Lines in text:{text_lines} \t and tags: {tag_lines}')
    
    return text, tags


def transform_token(token, to_lower=True, replace_num=True, remove_punct=False):
    """
    Perform preliminary transformations
    to spaCy token

    Parameters
    ----------
    token : spaCy token object
        object containing the token text and tags
    to_lower : bool
        whether to convert tokens to lowercase
        (default: True)
    replace_num : bool
        whether to replace numbers with number
        format, e.g. 12.5 => ##.# (default: True)
    remove_punct : bool
        whether to remove punctuation tokens
        (default: False)
    """
    # Skip punctuation token
    if remove_punct and token.pos_ == 'PUNCT':
        return None

    token_text = token.lower_ if to_lower else token.text
    
    if replace_num and token.pos_ == 'NUM':
        token_text = format_num(token_text)
    
    return token_text


def format_num(number):
    """
    Change number string to a placeholder
    format. For example:
        12.43 => ##.##

    Requirements
    ------------
    import re

    Parameters
    ----------
    number : str
        number to format
    """

    return re.sub(r"[0-9]", '#', number)


def seqlist_from_raw_text(datafile, dataset_savefile, to_lower=True, replace_num=True, remove_punct=False):
    """
    Convert a raw text file into a dependency tree and a sequence
    list dictionary and save it to file

    Requirements
    ------------
    import spacy
    import numpy as np

    Parameters
    ----------
    datafile : str
        path to the raw text data file
    dataset_savefile : str
        path to save the new dataset file to
    to_lower : bool
        whether to convert tokens to lowercase
        (default: True)
    replace_num : bool
        whether to replace numbers with number
        format, e.g. 12.5 => ##.# (default: True)
    remove_punct : bool
        whether to remove punctuation tokens
        (default: False)
    """
    with open(datafile, 'r', encoding='utf-8') as d:
        print(f'Reading data in {datafile}')
        data = d.read().splitlines()

    print(f'Data size: {len(data)}')
    
    spacy.prefer_gpu()
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'textcat'])

    tok_dataset = []

    # with open(dataset_savefile, 'w+', encoding='utf-8') as s:
    for i, sent in enumerate(data):
        doc = nlp(sent)
        seq = []

        for token in doc:
            token_text = transform_token(token, to_lower=to_lower, replace_num=replace_num, remove_punct=remove_punct)

            # Skip empty strings and None values
            if token_text: seq.append([token_text, token.pos_])

        # s.write(seq)
        # s.write('\n')
        tok_dataset.append(seq)

        if i % (len(data)/10) == 0:
            print(f'{i} sentences processed', flush=True)
    
    print(f'Saving data ({len(tok_dataset)} sentences) to {dataset_savefile}')
    np.save(dataset_savefile, tok_dataset)


def numericalise_dataset(skipgram_dataset, num_data_file, vocabulary, to_tensor=False, device=torch.device('cpu'), write_batch=1000000, has_header=True):
    '''
    FUNCTION FOR SKIP GRAM DATASET
    Convert Skip gram CSV dataset from words to the
    corresponding indices in the provided vocabulary 

    Requirements
    ------------
    import torch
    
    Parameters
    ----------
    skipgram_dataset : str
        path to the file containing the NPY dataset
        with the Skip-gram dataset
    num_data_file : str
        path to save the numericalised data to
    vocabulary : torchtext.vocab
        vocabulary object to use to numericalise
    to_tensor : bool, optional
        whether to save numericalised data as tensors
        (default: False)
    device : torch.device, optional
        device for tensor construction
        (default: torch.device('cpu'))
    write_batch : int, optional
        how many datapoints to write per operation
        (default: 1000000)
    has_header : bool, optional
        whether the CSV data file has a header, so
        the first row can be skipped (default: True)
    '''
    data = np.load(skipgram_dataset)
    num_dataset = []

    WORD_IX = 0
    is_augmented = False

    for i, row in enumerate(data):
        if len(row) == 5:
            focus, ctx, sent_num, focus_i, ctx_pos = row
        elif len(row) == 6:
            synonym, ctx, sent_num, focus_i, ctx_pos, focus = row
            is_augmented = True
        if has_header and i == 0:
            print(f'Skipping header: {row}')
            continue
        
        focus_word = focus[WORD_IX] if not is_augmented else synonym
        focus_ix = vocabulary.stoi[focus_word]
        ctx_ix = vocabulary.stoi[ctx[WORD_IX]]

        num_dataset.append([focus_ix, ctx_ix])

        if not i % write_batch:
            print(f'{i} word pairs numericalised', flush=True)
    
    del data

    if to_tensor:
        print(f'Saving {len(num_dataset)} numericalised Skip-gram datapoints to tensor file at {num_data_file}')
        tensor_data = torch.tensor(num_dataset, device=device, dtype=torch.long)
        torch.save(tensor_data, num_data_file)
    else:
        print(f'Saving {len(num_dataset)} numericalised Skip-gram datapoints to NPY file at {num_data_file}')
        np.save(num_data_file, num_dataset)


def numericalise_dataset_csv(data_path, save_path, vocabulary, write_batch=100000, has_header=True):
    '''
    FUNCTION FOR SKIP GRAM DATASET
    Convert Skip gram CSV dataset from words to the
    corresponding indices in the provided vocabulary 

    Requirements
    ------------
    import csv
    
    Parameters
    ----------
    data_path : str
        path to the file containing the JSON dataset
        with the sequence list and dependency parse
        trees
    save_path : str
        path to save the numericalised data to
    vocabulary : torchtext.vocab
        vocabulary object to use to numericalise
    write_batch : int, optional
        how many datapoints to write per operation
        (default: 100000)
    has_header : bool, optional
        whether the CSV data file has a header, so
        the first row can be skipped (default: True)
    '''
    with open(data_path, 'r', encoding='utf-8') as d, \
        open(save_path, 'w+', encoding='utf-8') as s:

        print(f'Writing numericalised dataset to {save_path}')

        csv_data = csv.reader(d)
        if has_header:
            header = next(csv_data)
        writer = csv.writer(s)
        skipgram_data = [['focus_word', 'context_word']]

        for i, row in enumerate(csv_data):
            if len(row) >= 2:
                focus_word = row[0]
                context_word = row[1]
                # if focus_word in vocabulary.stoi.keys():
                #     focus_ix = vocabulary.stoi[focus_word]
                # else:
                #     focus_ix = vocabulary.unk_index

                # context_word = row[1]
                # if context_word in vocabulary.stoi.keys():
                #     context_ix = vocabulary.stoi[context_word]
                # else:
                #     context_ix = vocabulary.unk_index
                
                focus_ix = vocabulary.stoi[focus_word] if focus_word in vocabulary.stoi.keys() else vocabulary.unk_index
                context_ix = vocabulary.stoi[context_word] if context_word in vocabulary.stoi.keys() else vocabulary.unk_index
                skipgram_data.append([focus_ix, context_ix])

                if not i % write_batch:
                    writer.writerows(skipgram_data)
                    print(f'{i} lines written', flush=True)
                    skipgram_data = []
        
        print(f'Finished writing file: {i} lines')


def csv_reader_check_header(file_pointer):
    """
    Check whether a CSV file has a header,
    if it does skip it
    NOTE: Assumes second column of CSV is
    always numeric except for the header row

    Requirements
    ------------
    import csv

    Parameters
    ----------
    file_pointer : file object
        open CSV file to read

    Returns
    -------
    reader : csv.reader object
    """

    reader = csv.reader(file_pointer)
    header = next(reader)
    try:
        int(header[1])
    except:
        print(f'File has header, skipping: {header}')
    else:
        file_pointer.seek(0)
    
    return reader


def numeric_csv_to_npy(source_file, save_file):
    """
    Translate numericalised CSV dataset format
    to NPY format.
    
    NOTE: this function loads the full source
    file to memory, which might be problematic
    for larger files
    
    Requirements
    ------------
    import numpy as np
    import csv
    csv_reader_check_header (local function) 
    
    Parameters
    ----------
    source_file : str
        filepath to the source file, assumed
        to be a CSV file with two columns (no
        header) containing word indices (ints)
    save_file : str
        filepath to write the npy file to
    """
    with open(source_file, 'r') as f:
        # data = csv.reader(f)
        data = csv_reader_check_header(f)
        i = 0
        rows = []
        for row in data:
            rows.append([int(row[0]), int(row[1])])
        np.save(save_file, rows)


def train_validate_split(tokenised_npy_file, train_savefile, val_savefile, proportion=0.85):
    """
    Splits the raw data into a training and a
    validation set according to a specified
    proportion and saves each set to a separate
    file
    
    Requirements
    ------------
    import numpy as np
    
    Parameters
    ----------
    tokenised_npy_file : str
        filepath to the tokenised NPY file (assumed
        to be a list of sentences, which are lists
        of words)
    train_savefile : str
        filepath to the file to save the training
        data to
    val_savefile : str
        filepath to the file to save the validation
        data to
    proportion : float, optional
        proportion of training data to sample from
        the full dataset (default: 0.85)
    
    NOTE: memory intensive, possibly due to NumPy indexing
    """
    print(f'Train ({proportion})/Validate ({1-proportion}) split for {tokenised_npy_file}')
    full_dataset = np.load(tokenised_npy_file)
    split_mask = np.random.choice(a=[True, False], size=(len(full_dataset)), p=[proportion, 1-proportion])

    num_training = np.sum(split_mask)
    num_validation = len(full_dataset) - num_training

    print(f'Writing {num_training} training samples to {train_savefile}')
    np.save(train_savefile, full_dataset[split_mask])

    print(f'Writing {num_validation} validation samples to {val_savefile}')
    np.save(val_savefile, full_dataset[~split_mask])


def raw_text_train_validate_split(clean_data_file, train_savefile, val_savefile, tags_data_file=None, train_tags_savefile=None, val_tags_savefile=None, proportion=0.85):
    """
    Splits the raw data into a training and a
    validation set according to a specified
    proportion and saves each set to a separate
    file
    
    Requirements
    ------------
    import numpy as np
    
    Parameters
    ----------
    clean_data_file : str
        filepath to the clean text data (assumed
        to be separated into lines)
    train_savefile : str
        filepath to the file to save the training
        data to
    val_savefile : str
        filepath to the file to save the validation
        data to
    tags_data_file : str, optional
        filepath to the file containing the POS tags
        for the clean data (default: None)
    train_tags_savefile : str, optional
        filepath to save the training POS tags
        to (default: None)
    val_tags_savefile : str, optional
        filepath to save the validation POS tags
        to (default: None)
    proportion : float, optional
        proportion of training data to sample from
        the full dataset (default: 0.85)
    
    NOTE: memory intensive, possibly due to NumPy indexing
    """
    
    process_tags = tags_data_file and train_tags_savefile and val_tags_savefile
    
    if process_tags:
        tags_file = open(tags_data_file, 'r')
        train_tags_file = open(train_tags_savefile, 'w+')
        val_tags_file = open(val_tags_savefile, 'w+')
    else:
        print('No tags file, skipping')
        tags_file = dummy_context_mgr()
        train_tags_file = dummy_context_mgr()
        val_tags_file = dummy_context_mgr()
   
    with open(clean_data_file, 'r', encoding='utf-8') as d, \
        tags_file as td,\
        open(train_savefile, 'w+', encoding='utf-8') as train_save, \
        open(val_savefile, 'w+', encoding='utf-8') as val_save, \
        train_tags_file as train_tag_save, \
        val_tags_file as val_tag_save:
        
        print('Train/Validate split for', clean_data_file)
        data = d.readlines()
        tags = td.readlines()
        
        data_size = len(data)
        tags_size = len(tags)
        
        print('Data length', data_size)
        print('Tags length', tags_size)

        random_ix = np.random.randint(data_size)

        print(f'Random datapoint and POS tags at index {random_ix}: \n {data[random_ix]} \t {tags[random_ix]}')
        
        if data_size != tags_size:
            raise Exception('Data and tags sizes must match: %d != %d' % (data_size, tags_size))
        
        indices = np.random.choice(data_size, data_size, replace=False)
        num_train = int(proportion * data_size)
        
        print('Number of training examples:', num_train)
        
        train_data = ''
        for i in indices[:num_train]:
            train_data += data[i]
        print('Writing training set to', train_savefile)
        train_save.write(train_data)
        del train_data
        
        val_data = ''
        for i in indices[num_train:]:
            val_data += data[i]
        print('Writing validation set to', val_savefile)
        val_save.write(val_data)
        del val_data
        del data

        if process_tags:
            train_tags = ''
            for i in indices[:num_train]:
                train_tags += tags[i]
            print('Writing training tag set to', train_tags_savefile)
            train_tag_save.write(train_tags)
            del train_tags
            
            val_tags = ''
            for i in indices[num_train:]:
                val_tags += tags[i]
            print('Writing validation tag set to', val_tags_savefile)
            val_tag_save.write(val_tags)
            del val_tags
        
        train_size = len(indices[:num_train])
        val_size = len(indices[num_train:])
        
        print('Train size', train_size)
        print('Val size', val_size)
        

def dataset_sampling(data_file, augm_data_file, dataset_file, augm_dataset_file, max_context=5):
    """
    From existing SkipGram word pair dataset
    sample through the context position, align
    sampled pairs with augmented pairs and sample
    a single synonym from this alignment
    
    Input files are expected to be CSV files with
    the following format, where the first row is
    the header:
        Natural dataset:
        - 0 : focus_word
        - 1 : context_word
        - 2 : sent_num
        - 3 : focus_index
        - 4 : context_position
        - 5 : book_number
        
        Augmented dataset:
        - 0 : synonym
        - 1 : context_word
        - 2 : sent_num
        - 3 : focus_index
        - 4 : context_position
        - 5 : focus_word
        - 6 : book_number
    
    Requirements
    ------------
    import csv
    import re
    import random
    
    Parameters
    ----------
    data_file : str
        path to source dataset file
    augm_data_file : str
        path to source augmented dataset
        file
    dataset_file : str
        path to write sampled dataset file to 
    augm_dataset_file : str
        path to write sampled augmented dataset
        file to
    max_context : int, optional
        maximum size of the context window, all
        samples will be taken by sampling from 1
        to this number (default: 5)
    """
    # Open the two files at once: unaltered dataset
    # and augmented dataset
    with open(data_file, 'r', encoding='utf-8', errors='replace') as d_file, \
        open(augm_data_file, 'r', encoding='utf-8', errors='replace') as a_file, \
        open(dataset_file, 'w+', encoding='utf-8', errors='replace', newline='') as s_file, \
        open(augm_dataset_file, 'w+', encoding='utf-8', errors='replace', newline='') as a_s_file:
        
        print('Data file: ', data_file)
        print('Augmented data file: ', augm_data_file)
        data = csv.reader(d_file)
        a_data = csv.reader(a_file)
        
        # Create two CSV writer objects: one for
        # original data and another for augmented data
        sample_file = csv.writer(s_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
        a_sample_file = csv.writer(a_s_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
        
        # Get the columns from the dataset
        header = next(data)
        cols = {name : i for i, name in enumerate(header)}
        
        # Get the columns from the augmented dataset
        a_header = next(a_data)
        a_cols = {name : i for i, name in enumerate(a_header)}
        
        print('Columns: ', cols)
        print('Augmented Columns: ', a_cols)
                
        has_book_num = 'book_number' in cols.keys() and 'book_number' in a_cols.keys()
        if has_book_num: print('Processing data with book numbers')

        # Initialise data arrays with the header information
        sample_file.writerow(header)
        a_sample_file.writerow(a_header)
        
        # Read the first row in the augmented dataset
        a_row = next(a_data, False)
        
        # Get the index columns
        a_book_num = int(a_row[a_cols['book_number']]) if has_book_num else 0
        a_sent_num = int(a_row[a_cols['sent_num']])
        a_focus_i = int(a_row[a_cols['focus_index']])
        a_ctx_pos = int(a_row[a_cols['context_position']])
        
        # Iterate through the dataset
        for row in data:
            # if row is empty, skip
            if not row: continue
            # If the row is a header skip it
            if re.search('[A-Za-z]+', row[cols['context_position']]): continue
            
            # Sample a random number between 1 and the
            # full context size
            rand_ctx = random.randint(1, max_context)
            
            # If sampled number is smaller than context position
            # add row to dataset
            if rand_ctx >= abs(int(row[cols['context_position']])):
                sample_file.writerow(row)
                
                book_num = int(row[cols['book_number']]) if has_book_num else 0
                sent_num = int(row[cols['sent_num']])
                focus_i = int(row[cols['focus_index']])
                ctx_pos = int(row[cols['context_position']])
                
                # Cycle through the augmented set while its
                # indices are smaller or equal to the ones
                # in the full dataset, or while there are
                # more rows
                while(
                    book_num >= a_book_num and
                    sent_num >= a_sent_num and
                    focus_i >= a_focus_i and
                    #ctx_pos >= a_ctx_pos and
                    a_row != False
                    ):
                    
                    # If all indices are the same, add the
                    # synonym row to the sampled augmented
                    # dataset
                    if(
                        book_num == a_book_num and
                        sent_num == a_sent_num and
                        focus_i == a_focus_i and
                        #ctx_pos == a_ctx_pos
                        rand_ctx >= abs(a_ctx_pos)
                        ):
                        a_sample_file.writerow(a_row)
                    
                    # Get the next row in the augmented data
                    a_row = next(a_data, False)
                    # If more rows, update the dataset indices
                    # if a_row != False:
                    if a_row:
                        # If the row is a header skip it
                        if re.search('[A-Za-z]+', a_row[cols['context_position']]):
                            a_row = next(a_data, False)
                            print('Bad row: ', a_row)
                        
                        # if a_row != False:
                        if a_row:
                            a_book_num = int(a_row[a_cols['book_number']]) if has_book_num else 0
                            a_sent_num = int(a_row[a_cols['sent_num']])
                            a_focus_i = int(a_row[a_cols['focus_index']])
                            a_ctx_pos = int(a_row[a_cols['context_position']])


def select_synonyms(data_file, save_file, vocabulary, syn_selection='ml'):
    """
    Find synonyms in the dataset,
    check whether they appear in the
    vocabulary. If multiple synonyms
    for a word appear in the vocabulary,
    randomly sample one. Alternatively,
    the synonym that appears the most in
    the data can be selected.

    NOTE: refactored to work with torchtext.Vocab
    
    The source data should have the
    augmented dataset format:
        - 0 : synonym
        - 1 : context_word
        - 2 : sent_num
        - 3 : focus_index
        - 4 : context_position
        - 5 : focus_word
        - 6 : book_number
    
    Requirements
    ------------
    import numpy as np
    import random
    import csv
    
    Parameters
    ----------
    data_file : str
        path to source (augmented) dataset file
    save_file : str
        path to write selected synonym dataset
        file to
    vocabulary : torchtext.Vocab
        torchtext vocabulary object
    syn_selection : str, optional
        synonym selection strategy, possible
        values are:
        - ml - maximum likelihood
        - s1 - randomly sample one
        - sw - randomly sample one (weighted by freq) #TODO
        - sn - randomly sample any number of syns
    """
    
    with open(data_file, 'r') as d, \
        open(save_file, 'w+', newline='') as syn_file:
        
        temp = []
        syns_temp = []
        
        data = csv.reader(d)
        # v_data = csv.reader(v)
        
        writer = csv.writer(syn_file, quoting=csv.QUOTE_ALL)
        
        # Read full vocabulary file, keep a list
        # of only the words
        # voc_full = [i for i in v_data]
        # vocabulary = [i[0] for i in voc_full]
        # vocabulary_counts = [i[1] for i in voc_full]
        # del voc_full
        
        # Get the columns from the dataset
        header = next(data)
        cols = {name : i for i, name in enumerate(header)}
        
        has_book_num = 'book_number' in cols.keys()
        if has_book_num: print('Processing data with book numbers')

        writer.writerow(header)
        
        for row in data:
            if not row: continue # Skip empty rows

            if len(temp) == 0:
                temp.append(row)
            else:
                last_row = temp[-1:][0]
                
                if has_book_num:
                    row_book_num = row[cols['book_number']]
                    last_row_book_num = last_row[cols['book_number']]
                else:
                    row_book_num = 0
                    last_row_book_num = 0

                if(row_book_num == last_row_book_num and
                    row[cols['sent_num']] == last_row[cols['sent_num']] and
                    row[cols['focus_index']] == last_row[cols['focus_index']]):
                    # If book, sentence, and focus is the same
                    # store as possible synonym
                    temp.append(row)
                else:
                    # When different, we finished processing
                    # the synonyms. Next step is selecting the
                    # final synonyms
                    syns = [i[cols['synonym']] for i in temp]
                    # Get onl
                    syns = np.unique(syns)
                    
                    syns_temp = []
                    
                    # For every unique synonym, check if it is
                    # in the vocabulary, if it isn't skip it
                    for syn in syns:
                        # if syn in vocabulary:
                        if syn in vocabulary.stoi.keys():
                            syns_temp.append(syn)
                        #else:
                            #print('\t\t', syn, ' is not in the vocabulary')
                    
                    # If at least one synonym is in the vocabulary
                    if len(syns_temp) > 0:
                        # If multiple synonyms, look for the one
                        # that appears most frequently
                        # TODO: change this, so it doesn't alter
                        # the word distributions (i.e. gives too
                        # much weight to frequent synonyms). Consider
                        # random sampling
                        if len(syns_temp) > 1:
                            retained_syn = ''
                            
                            if syn_selection == 'ml':
                                smallest_i = len(vocabulary)
                                for syn in syns_temp:
                                    # index = vocabulary.index(syn)
                                    index = vocabulary.stoi[syn]
                                    if index < smallest_i:
                                        smallest_i = index
                                # retained_syn = vocabulary[smallest_i]
                                retained_syn = vocabulary.itos[smallest_i]
                            elif syn_selection == 's1':
                                retained_syn = np.random.choice(syns_temp)
                            elif syn_selection == 'sn':
                                num_syns = np.random.randint(1, len(syns_temp)+1)
                                retained_syn = np.random.choice(syns_temp, num_syns, replace=False)
                            elif syn_selection == 'sw' or syn_selection == 'swn':
                                # List the indices for the synonyms
                                # indices = [vocabulary.index(syn) for syn in syns_temp]
                                indices = [vocabulary.stoi[syn] for syn in syns_temp]
                                # Get the counts for each synonym
                                # collect them in a list
                                # counts = [int(vocabulary_counts[i]) for i in indices]
                                counts = [vocabulary.freqs[vocabulary.itos[i]] for i in indices]
                                # Add up all counts
                                normaliser = np.sum(counts)
                                # Calculate weights by dividing
                                # counts by the normaliser
                                weights = counts / normaliser
                                # Randomly sample from list of synonyms
                                # weighted by the normalised counts
                                if syn_selection == 'swn':
                                    num_syns = np.random.randint(1, len(syns_temp)+1)
                                else:
                                    num_syns = 1
                                retained_syn = np.random.choice(syns_temp, num_syns, replace=False, p=weights)
                            else:
                                raise ValueError("unrecognised syn_selection %r" % syn_selection)
                                
                            # Check if retained_syn (string or list)
                            # is empty
                            if len(retained_syn) > 0:
                                # Changed syntax form '==' to 'in' to
                                # solve the case of multiple retained
                                # synonyms (i.e. deal with lists, not
                                # only strings)
                                temp = [i for i in temp if i[cols['synonym']] in retained_syn]
                                #print(retained_syn, ' retained syn')
                                writer.writerows(temp)
                                #print(temp)
                        else:
                            temp = [i for i in temp if i[cols['synonym']] == syns_temp[0]]
                            #print('Only syn: ', temp[0])
                            writer.writerows(temp)
                            #print(temp)
                    
                    # After processing the synonyms, restart
                    # the temp variable with the current row
                    temp = [row]
                
                #temp.append(row)



# Code required for conditional "with", used
# to only open the synonyms file when not None
@contextlib.contextmanager
def dummy_context_mgr():
    yield None



def seqlist_to_skipgram_data(seqlist_dataset, skipgram_dataset, ctx_size=5, write_batch=10000):
    """
    Generate datasets in the Skip Gram format
    (Mikolov et al., 2013): word pairs
    consisting of a centre or 'focus' word and
    the words within its context window

    The seqlist dataset is an NPY file expected
    to be an array of sentences, where each sentence
    is a list of [word, POS_tag] tuples
    
    The dataset is saved to an NPY file with the
    following columns:
        - 0 : focus_word (word, POS_tag)
        - 1 : context_word (word, POS_tag)
        - 2 : sent_num
        - 3 : focus_index
        - 4 : context_position
    
    Requirements
    ------------
    import numpy as np
    
    Parameters
    ----------
    seqlist_data_file : str
        path to raw text source file
    skipgram_data_file : str
        path to dataset save file
    ctx_size : int, optional
        context window size (default: 5)
    write_batch : int, optional
        how many datapoints to write per operation
        (default: 10000)
    """
    skipgram_data = [['focus_word', 'context_word', 'sent_num', 'focus_index', 'context_position']]

    data = np.load(seqlist_dataset)

    # Go through all sentences tokenised by spaCy
    # Word pairs are constrained to sentence appearances,
    # i.e. no inter-sentence word pairs
    for sent_i, sent in enumerate(data):
        num_tokens = len(sent)
        if num_tokens < 2: continue

        for focus_i, focus_word in enumerate(sent):
            ctx_min = max((focus_i - ctx_size), 0)
            ctx_max = min((focus_i + ctx_size), num_tokens-1)

            word_pairs = []

            # Go through every context word in the window
            for ctx_i in range(ctx_min, ctx_max+1):
                if ctx_i == focus_i: continue
                ctx_word = sent[ctx_i]
                ctx_pos = ctx_i - focus_i

                if focus_word and ctx_word:
                    word_pairs.append([
                            focus_word,
                            ctx_word,
                            sent_i,
                            focus_i,
                            ctx_pos
                        ])
            
            if word_pairs: skipgram_data.extend(word_pairs)
            
        if sent_i == 0:
            print(f'First sentence: {sent} \nWord pairs:\n{skipgram_data}\n')

        if not sent_i % write_batch:
            print(f'{sent_i} sentences processed', flush=True)

    del data

    print(f'Saving {len(skipgram_data)} Skip-gram datapoints to {skipgram_dataset}')
    np.save(skipgram_dataset, skipgram_data)


def skipgram_sampling(skipgram_dataset, skipgram_sampled_dataset, max_context=5, has_header=True):
    """
    From existing SkipGram word pair dataset
    sample through the context position

    %%%, align sampled pairs with augmented pairs and sample a single synonym from this alignment
    
    Input files are expected to be NPY files with
    the following format:
        - 0 : focus_word (word, POS_tag)
        - 1 : context_word (word, POS_tag)
        - 2 : sent_num
        - 3 : focus_index
        - 4 : context_position
    
    Requirements
    ------------
    import numpy as np

    import re
    import random
    
    Parameters
    ----------
    skipgram_dataset : str
        path to source dataset file
    skipgram_sampled_dataset : str
        path to write sampled dataset file to
    max_context : int, optional
        maximum size of the context window, all
        samples will be taken by sampling from 1
        to this number (default: 5)
    """
    data = np.load(skipgram_dataset)
    
    if has_header: sampled_data = [['focus_word', 'context_word', 'sent_num', 'focus_index', 'context_position']]
    
    # Sample a random number between 1 and the
    # full context size for every word pair
    # ()
    rand_contexts = np.random.choice(a=np.arange(1, max_context+1), size=len(data), replace=True)

    for i, [focus, ctx, sent_num, focus_i, ctx_pos] in enumerate(data):
        if has_header and i == 0:
            print(f'Skipping header: {[focus, ctx, sent_num, focus_i, ctx_pos]}')
            sampled_data = [[focus, ctx, sent_num, focus_i, ctx_pos]]
            continue
        
        # If sampled number is smaller than context position
        # add word pair to dataset
        if rand_contexts[i] >= abs(ctx_pos):
            sampled_data.append([focus, ctx, sent_num, focus_i, ctx_pos])
    
    del data

    print(f'Saving {len(sampled_data)} sampled Skip-gram datapoints to {skipgram_sampled_dataset}')
    np.save(skipgram_sampled_dataset, sampled_data)


def augment_npy_skipgram(skipgram_dataset, augm_dataset_file, vocabulary, ctx_size=5, write_batch=1000000, has_header=True, syn_selection='s1'):
    """
    Augment a Skip-gram dataset with synonyms
    for the focus word. Synonyms are extracted
    from synsets of the WordNet knowledge base
    that match the word and POS tag of the focus
    word. The Skip-gram dataset is an NPY file
    with the following columns:
        - 0 : focus_word (word, POS_tag)
        - 1 : context_word (word, POS_tag)
        - 2 : sent_num
        - 3 : focus_index
        - 4 : context_position
        
    
    Save augmented dataset to an NPY filw with the
    following columns:
        - 0 : synonym (word, POS_tag)
        - 1 : context_word (word, POS_tag)
        - 2 : sent_num
        - 3 : focus_index
        - 4 : context_position
        - 5 : focus_word (word, POS_tag)
    
    Requirements
    ------------
    synonym_selection (local function)
    import numpy as np
    from nltk.corpus import wordnet as wn
        WordNet object from NLTK
    
    Parameters
    ----------
    skipgram_dataset : str
        path to Skip-gram NPY dataset
    augm_dataset_file : str
        path to augmented dataset save file
    vocabulary : torchtext.Vocab
        torchtext vocabulary object
    ctx_size : int, optional
        context window size (default: 5)
    write_batch : int, optional
        how many datapoints to write per operation
        (default: 1000000)
    has_header : bool, optional
        whether the dataset has a header row
        (default: True)
    syn_selection : str, optional
        synonym selection strategy, possible
        values are:
        - ml - maximum likelihood
        - s1 - randomly sample one
        - sn - randomly sample any number of syns
        - sw - randomly sample one (weighted by freq)
        - swn - randomly sample any number of syns
                (weighted by freq)
        (default: 's1')
    """
    
    # dataset = [['focus_word', 'context_word', 'sent_num', 'focus_index', 'context_position']]
    augm_dataset = [['synonym', 'context_word', 'sent_num', 'focus_index', 'context_position', 'focus_word']]
    # Convert universal POS tags to WordNet types
    # https://universaldependencies.org/u/pos/all.html
    # (skip proper nouns)
    wn_tags = {
        'ADJ': wn.ADJ,
        'ADV': wn.ADV,
        'NOUN': wn.NOUN,
        #'PROPN': wn.NOUN,
        'VERB': wn.VERB
    }

    data = np.load(skipgram_dataset)
    
    WORD_IX = 0
    TAG_IX = 1
    
    synonym = ''
    prev_word = ''
    prev_sent = -1
    prev_ix = -1

    for i, [focus, ctx, sent_num, focus_i, ctx_pos] in enumerate(data):
        if has_header and i == 0:
            print(f'Skipping header: {[focus, ctx, sent_num, focus_i, ctx_pos]}')
            continue
        
        focus_word = focus[WORD_IX]
        focus_tag = focus[TAG_IX]
        context_word = ctx[WORD_IX]
        context_tag = ctx[TAG_IX]

        # If the POS tag is part of the
        # pre-specified tags
        if focus_tag in wn_tags:
            # If focus word is the same, repeat the same synonym
            if prev_word != focus_word or prev_sent != sent_num or prev_ix != focus_i:
                synsets = wn.synsets(focus_word, wn_tags[focus_tag])
                # Single occurrence of every synonym
                synonym_set = set()
                
                for syn in synsets:
                    for lem in syn.lemmas():
                        # Get the synonym in lowercase
                        synonym_set.add(lem.name().lower())

                synonym = synonym_selection(
                            synonym_set,
                            vocabulary,
                            syn_selection=syn_selection)
            else:
                prev_word = focus_word
                prev_sent = sent_num
                prev_ix = focus_i

            if synonym:
                augm_dataset.append([synonym, [context_word, context_tag], sent_num, focus_i, ctx_pos, [focus_word, focus_tag]])

        if not i % write_batch:
            print(f'{i} word pairs processed', flush=True)
    
    del data

    print(f'Saving {len(augm_dataset)} augmented Skip-gram datapoints to {augm_dataset_file}')
    np.save(augm_dataset_file, augm_dataset)
            


def synonym_selection(synonym_set, vocabulary, syn_selection='s1'):
    """
    Given a set of synonyms, select one
    or a subset based on specific criteria
    like their frequency in the data or
    random selection
    
    Requirements
    ------------
    import numpy as np
    import random
    import csv
    
    Parameters
    ----------
    synonym_set : set
        set of available synonyms
    vocabulary : torchtext.Vocab
        torchtext vocabulary object
    syn_selection : str, optional
        synonym selection strategy, possible
        values are:
        - ml - maximum likelihood
        - s1 - randomly sample one
        - sn - randomly sample any number of syns
        - sw - randomly sample one (weighted by freq)
        - swn - randomly sample any number of syns
                (weighted by freq)
        (default: 's1')
    """
    potential_syns = []
    syn_freqs = []
    
    for syn in synonym_set:
        # if syn in vocabulary:
        if syn in vocabulary.stoi.keys():
            potential_syns.append(syn)
            syn_freqs.append(vocabulary.freqs[syn])

    syn_freqs /= np.sum(syn_freqs)

    if len(potential_syns) == 0: return []
    if len(potential_syns) == 1: return potential_syns[0]

    # Maximum likelihood, select synonym with
    # the highest frequency in the vocabulary
    if syn_selection == 'ml':
        return potential_syns[np.argmax(syn_freqs)]

    # Pick one synonym randomly
    if syn_selection == 's1':
        return np.random.choice(potential_syns)
    
    # Pick one synonym randomly, weighted by
    # its frequency in the dataset
    if syn_selection == 'sw':
        return np.random.choice(potential_syns, p=syn_freqs)
    
    num_syns = np.random.randint(1, len(potential_syns)+1)
    
    # Pick a random number of synonyms randomly
    if syn_selection == 'sn':
        return np.random.choice(potential_syns, size=num_syns, replace=False)
    
    # Pick a random number of synonyms randomly,
    # weighted by their frequencies in the dataset
    if syn_selection == 'swn':
        return np.random.choice(potential_syns, p=syn_freqs, size=num_syns, replace=False)
    else:
        raise ValueError(f'Unrecognised syn_selection {syn_selection}')


def process_bnc_data(raw_data_file, dataset_file, tags_file=None, augm_dataset_file=None, ctx_size=5, write_batch=10000):
    """
    ADAPTED FOR THE BNC DATASET
    
    Generate datasets in the Skip Gram format
    (Mikolov et al., 2013): word pairs
    consisting of a centre or 'focus' word and
    the words within its context window
    
    The dataset is saved to a CSV file with the
    following columns:
        - 0 : focus_word
        - 1 : context_word
        - 2 : sent_num
        - 3 : focus_index
        - 4 : context_position
        
    
    Augmented dataset:
        - 0 : synonym
        - 1 : context_word
        - 2 : sent_num
        - 3 : focus_index
        - 4 : context_position
        - 5 : focus_word
    
    Requirements
    ------------
    import csv
        CSV reading and writing library
    import re
        regular expression library
    import os.path
        filepath functions
    from nltk.corpus import wordnet as wn
        WordNet object from NLTK
    
    Parameters
    ----------
    raw_data_file : str
        path to raw text source file
    dataset_file : str
        path to dataset save file
    tags_file : str, optional
        path to POS tags corresponding to the
        raw text (default: None)
    augm_dataset_file : str, optional
        path to augmented dataset save file
        (default: None)
    ctx_size : int, optional
        context window size (default: 5)
    write_batch : int, optional
        how many datapoints to write per operation
        (default: 10000)
    """
    
    num_lines = 0
    num_words = 0
    
    vocabulary = []
    
    dataset = [['focus_word', 'context_word', 'sent_num', 'focus_index', 'context_position']]
    
    augment = (tags_file is not None) and (augm_dataset_file is not None)
    
    if augment:
        augm_dataset = [['synonym', 'context_word', 'sent_num', 'focus_index', 'context_position', 'focus_word', 'pos_tag']]
        # Convert universal POS tags to WordNet types
        # https://universaldependencies.org/u/pos/all.html
        # (skip proper nouns)
        wn_tag_dict = {
            'ADJ': wn.ADJ,
            'ADV': wn.ADV,
            'SUBST': wn.NOUN,
            #'PROPN': wn.NOUN,
            'VERB': wn.VERB
        }
        open_tags_file = open(tags_file, 'r')
        open_augm_file = open(augm_dataset_file, 'w+')
    else:
        print('No tags file, skipping')
        augm_dataset = []
        open_tags_file = dummy_context_mgr()
        open_augm_file = dummy_context_mgr()
    
    # Open the file with UTF-8 encoding. If a character
    # can't be read, it gets replaced by a standard token
    with open(raw_data_file, 'r', encoding='utf-8', errors='replace') as f, \
        open_tags_file as td, \
        open(dataset_file, 'w+', encoding='utf-8', errors='replace') as d, \
        open_augm_file as a:
        print('Cleaning and processing ', raw_data_file)
        
        data = f.readlines()
        wr = csv.writer(d, quoting=csv.QUOTE_NONNUMERIC)
        if augment:
            tags = td.readlines()
            wra = csv.writer(a, quoting=csv.QUOTE_NONNUMERIC)
        
        sent_num = 0
        total_word_pairs = 0
        total_augm_word_pairs = 0
        
        # Go through all sentences tokenised by spaCy
        # Word pairs are constrained to sentence appearances,
        # i.e. no inter-sentence word pairs
        for sent_i, sent in enumerate(data):
            # ## DEBUGGING STEP, REMOVE FOR PRODUCTION!
            # if sent_i > 2110000:
            #     token_list = [w for w in sent.strip().split(' ') if w != '']
            #     sent_len = len(token_list)
            #     sent_tags = tags[sent_i].strip().split(' ')
                
            #     if sent_len != len(sent_tags):
            #         print(f'sent_i: {sent_i} \t sent_len: {sent_len} \t len(sent_tags): {len(sent_tags)} \t sent_tags: {sent_tags} \t sent: "{token_list}"')
            #         # return False
            # '''
            # Remove multiple white spaces
            sent = ' '.join(sent.split())
            token_list = sent.strip().split(' ') #[token for token in sent]
            num_tokens = len(token_list)
            
            # Skip processing if sentence is only one word
            if num_tokens > 1:
                for focus_i, token in enumerate(token_list):
                    word_pairs = []
                    augment_pairs = []
                    
                    # BYPASSED: original formulation, sampling context
                    # size, from 1 to ctx_size
                    #context_size = random.randint(1, ctx_size)
                    context_size = ctx_size
                    
                    context_min = focus_i - context_size if (focus_i - context_size >= 0) else 0
                    context_max = focus_i + context_size if (focus_i + context_size < num_tokens-1) else num_tokens-1
                    
                    focus_word = token
                        
                    # Go through every context word in the window
                    for ctx_i in range(context_min, context_max+1):
                        if (ctx_i != focus_i):
                            context_word = token_list[ctx_i]
                            
                            ctx_pos = ctx_i - focus_i
                            
                            if focus_word and context_word:
                                word_pairs.append([focus_word, context_word,sent_num, focus_i, ctx_pos])
                    
                    # If word_pairs is not empty, that means there is
                    # at least one valid word pair. For every non-stop focus
                    # word in these pairs, augment the dataset with external
                    # knowledge bases
                    if len(word_pairs) > 0 and augment:
                        sent_tags = tags[sent_i].strip().split(' ')
                        if num_tokens != len(sent_tags):
                            continue
                        
                        word_pos_tag = sent_tags[focus_i]
                        
                        # If the POS tag is part of the
                        # pre-specified tags
                        if word_pos_tag in wn_tag_dict:
                            synsets = wn.synsets(focus_word, wn_tag_dict[word_pos_tag])
                            
                            # Keep track of accepted synonyms,
                            # to avoid adding the same synonym
                            # multiple times to the dataset
                            accepted_synonyms = []
                            
                            # Cycle through the possible synonym
                            # sets in WordNet
                            for syn_num, syn in enumerate(synsets):
                                # Cycle through all the lemmas in
                                # every synset
                                for lem in syn.lemmas():
                                    # Get the synonym in lowercase
                                    synonym = lem.name().lower()
                                    
                                    # Removes multi-word synonyms
                                    # as well as repeated synonyms
                                    if not re.search('[-_]+', synonym) and focus_word != synonym and synonym not in accepted_synonyms:
                                        accepted_synonyms.append(synonym)
                                        
                                        for fw, c, sn, fi, cp in word_pairs:
                                            augment_pairs.append([synonym, c, sn, fi, cp, fw])
                    
                    if len(word_pairs) > 0:
                        dataset.extend(word_pairs)
                    if len(augment_pairs) > 0:
                        augm_dataset.extend(augment_pairs)
                sent_num += 1

            if not sent_i % write_batch:
                # only write to file every write_batch
                print(f'{sent_i} sentences processed, adding {len(dataset)} new rows to {dataset_file}', flush=True)
                total_word_pairs += len(dataset)
                wr.writerows(dataset)
                dataset = [[]] # restart dataset list
                if augment:
                    print(f'{sent_i} sentences processed, adding {len(augm_dataset)} new augmented rows to {augm_dataset_file}', flush=True)
                    total_augm_word_pairs += len(augm_dataset)
                    wra.writerows(augm_dataset)
                    augm_dataset = [[]]

    print(f'Finished processing: \n\t {total_word_pairs} word pairs in natural dataset \n\t {total_augm_word_pairs} word pairs in augmented dataset')
    
    '''
    with open(dataset_file, 'w+', newline='', encoding='utf-8') as f:
        wr = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        wr.writerows(dataset)
        
    if augment:
        with open(augm_dataset_file, 'w+', newline='', encoding='utf-8') as f:
            wr = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
            wr.writerows(augm_dataset)
    '''


def basic_tokenise(datafile, preserve_sents=True):
    """
    Tokenise a raw text file by simply splitting
    by white spaces
    
    Parameters
    ----------
    datafile : str
        path to text file to tokenise
    preserve_sents : bool, optional
        whether to use preserve the sentence
        separation by constructing a list of
        lists, if false returns a single list
        with all words in the text
        (default: True)
    
    Returns
    -------
    [str] OR [[str]]
        list, or list of lists, of tokenised text
    """
    tokenised_data = []
    with open(datafile, 'r', encoding='utf-8') as d:
        i = 0
        j = 0
        for line in d.readlines():
            words = line.strip().split(' ')
            if preserve_sents:
                tokenised_data.append(words)
            else:
                tokenised_data.extend(words)
            i += len(words)
            j += 1
        print('Num words ', i)
        print('Num lines ', j)
        print('Last words', words)
    
    return tokenised_data


def word_counts(tokenised_data, save_file):
    """
    Given a list (or list of lists) of tokenised
    text data calculates word counts and saves
    a CSV file consisting of:
    - word
    - raw count
    - frequency (normalised count)
    
    Requirements
    ------------
    from collections import Counter
    import csv
    
    Parameters
    ----------
    tokenised_data : [str] OR [[str]]
        list (or list of lists) of words to count
    save_file : str
        filepath to the data file to save counts to
    """
    
    if isinstance(tokenised_data[0], list):
        print('Data is multidimensional, flattening...')
        tokenised_data = [item for sublist in tokenised_data for item in sublist]
    
    word_counts = Counter(tokenised_data)
    total_words = sum(word_counts.values())
    
    counts_list = [[word, num, (float(num)/total_words)] for word, num in word_counts.items()]
    
    print(word_counts.most_common(10))
    print('Number of words: ', total_words)
    print('Number of distinct words: ', len(counts_list))
    
    with open(save_file, 'w+', encoding='utf-8', newline='') as s:
        writer = csv.writer(s)
        writer.writerows(counts_list)


def sample_and_process_books(gutenberg_path, processed_books_path, num_books=10, verbose=True):
    """
    Build a validation dataset from text
    sources. This implementation is designed
    to work with Gutenberg books. It randomly
    samples books, checks they are not part
    of the training set ('processed_books'),
    and processes them following the same
    steps as with the training data.
    
    This functions generates two CSV files,
    one for the natural data, and another
    for the augmented data.
    
    Requirements
    ------------
    import datetime
    from utils.funcs import sample_files, process_gutenberg_data
    
    Parameters
    -------
    gutenberg_path : str
        path to the Gutenberg books directory,
        the directory should be populated only
        by books in Gutenberg format (simple
        text)
    processed_books_path : str
        path to the list of processed books
        file. This helps prevent a book being
        processed more than once
    num_books : int, optional
        number of books to sample and process
        (default: 10)
    verbose : bool, optional
        whether to print out information
        about the sampled books and
        processing (default: True)
    
    Returns
    -------
    str, str
        [0] path to the dataset file
        [1] path to the augmented dataset
            file
    """
    sampled_files = sample_files(gutenberg_path, num_samples=num_books, verbose=verbose)
    
    sampled_books = []

    with open(processed_books_path, 'r') as f:
        processed_books = f.read().splitlines()
        
        for book in sampled_files:
            if book in processed_books:
                if verbose:
                    print(book, 'is in the processed books')
            else:
                sampled_books.append(book)
        
        now_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
        dataset_file = 'dataset/' + now_datetime + '-dataset.csv'
        augm_dataset_file = 'dataset/' + now_datetime + '-augm_dataset.csv'
        
        for doc_num, book in enumerate(sampled_books):
            read_file = gutenberg_path + book
            processed = process_gutenberg_data(read_file, dataset_file, augm_dataset_file, doc_num, ctx_size=5)
    
    if verbose:
        print('Sampled books:')
        print(sampled_books)
    
    return dataset_file, augm_dataset_file



def skipgram_data_from_gutenberg(gutenberg_path, processed_books_path, vocab_file, syn_sel_file, num_books=10):
    """
    Full pipeline to build SkipGram dataset
    from Gutenberg books. Requires a directory
    containing only Gutenberg books in simple
    text format.
    
    Steps:
    - Cleans and preprocesses the data
    - Constructs SkipGram word pairs and synonym
      augmented word pairs
    - Samples natural word pairs by their position
      in the context window (from Mikolov et al.,
      2013)
    - Samples a single synonym per focus word
    - Creates a lightweight dataset containing
      only the indices of the words in the pair
    - Saves optimised lightweight file in NPY
      format
    
    Saves a series of files:
    
    sample_and_process_books
    - dataset/2019-06-12_13.54.17-dataset.csv
    - dataset/2019-06-12_13.54.17-augm_dataset.csv
    
    dataset_sampling
    - dataset/2019-06-12_13.54.17-dataset_SMPLD.csv
    - dataset/2019-06-12_13.54.17-augm_dataset_SMPLD.csv
    
    select_synonyms
    - syn_sel_file
    
    lightweight_dataset
    - dataset/2019-06-12_13.54.17-dataset_SMPLD_LT.csv
    - dataset/2019-06-12_13.54.17-augm_dataset_SMPLD_LT.csv
    
    Requirements
    ------------
    from utils.funcs import sample_files, process_gutenberg_data, dataset_sampling, select_synonyms, lightweight_dataset, lt_to_npy
    sample_and_process_books()
    
    Parameters
    ----------
    gutenberg_path : str
        path to the Gutenberg books directory,
        the directory should be populated only
        by books in Gutenberg format (simple
        text)
    processed_books_path : str
        path to the list of processed books
        file. This helps prevent a book being
        processed more than once
    vocab_file : str
        path to the vocabulary file
    syn_sel_file : str
        path to the synonym selection file
    """
    data_file, augm_dataset_file = sample_and_process_books(gutenberg_path, processed_books_path, num_books=num_books)
    
    smpld_data_file = re.sub('.csv', '_SMPLD.csv', dataset_file)
    smpld_augm_data_file = re.sub('.csv', '_SMPLD.csv', dataset_file)
    
    dataset_sampling(data_file, augm_data_file, smpld_data_file, smpld_augm_data_file)
    
    select_synonyms(smpld_augm_data_file, syn_sel_file, vocab_file, syn_selection='sw')
    
    LT_smpld_data_file = re.sub('.csv', '_LT.csv', smpld_dataset_file)
    LT_smpld_augm_data_file = re.sub('.csv', '_LT.csv', smpld_augm_dataset_file)
    
    lightweight_dataset(smpld_data_file, vocab_file, LT_smpld_data_file)
    lightweight_dataset(syn_sel_file, vocab_file, LT_smpld_augm_data_file)
    
    npy_data_file = re.sub('.csv', '', LT_smpld_dataset_file)
    npy_augm_data_file = re.sub('.csv', '', LT_augm_smpld_dataset_file)
    
    lt_to_npy(LT_smpld_data_file, npy_data_file)    
    lt_to_npy(LT_smpld_augm_data_file, npy_augm_data_file)