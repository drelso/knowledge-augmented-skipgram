###
#
# Dataset processing and sampling utilities
#
###

from .funcs import sample_files, process_gutenberg_data, dataset_sampling, select_synonyms, lightweight_dataset, lt_to_npy
import time
import datetime

import contextlib

import csv
import re
from nltk.corpus import wordnet as wn


# Code required for conditional "with", used
# to only open the synonyms file when not None
@contextlib.contextmanager
def dummy_context_mgr():
    yield None

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
    ctx_size : int, optional
        context window size (default: 5)
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
                        # print(sent, token, sent_tags, focus_i)
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