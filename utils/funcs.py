###
#
# Support functions
#
###

"""
Required imports
"""
import re # doc_stats / gutenberg_spacing / process_data / dataset_sampling
import csv # build_vocabulary / word_ID / word_zipf_distribution / dataset_sampling / select_synonyms / lightweight_dataset 
from scipy import special # word_zipf_distribution / 
import numpy as np # word_zipf_distribution / select_synonyms / save_param_to_npy
from os import listdir # sample_files
import os.path # process_data
from random import choices # sample_files
import random # dataset_sampling / select_synonyms

import os
import psutil

import spacy # doc_stats / process_data
from nltk import pos_tag # process_data
from nltk.corpus import wordnet as wn # process_data

from collections import Counter # process_data

import torch

def dir_validation(dir_path):
    '''
    Model directory housekeeping: make sure
    directory exists, if not create it and make
    sure the path to the directory ends with '/'
    to allow correct path concatenation

    Requirements
    ------------
    import os

    Parameters
    ----------
    dir_path : str
        directory path to validate or correct
    
    Returns
    -------
    str
        validated directory path
    '''
    if not os.path.isdir(dir_path):
        print(f'{dir_path} directory does not exist, making directory')
        os.mkdir(dir_path)
    if not dir_path.endswith('/'): dir_path += '/'
    return dir_path


def print_parameters(parameters):
    '''
    Pretty print all model parameters

    Parameters
    ----------
    parameters : {str : X }
        parameter dictionary, where the keys are
        the parameter names with their corresponding
        values
    '''
    
    # PRINT PARAMETERS
    print('\n=================== MODEL PARAMETERS: =================== \n')
    for name, value in parameters.items():
        # num_tabs = int((32 - len(name))/8) + 1
        # tabs = '\t' * num_tabs
        num_spaces = 30 - len(name)
        spaces = ' ' * num_spaces
        print(f'{name}: {spaces} {value}')
    print('\n=================== / MODEL PARAMETERS: =================== \n')


def memory_stats(device=torch.device('cpu')):
    '''
    Memory usage for a specific device
    (ONLY WRITTEN FOR GPU MEMORY)
    TODO: implement for CPU

    Parameters
    ----------
    device : torch.device, optional
        the torch device to track memory for
        (default: torch.device('cpu'))
    '''
    conversion_rate = 2**30 # CONVERT TO GB
    # print('\n +++++++++++ torch.cuda.memory_stats\n')
    # print(torch.cuda.memory_stats(device=device))
    
    print('\n +++++++++++ torch.cuda.memory_summary\n')
    print(torch.cuda.memory_summary(device=device))
    
    # print('\n +++++++++++ torch.cuda.memory_snapshot\n')
    # print(torch.cuda.memory_snapshot())

    print('\n\n +++++++++++ torch.cuda.memory_allocated\n')
    print((torch.cuda.memory_allocated(device=device)/conversion_rate), 'GB')
    print('\n\n +++++++++++ torch.cuda.max_memory_allocated\n')
    print((torch.cuda.max_memory_allocated(device=device)/conversion_rate), 'GB')
    print('\n\n +++++++++++ torch.cuda.memory_reserved\n')
    print((torch.cuda.memory_reserved(device=device)/conversion_rate), 'GB')
    print('\n\n +++++++++++ torch.cuda.max_memory_reserved\n')
    print((torch.cuda.max_memory_reserved(device=device)/conversion_rate), 'GB')


def get_stop_words():
    """
    Get list of stop words from the
    NLTK library
    
    Returns
    -------
    list of strings
        list of stop words
    
    """
    stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    
    return stop_words


def build_vocabulary(counts_file, save_file, min_counts=None, vocab_size=10000):
    """
    Builds a vocabulary file from the
    word counts in a text corpus. The
    word counts should be in a CSV file
    with the following columns (no header):
    - 0 : word
    - 1 : counts
    - 2 : frequencies (counts / total words)
    
    The vocabulary size can be constrained
    based on two separate criteria:
    - Overall vocabulary size
    - Minimum appearances of a word in the
      corpus
    
    Saves a vocabulary CSV file with the same
    columns as the counts file.
    
    Requirements
    ------------
    import csv
    
    Parameters
    ----------
    counts_file : str
        path to the file containing
        the word counts
    save_file : str
        path to save the vocabulary
        file to
    min_counts : int, optional
        minimum appearances to include
        a word in the vocabulary
        (default: None)
    vocab_size : int, optional
        overall vocabulary size, this
        parameter is overridden when
        min_counts is not None
        (default: 10000)
    """
    with open(counts_file, 'r') as f:
        print('Opening file: ', counts_file)
        data = csv.reader(f, delimiter=',')
        
        # Sorting based on word counts (can use
        # percentage too, in float(row[2]))
        sorted_list = sorted(data, key=lambda row: int(row[1]), reverse=True)
        
        vocabulary = []
        # Initialise the cutoff to be the
        # end of the list
        vocab_cutoff = len(sorted_list)
        
        if min_counts is None:
            vocab_cutoff = vocab_size
        else:
            for i, word in enumerate(sorted_list):
                vocab_cutoff = i-1
                
                if int(word[1]) < int(min_counts):
                    break
            
            # If no elements made the cut raise
            # an exception
            if vocab_cutoff < 0:
                raise Exception('Empty list: no word appears more than %d times!' % (min_counts))
        
        # No checks required, if vocab size is
        # larger than list, the full (sorted)
        # list is returned
        vocabulary = sorted_list[:vocab_cutoff]
        
        print('Num words: ', len(vocabulary))
        print('Vocabulary start: ', vocabulary[:10])
        print('Vocabulary end: ', vocabulary[-10:])
        
        
        with open(save_file, 'w+', newline='') as f:
            wr = csv.writer(f, quoting=csv.QUOTE_ALL)
            wr.writerows(vocabulary)


def word_ID(word, vocab_file):
    """
    Get the ID of a word from a vocabulary
    file
    
    Requires
    --------
    import csv
    
    Parameters
    ----------
    word : str
        word to get the ID of
    vocab_file : str
        path to canonical dictionary file
    
    Returns
    -------
    int
        ID of the word, or -1 if word
        is not in the vocabulary
    
    """
    with open(vocab_file, 'r') as f:
        data = csv.reader(f)
        i = 0
        for row in data:
            if row[0] == word:
                print(i, row)
                return i
            i += 1
    return -1


def doc_stats(document):
    """
    Cleans a text document, tokenises it with
    spaCy, counts the words and creates a set
    of the unique words
    
    Requirements
    ------------
    import re
        regex library
    import spacy
        NLP library
    nlp = spacy.load('en_core_web_sm')
        Load the (previously downloaded)
        pre-trained spaCy models
        
    Parameters
    ----------
    document : str
    
    Returns
    -------
    set of str
        unique words in the document
    int
        number of words in the document
    """
    data = re.sub('\.*\n{2,}', '. ', document)
    data = re.sub('\n', ' ', data)
    data = re.sub('\s+', ' ', data)
    data.strip()
    
    processed_doc = nlp(data)
    
    #ignore_pos = ['PUNCT', 'SYM', 'X']
    
    word_list = []
    
    num_words = 0
    dist_words = set()
    
    # Check if missing words are not letters
    #pattern = '([^A-Za-z])+'
    
    for token in processed_doc:
        word = token.text.lower()
        
        # if not re.search(pattern, word):
        num_words += 1
        dist_words.add(word)
    
    return dist_words, num_words



def sample_files(dir_path, num_samples=10, verbose=False):
    """
    Randomly sample files from a directory
    
    Requirements
    ------------
    from os import listdir
        os utility to list directory
        contents
    from random import choices
        random sampling function
    
    Parameters
    ----------
    dir_path : str
        path to the directory to sample
    num_samples : int, optional
        number of files to sample
    verbose : bool, optional
        whether to print out the names of
        the sampled files
    
    Returns
    -------
    list of str
        list of sampled filenames
    """
    dir_contents = listdir(dir_path)
    book_selection = choices(dir_contents, k=num_samples)
    
    if verbose: print("\n".join(book_selection))
    
    return book_selection


def gutenberg_spacing(text):
    """
    Regular expressions to deal with white
    space in Gutenberg book files
    
    Requirements
    ------------
    import re
        regular expression library
    
    Parameters
    ----------
    text : str
        the text string to be processed
    
    Returns
    -------
    str
        clean text string
    """
    
    # Replace multiple new lines with a period, this
    # facilitates sentence segmentation
    clean_text = re.sub('\.*\n{2,}', '. ', text)
    # Replace single new line with white space
    # Gutenberg data has a lot of new lines in the middle
    # of sentences, which is problematic for sentence processing
    clean_text = re.sub('\n', ' ', clean_text)
    # Replace multiple white spaces or tabs with a single
    # white space
    clean_text = re.sub('\s+', ' ', clean_text)
    # Finally, remove trailing white spaces
    return clean_text.strip()


def process_gutenberg_data(read_file, dataset_file, augm_dataset_file, doc_num=-1, ctx_size=5):
    
    """
    Generate datasets in the Skip Gram format
    (Mikolov et al., 2013): word pairs
    consisting of a centre or 'focus' word and
    the words within its context window
    A 'natural' and an augmented dataset are 
    constructed from a text file. The natural
    dataset cycles through each word and treats
    it as a focus word to construct the
    corresponding word pairs. The augmented
    dataset replaces each focus word (only
    adjectives, adverbs, nouns, and verbs)
    with its synonyms from WordNet
    
    The datasets are saved to two CSV files
    with the following columns:
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
    import spacy
        spaCy NLP library
    from nltk import pos_tag
        NLTK POS tagger
    from nltk.corpus import wordnet as wn
        WordNet and FrameNet NLTK libraries
    import re
        regular expression library
    import os.path
        filepath functions
    
    Parameters
    ----------
    read_file : str
        path to raw text source file
    dataset_file : str
        path to dataset save file
    augm_dataset_file : str
        path to augmented dataset save file
    doc_num : int, optional
        document index (useful when
        processing multiple documents)
        (default: -1)
    ctx_size : int, optional
        context window size (default: 5)
    
    Returns
    -------
    bool
        False if file is too large to be
        processed, True otherwise
    
    TODO
    ----
    - Unifying upper/lowercase
    - Addressing Named Entities (token.ent_iob_, token.ent_type_)
    """
    num_lines = 0
    num_words = 0
    full_counter = Counter()
    
    vocabulary = []
    
    # Disable NER and categorisation to lighten processing
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'textcat'])
    
    # If the dataset file does not exist, add a header
    if os.path.exists(dataset_file):
        dataset = []
    else:
        dataset = [['focus_word', 'context_word', 'sent_num', 'focus_index', 'context_position', 'book_number']]
    if os.path.exists(augm_dataset_file):
        augm_dataset = []
    else:
        augm_dataset = [['synonym', 'context_word', 'sent_num', 'focus_index', 'context_position', 'focus_word', 'book_number']]
    
    # Open the file with UTF-8 encoding. If a character
    # can't be read, it gets replaced by a standard token
    with open(read_file, 'r', encoding='utf-8', errors='replace') as f:
        print('Cleaning and processing ', read_file)
        
        data = f.read()
        # Quick and dirty fix for long files
        # spaCy has a character limit of 1,000,000
        # TODO: split large files into multiple manageable files
        if len(data) > 999999:
            print('File ' + read_file + ' is too large to process. Skipping...')
            return False
        
        # After cleaning, the data is
        # processed with spaCy
        doc = nlp(gutenberg_spacing(data))
        # Might be unnecessary, deletes the variable
        # to clear memory
        del data
        
        # Go through all sentences tokenised by spaCy
        # Word pairs are constrained to sentence appearances,
        # i.e. no inter-sentence word pairs
        for sent_num, sent in enumerate(doc.sents):
            
            token_list = [token for token in sent]
            num_tokens = len(token_list)
            
            # Skip processing if sentence is only one word
            if len(token_list) > 1:
                for focus_i, token in enumerate(token_list):
                    t_pos = token.pos_
                    
                    # List of ignored tags
                    ignore_pos = ['PUNCT', 'SYM', 'X']
                    
                    # Temporary list of generated pairs for the current
                    # focus word. Will be used later when searching for
                    # synonyms for the focus word
                    word_pairs = []
                    augment_pairs = []
                    
                    # Only process if focus word is not punctuation
                    # (PUNCT), symbol (SYM), or unclassified (X), and
                    # if token is not only a symbol (not caught by spaCy)
                    if (t_pos not in ignore_pos
                        and re.sub(r'[^\w\s]', '', token.text).strip() != ''):
                        
                        # BYPASSED: original formulation, sampling context
                        # size, from 1 to ctx_size
                        #context_size = random.randint(1, ctx_size)
                        context_size = ctx_size
                        
                        context_min = focus_i - context_size if (focus_i - context_size >= 0) else 0
                        
                        context_max = focus_i + context_size if (focus_i + context_size < num_tokens-1) else num_tokens-1
                        
                        focus_word = token.text.lower()
                        
                        # Go through every context word in the window
                        for ctx_i in range(context_min, context_max+1):
                            # Check that context index is not the same as
                            # focus, that the context word is not in our
                            # 'ignore' list, and that the context is not
                            # white space or punctuation
                            if (ctx_i != focus_i
                                and token_list[ctx_i].pos_ not in ignore_pos
                                and re.sub(r'[^\w\s]', '', token_list[ctx_i].text).strip() != ''):
                                # Changing everything to lower case
                                # A more principled approach would uppercase
                                # named entities such as persons, companies
                                # or countries:
                                #   if token.ent_iob_ != 'O':
                                #       token.text.capitalize()
                                context_word = token_list[ctx_i].text.lower()
                                
                                ctx_pos = ctx_i - focus_i
                                
                                # If passes all checks (context different
                                # from target and neither tagged as)
                                word_pairs.append([focus_word, context_word,sent_num, focus_i, ctx_pos, doc_num])
                                
                        # If word_pairs is not empty, that means there is
                        # at least one valid word pair. For every non-stop focus
                        # word in these pairs, augment the dataset with external
                        # knowledge bases
                        if len(word_pairs) > 0 and not token.is_stop:
                            
                            # Convert to list of text words for NLTK
                            text_list = [token.text for token in token_list]
                            # POS tag with NLTK
                            nltk_pos = [pos_tag(text_list)]
                            
                            # We create nltk_pos with a single sentence
                            # so we can access it directly. We are accessing:
                            #   - Sentence 0
                            #   - Word number focus index
                            #   - POS tag (second column)
                            nltk_pos_tag = nltk_pos[0][focus_i][1]
                            
                            # If the POS tags of spaCy and NLTK agree
                            # then continue with the augmentation
                            if token.tag_ == nltk_pos_tag:
                                #print('Tags agree for ', nltk_pos[0][focus_i][1])
                                
                                # Convert universal POS tags to WordNet types
                                # https://universaldependencies.org/u/pos/all.html
                                # (skip proper nouns)
                                wn_tag_dict = {
                                    'ADJ': wn.ADJ,
                                    'ADV': wn.ADV,
                                    'NOUN': wn.NOUN,
                                    #'PROPN': wn.NOUN,
                                    'VERB': wn.VERB
                                }
                                
                                # If the POS tag is part of the
                                # pre-specified tags
                                if token.pos_ in wn_tag_dict:
                                    synsets = wn.synsets(focus_word, wn_tag_dict[token.pos_])
                                    
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
                                                
                                                for fw, c, sn, fi, cp, _ in word_pairs:
                                                    augment_pairs.append([synonym, c, sn, fi, cp, fw, doc_num])
                    
                    if len(word_pairs) > 0:
                        dataset.extend(word_pairs)
                    
                    if len(augment_pairs) > 0:
                        augm_dataset.extend(augment_pairs)
                    

    if len(dataset) > 0: print('Original dataset: ', len(dataset), len(dataset[0]))
    if len(augm_dataset) > 0: print('Augmented dataset: ', len(augm_dataset), len(augm_dataset[0]))
    
    # Look for the dataset file, if it doesn't exist create it ('a+')
    with open(dataset_file, 'a+', newline='') as f:
        wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        wr.writerows(dataset)
        
    with open(augm_dataset_file, 'a+', newline='') as f:
        wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        wr.writerows(augm_dataset)      
    
    return True


def get_word_knowledge(word, verbose=False):
    """
    Query the FrameNet and WordNet
    information related to a given word
    
    Parameters
    ----------
    word : str
        word to query
    verbose : bool, optional
        whether to print the information
        obtained from the query (default: False)
    
    Returns
    -------
    object
        FrameNet object returned by query
    object
        WordNet synset object returned by
        query
    """
    frames = fn.frames_by_lemma(r'(?i)'+word)
    synsets = wn.synsets(word)
    
    if verbose:
        print('Frames: ', frames)
        print('Synsets: ', synsets)
    
    return frames, synsets


# def memory_usage(legend='Memory usage'):
#     '''
#     Prints present CPU memory usage in percentage

#     Requirements
#     ------------
#     import os
#     import psutil

#     Parameters
#     ----------
#     legend : str, optional
#         legend to print with the usage information
#     '''
#     process = psutil.Process(os.getpid())
#     print(f'\n{"=" * 16} {legend} {"=" * 16}')
#     print(process.memory_percent())
#     print(f'{"=" * 16} {legend} {"=" * 16}\n')

def mem_check(device, legend=0):
    conversion_rate = 2**30 # CONVERT TO GB
    print(f'\n\n Mem check {legend}\n')
    print('GPU Usage:')
    mem_alloc = torch.cuda.memory_allocated(device=device) / conversion_rate
    mem_reserved = torch.cuda.memory_reserved(device=device) / conversion_rate
    os.system('nvidia-smi')
    print(f' +++++++++++ torch.cuda.memory_allocated {mem_alloc}GB', flush=True)
    print(f' +++++++++++ torch.cuda.memory_reserved {mem_reserved}GB \n', flush=True)
    print('\n\nCPU Usage:')
    pid = os.getpid()
    proc = psutil.Process(pid)
    mem_gb = "{:.2f}".format(proc.memory_info()[0]/2.**30)
    mem_percent = "{:.2f}".format(proc.memory_percent())
    print(f' +++++++++++ CPU used: {mem_gb}GB \t {mem_percent}%')


def lightweight_dataset(data_file, vocab_file, save_file):
    """
    Transforms the current SkipGram CSV
    dataset into a ligthweight version
    which consists of only word index pairs.
    
    Assumes first line of file is header
    with column names.
    
    TODO: save to NPY format
    
    Requirements
    ------------
    import csv
    
    Parameters
    ----------
    data_file : str
        path to source dataset file
    vocab_file : str
        path to canonical dictionary file
    save_file : str
        path to write selected synonym dataset
        file to
    
    """
    with open(data_file, 'r', encoding='utf-8', errors='replace') as d, \
        open(vocab_file, 'r', encoding='utf-8', errors='replace') as v, \
        open(save_file, 'w+', encoding='utf-8', errors='replace') as f:
        
        data = csv.reader(d)
        vocab_reader = csv.reader(v)
        
        # Get the columns from the dataset
        header = next(data)
        cols = {name : i for i, name in enumerate(header)}
        # {'focus_word': 0, 'context_word': 1, 'sent_num': 2, 'focus_index': 3, 'context_position': 4, 'book_number': 5}
        
        print('Processing file %r' % (data_file))
        print('Header: ', header)
        print('Saving to file %r' % (save_file))
        
        vocabulary = [w for w in vocab_reader]
        vocab_words = [w[0] for w in vocabulary]
        del vocabulary
        
        wr = csv.writer(f)
        
        num_word_pairs = 0
        missing_word_pairs = 0
        
        for row in data:
            # Check if file is natural or augmented
            focus = row[cols['synonym']] if 'synonym' in cols.keys() else row[cols['focus_word']]
            context = row[cols['context_word']]

            try:
                focus_i = vocab_words.index(focus)
                context_i = vocab_words.index(context)
                
                wr.writerow([focus_i, context_i])
                num_word_pairs += 1
            except:
                missing_word_pairs += 1
                #print(focus, context, 'not in dictionary')
                continue
        
        print('Processed %d word pairs. Missing %d pairs (not in vocabulary)' % (num_word_pairs, missing_word_pairs))


def save_param_to_npy(model, param_name, path):
    """
    Save PyTorch model parameter to NPY file
    
    Requirements
    ------------
    import numpy as np
    
    Parameters
    ----------
    model : PyTorch model
        the model from which to get the
        parameters
    param_name : str
        name of the parameter weights to
        save
    path : str
        path to the file to save the parameters
        to
    
    """
    for name, param in model.named_parameters():
        if name == param_name + '.weight':
            weights = param.data.cpu().numpy()
    
    np.save(path, weights)
    
    print("Saved ", param_name, " to ", path)


def get_vector_norms(vecs_file, save_file, norm=2):
    '''
    Calculate vector norms for vectors in NumPy
    file
    
    Requirements
    ------------
    import numpy as np
    
    Parameters
    ----------
    vecs_file : str
        path to the NPY file containing the
        vectors
    save_file : str
        path to save the calculated norms to
    norm : int/str, optional
        type of norm to use as described in
        numpy.linalg.norm (e.g. 'fro', 'nuc', 2)
        (default: 2)
    '''
    embs = np.load(embs_file)
    norms = np.linalg.norm(embs, ord=norm, axis=1)
    
    np.save(save_file, norms)


def npy_to_tsv(npy_file, vocab_file, save_file):
    """
    Convert NPY embeddings into a TSV file in
    word2vec format: word in the first column
    followed by the vector values
    
    Requirements
    ------------
    import numpy as np
    import csv
    
    Parameters
    ----------
    npy_file : str
        Path to the NPY embeddings file
    vocab_file : str
        Path to the vocabulary file
    save_file : str
        Path to save the TSV file to
    """
    
    embs = np.load(npy_file)
    
    with open(vocab_file, 'r') as v, \
        open(save_file, 'w+', newline='') as f:
        
        vocab = [w[0] for w in csv.reader(v)]
        writer = csv.writer(f, delimiter=' ', quoting=csv.QUOTE_NONNUMERIC)
        
        writer.writerow([len(vocab), 300])
        
        for i, row in enumerate(embs):
            if vocab[i] == ' ': vocab[i] = '\s'
            
            row_list = [vocab[i]]
            row_list.extend(row)
            writer.writerow(row_list)