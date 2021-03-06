###
#
# Tree2Seq training utility functions
#
##

import csv
from collections import Counter
import contextlib
import json
import time
import math

import numpy as np

import torch
import torchtext


## HACKISH: INITIALISE THE DEFAULT DEVICE ACCORDING TO
## WHETHER GPU FOUND OR NOT. NECESSARY TO PASS THE RIGHT
## DEVICE TO TREE PREPROCESSING PIPELINE
## TODO: CHANGE INTO AN ARGUMENT TO THE PIPELINE
default_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_vocabulary(counts_file, min_freq=1):
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

    print(f'Constructing vocabulary from counts file in {counts_file}')

    with open(counts_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            # FIRST COLUMN IS ASSUMED TO BE THE WORD AND
            # THE SECOND COLUMN IS ASSUMED TO BE THE COUNT
            counts_dict[row[0]] = int(row[1])

    counts = Counter(counts_dict)
    del counts_dict
    
    vocabulary = torchtext.vocab.Vocab(counts, min_freq=min_freq, specials=['<unk>', '<sos>', '<eos>', '<pad>'])
    print(f'{len(vocabulary)} unique tokens in vocabulary with (with minimum frequency {min_freq})')
    
    return vocabulary


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
    
    param_file = path + '-' + param_name
    
    np.save(param_file, weights)
    
    print("Saved ", param_name, " to ", path)


def list_to_tensor(x_list, device=torch.device('cpu')):
    return torch.tensor(x_list, device=device, dtype=torch.long)#dtype=torch.int)


def treedict_to_tensor(treedict, device=torch.device('cpu')):#default_device):
    tensor_dict = {}
    for key, value in treedict.items():
        if torch.is_tensor(value):
            tensor_dict[key] = value#.clone().detach().requires_grad_(True)
        else:
            tensor_dict[key] = torch.tensor(value, device=device, dtype=torch.long)#dtype=torch.int)#float).requires_grad_(True)#
    return tensor_dict


def construct_dataset_splits(dataset_path, vocabulary, split_ratio=.8):
    '''
    Construct torchtext.Dataset object and split into training,
    test and validation sets

    Requirements
    ------------
    import torchtext

    Parameters
    ----------
    dataset_path : str
        path to the file containing the JSON dataset
        with the sequence list and dependency parse
        trees
    vocabulary : torchtext.vocab
        vocabulary object to use to numericalise
    split_ratio : float, optional
        split ratio for the training set, the rest is
        used for validation (default: .8)
    
    Returns
    -------
    torchtext.Dataset
        train data split as torchtext.Dataset object
    torchtext.Dataset
        test data split as torchtext.Dataset object
    torchtext.Dataset
        validation data split as torchtext.Dataset object
    '''

    # seq_preprocessing = torchtext.data.Pipeline(list_to_tensor)
    # tree_preprocessing = torchtext.data.Pipeline(treedict_to_tensor)

    print(f'Constructing dataset from {dataset_path}')

    FOCUS_FIELD = torchtext.data.Field(
                                sequential=False, 
                                use_vocab=True,#vocab,
                                # preprocessing=seq_preprocessing
                            )

    CONTEXT_FIELD = torchtext.data.Field(
                                sequential=False,
                                use_vocab=True,
                                # preprocessing=tree_preprocessing,
                                is_target=True
                            )

    FOCUS_FIELD.vocab = vocabulary
    CONTEXT_FIELD.vocab = vocabulary

    skipgram_data = torchtext.data.TabularDataset(
        path=dataset_path,
        format='csv',
        skip_header=True,
        # fields={   # "focus_word","context_word","sent_num","focus_index","context_position"
        #     'focus'   :   ('focus_word', FOCUS_FIELD),
        #     'context'  :   ('context_word', CONTEXT_FIELD)
        #         }
        fields = [
            ('focus_word', FOCUS_FIELD),
            ('context_word', CONTEXT_FIELD),
            ('sent_num', None),
            ('focus_index', None),
            ('context_position', None)
        ]
        )

    train, validate = skipgram_data.split(split_ratio=split_ratio)

    print(f'Split sizes: \t train {len(train)} \t validate {len(validate)}')

    return train, validate


@contextlib.contextmanager
def dummy_context_mgr():
    '''
    Code required for conditional "with"

    Requirements
    ------------
    import contextlib
    '''
    yield None


def mem_check(device, num=0):
    conversion_rate = 2**30 # CONVERT TO GB
    print(f'\n\n Mem check {num}')
    mem_alloc = torch.cuda.memory_allocated(device=device) / conversion_rate
    mem_reserved = torch.cuda.memory_reserved(device=device) / conversion_rate
    print(f'+++++++++++ torch.cuda.memory_allocated {mem_alloc}GB')
    print(f' +++++++++++ torch.cuda.memory_reserved {mem_reserved}GB \n')

def run_model(data_iter, model, optimizer, criterion, vocabulary, device=torch.device('cpu'), phase='train', print_epoch=True):
    '''
    Run training or validation processes given a 
    model and a data iterator.

    Requirements
    ------------
    treedict_to_tensor (local function)

    Parameters
    ----------
    data_iter : torchtext.data.Iterator
        data iterator to go through
    model : torch.nn.Module
        PyTorch model to train
    optimizer : torch.optim.Optimizer
        PyTorch optimizer object to use
    criterion : torch.nn.###Loss
        PyTorch loss function to use
    vocabulary : torchtext.Vocab
        vocabulary object to use
    device : torch.device or int
        device to run the model on
        (default: torch.device('cpu'))
    phase : str, optional
        whether to run a 'train' or 'validation'
        process (default: 'train')
    print_epoch : bool, optional
        whether to produce output in this epoch
        (default: True)
    
    Returns
    -------
    float
        full epoch loss (total loss for all batches
        divided by the number of datapoints)
    '''
    if phase == 'train':
        model.train()
        optimizer.zero_grad()
        grad_ctx_manager = dummy_context_mgr()
    else:
        model.eval()
        grad_ctx_manager = torch.no_grad()
    
    epoch_loss = 0.0
    i = 0
    vocab_size = len(vocabulary) # ALSO input_dim
    
    # HACKISH SOLUTION TO MANUALLY CONSTRUCT THE BATCHES
    # SINCE GOING THROUGH THE ITERATORS DIRECTLY FORCES
    # THE 'numericalize()' FUNCTION ON THE DATA, WHICH
    # WE NUMERICALISED PRIOR TO TRAINING TO SPEED UP
    # PERFORMANCE
    # RESTART BATCHES IN EVERY EPOCH
    # TODO: REMOVE 'numericalize()' FUNCTION TO USE 
    #       ITERATORS DIRECTLY
    data_batches = torchtext.data.batch(data_iter.data(), data_iter.batch_size, data_iter.batch_size_fn)

    start_time = time.time()
    
    # mem_check(device, num=1) # MEM DEBUGGING

    with grad_ctx_manager:
        for batch in data_batches:
            batch_input_list = []
            batch_target = []
            largest_seq = 0
            batch_size = len(batch)

            while len(batch):
                sample = batch.pop()

                batch_input_list.append(treedict_to_tensor(sample.tree, device=device))
                
                proc_seq = [vocabulary.stoi['<sos>']] + sample.seq + [vocabulary.stoi['<eos>']]
                if len(proc_seq) > largest_seq: largest_seq = len(proc_seq)
                batch_target.append(proc_seq)
                i += 1

            # mem_check(device, num=2) # MEM DEBUGGING

            # if there is more than one element in the batch input
            # process the batch with the treelstm.util.batch_tree_input
            # utility function, else return the single element
            if len(batch_input_list) > 1:
                batch_input = batch_tree_input(batch_input_list)
            else:
            #     # PREVIOUS IMPLEMENTATION, USED WITH TREE PREPROCESSING
                batch_input = batch_input_list[0] 
                # batch_input = treedict_to_tensor(sample.tree, device=device)
        
            # mem_check(device, num=3) # MEM DEBUGGING

            for seq in batch_target:
                # PAD THE SEQUENCES IN THE BATCH SO ALL OF THEM
                # HAVE THE SAME LENGTH
                len_diff = largest_seq - len(seq)
                seq.extend([vocabulary.stoi['<pad>']] * len_diff)
        
            # mem_check(device, num=4) # MEM DEBUGGING

            batch_target_tensor = torch.tensor(batch_target, device=device, dtype=torch.long).transpose(0, 1)
            
            if print_epoch and i == 1:
                print_preds = True
            else:
                print_preds = False

            checkpoint_sample = not i % math.ceil(len(data_iter) / 10)
            if print_epoch and checkpoint_sample:
                elapsed_time = time.time() - start_time
                print(f'\nElapsed time after {i} samples: {elapsed_time}', flush=True)
                mem_check(device, num=i) # MEM DEBUGGING
            
            # mem_check(device, num=5) # MEM DEBUGGING

            output = model(batch_input, batch_target_tensor, print_preds=print_preds)
            
            # mem_check(device, num=6) # MEM DEBUGGING
            
            ## seq2seq.py
            # "as the loss function only works on 2d inputs
            # with 1d targets we need to flatten each of them
            # with .view"
            # "we slice off the first column of the output
            # and target tensors (<sos>)"
            # print(f'\n\n ^^^^^^^^^^^^ \t PRE output.size() {output.size()}')
            # TODO: SLICE OFF ALL <sos> TOKENS IN BATCH
            # (REMOVE IXS RELATED TO batch_input['tree_sizes'])
            
            if batch_size == 1:
                output = output.view(-1, vocab_size)[1:]#.view(-1)#, output_dim)
                batch_target_tensor = batch_target_tensor.view(-1)[1:]
            else:
                output = output[1:].view(-1, vocab_size)
                # RESHAPING FUNCTION:
                # 1. REMOVE FIRST ROW OF ELEMENTS (<sos> TOKENS)
                # 2. TRANSPOSE TO GET CONCATENATION OF SEQUENCES
                # 3. FLATTEN INTO A SINGLE DIMENSION (.view(-1) DOES NOT WORK
                #    DUE TO THE TENSOR BEING NON-CONTIGUOUS)
                batch_target_tensor = batch_target_tensor[1:].T.reshape(-1)

            # mem_check(device, num=7) # MEM DEBUGGING

            loss = criterion(output, batch_target_tensor)
            epoch_loss += loss.item()
            
            if phase == 'train':
                loss.backward()
                # mem_check(device, num=8) # MEM DEBUGGING

                # PARTITIONED EMBEDDINGS
                # If a value for embedding partition is defined
                # and the current row is a synonym, zero the
                # gradient for the partition of the embedding
                # if emb_partition and is_syn:
                #     for name, param in model.named_parameters():
                #         param.grad[:,emb_partition:] = 0

                optimizer.step()
                # mem_check(device, num=9) # MEM DEBUGGING
            
            # if n > 10: break
    return epoch_loss / i