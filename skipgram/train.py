import torch
import torch.optim as optim

import time
from time import gmtime, strftime
import random
import numpy as np
import csv
import sys

from .nn import SkipGram
from .utils import init_sample_table, new_neg_sampling, save_param_to_npy


def train_augm_w2v(data_file, vocab_file, syns_file, model_file, checkpoints_folder, validation_file, embedding_size=300, epochs=10, batch_size=10, num_neg_samples=5, learning_rate=0.01, w2v_init=True, w2v_path=None, syn_augm=True, emb_npy_file=None, data_augmentation_ratio=.25):
    """
    Augmented dataset Word2Vec training regime.
    Randomly cycle through the data and synonym
    files, so augmented examples are interleaved
    with natural ones. Processes the training
    data in batches to speed up performance, and
    backpropagates and optimises (SGD) after each
    batch. Reports the loss after each epoch
    (divided by the number of datapoints processed,
    which is important since this number varies
    between epochs). Finally, freezes the gradients
    and runs inference on the validation set to
    report a validation score. Also reports the
    training times.
    
    Requirements
    ------------
    import torch
    import torch.optim as optim
    import nn.SkipGram as SkipGram
    import utils.init_sample_table as init_sample_table
    import utils.new_neg_sampling as new_neg_sampling
    
    Parameters
    ----------
    data_file : str
        filepath to the lightweight natural
        dataset
    vocab_file : str
        filepath to the vocabulary file
    syns_file : str
        filepath to the lightweight augmented
        dataset
    model_file : str
        path to save the final model to
    checkpoints_folder : str
        path to save the checkpoints to
        after each epoch
    validation_file : str
        filepath to the lightweight validation
        dataset
    embedding_size : int, optional
        embedding dimensions (default: 300)
    epochs : int, optional
        number of epochs to train for
        (default: 10)
    batch_size : int, optional
        size of the training batches (default: 10)
    num_neg_samples : int, optional
        number of negative datapoints to sample
        (default: 5)
    learning_rate : float, optional
        learning rate for the selected optimisation
        (default: 0.01)
    w2v_init : bool, optional
        whether to initialise the embeddings
        with word2vec vectors (default: True)
    syn_augm : bool, optional
        whether to augment the training data
        with synonym information (default: True)
    emb_npy_file : str, optional
        if defined, path to save NPY input
        embedding to (default: None)
    data_augmentation_ratio : float, optional
        proportion of augmented data to use during
        training (default: .25)
    """
    
    # Generating random numbers:
    # IMPORTANT: keep track of the random seed
    # of all experiments that are run
    RANDOM_SEED = 1
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True # This option impacts speed

    if torch.cuda.is_available(): print('CUDA is available, running on GPU')

    # If CUDA is available, default all tensors to CUDA tensors
    torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)
    
    print('##### PARAMETERS \n')
    print('data_file', data_file)
    print('vocab_file', vocab_file)
    print('syns_file', syns_file)
    print('model_file', model_file)
    print('checkpoints_folder', checkpoints_folder)
    print('validation_file', validation_file)
    print('embedding_size', embedding_size)
    print('epochs', epochs)
    print('batch_size', batch_size)
    print('num_neg_samples', num_neg_samples)
    print('learning_rate', learning_rate)
    print('w2v_init', ('True' if w2v_init else 'False'))
    print('w2v_path', (w2v_path if w2v_path else 'None'))
    print('syn_augm', (syn_augm if syn_augm else 'None'))
    print('emb_npy_file', (emb_npy_file if emb_npy_file else 'None'))
    print('data_augmentation_ratio', data_augmentation_ratio)
    print('\n\n')
    
    with open(data_file, 'r', encoding='utf-8', errors='replace') as d, \
        open(vocab_file, 'r', encoding='utf-8', errors='replace') as v, \
        open(syns_file, 'r', encoding='utf-8', errors='replace') as s, \
        open(validation_file, 'r', encoding='utf-8', errors='replace') as val:
        
        start_time = time.time()
        
        data = csv.reader(d)
        vocab_reader = csv.reader(v)
        syns = csv.reader(s)
        
        validation = [w for w in csv.reader(val)]
        
        vocabulary = [w for w in vocab_reader]
        vocab_words = [w[0] for w in vocabulary]
        vocab_counts = [int(w[1]) for w in vocabulary]
        
        # Calculate the vocabulary ratios
        # Elevate counts to the 3/4th power
        pow_counts = np.array(vocab_counts)**0.75
        normaliser = sum(pow_counts)
        # Normalise the counts
        vocab_ratios = pow_counts / normaliser
        
        sample_table = init_sample_table(vocab_counts)
        
        print('Size of sample table: ', sample_table.size)
        
        print('Total distinct words: ', len(vocabulary))
        print('Samples from vocab: ', vocabulary[:5])
        
        model = SkipGram(len(vocabulary), embedding_size, w2v_init=w2v_init, w2v_path=w2v_path)
        
        if torch.cuda.is_available():
            model.cuda()
            
        optimiser = optim.SGD(model.parameters(),lr=learning_rate)
        
        losses = []
        val_losses = []
        times = []
        
        # RUN VALIDATION BEFORE ANY TRAINING
        # Validation
        with torch.no_grad():
            print('Epoch 0 validation')
            print(len(validation))
            
            focus_ixs = []
            context_ixs = []
            neg_ixs = []
            num_points = 0
            epoch_loss = 0.
            
            for row in validation:
                focus_i = int(row[0])
                context_i = int(row[1])
                focus_ixs.append(focus_i)
                context_ixs.append(context_i)
                
                num_points += 1
                
                if num_points % batch_size == 0 \
                    and num_points != 1:
                    # Restart gradients
                    optimiser.zero_grad()
                    
                    neg_samples = new_neg_sampling(sample_table, num_samples=num_neg_samples, batch_size=batch_size)
                    
                    loss = model(focus_ixs, context_ixs, neg_samples)
                    
                    epoch_loss += loss.item() / num_points
                    
                    focus_ixs = []
                    context_ixs = []
            
            print('\n********\nValidation loss at epoch %d: %.2f' % (0, epoch_loss))
            print('(num points:', num_points, ')')
            
            val_losses.append(epoch_loss)
        
        if syn_augm:
            syns_data = [row for row in syns]
            num_syns = len(syns_data)
            print("Calculating synonym indices. Number of synonyms:", num_syns)
            syns_ixs = np.random.choice(num_syns, num_syns, replace=False)
            syns_ixs = np.append(syns_ixs, np.random.choice(num_syns, num_syns, replace=False))
            syns_ixs = np.append(syns_ixs, np.random.choice(num_syns, num_syns, replace=False))
            print('Finished calculating. Number of random synonym indices:', len(syns_ixs))
            
            # print('Syns data:', len(syns_data))
            # print('Syns datapoint 1:', syns_data[0])
        
        for epoch in range(epochs):
            start_time = time.time()
            # Reset the reader position and skip
            # the header file
            d.seek(0)
            s.seek(0)
            
            i_break = 0
            epoch_loss = 0.
            num_points = 0
            syn_index = 0
            
            # Rough ratio of "natural" vs. augmented examples
            # Current dataset sizes:
            #   1.5G dataset.csv
            #   473M synonyms_vocab3_sw.csv
            data_ratio = 1 - data_augmentation_ratio #.75
            
            # Initial value for row, to emulate a do-while loop
            row = True
            
            focus_ixs = []
            context_ixs = []
            neg_ixs = []
            
            # With the next(_, False) function, row becomes
            # false when there is no more data to read (only
            # true when there is no more data in either dataset)
            while row:
                if syn_augm:
                    # Randomly select either a "natural" or an
                    # augmented example to process
                    if random.random() > data_ratio:
                        # While the synonym index is smaller
                        # than the number of synonyms use the
                        # first random indices array, otherwise
                        # use the second array. If it is greater
                        # than twice the number of synonyms exit
                        if syn_index < len(syns_ixs):
                            syn_ix = syns_ixs[syn_index]
                            syn_index += 1
                            row = syns_data[syn_ix]
                        else:
                            print('Out of synonym indices')
                            row = False
                        
                        # row = next(syns, False)
                    else:
                        row = next(data, False)
                else:
                    row = next(data, False)
                if row:
                    focus_i = int(row[0])
                    context_i = int(row[1])
                    focus_ixs.append(focus_i)
                    context_ixs.append(context_i)
                    
                    num_points += 1
                    
                    if num_points % batch_size == 0 \
                        and num_points != 1:
                        # Restart gradients
                        optimiser.zero_grad()
                        
                        neg_samples = new_neg_sampling(sample_table, num_samples=num_neg_samples, batch_size=batch_size)
                        
                        loss = model(focus_ixs, context_ixs, neg_samples)
                        
                        epoch_loss += loss.item() / num_points
                        loss.backward()
                        optimiser.step()
                        
                        focus_ixs = []
                        context_ixs = []
                        
            
            print('\n********\nLoss at epoch %d: %.2f' % (epoch, epoch_loss))
            print('(num points:', num_points, ')')
            print('(num synonyms:', syn_index, ')')
            
            losses.append(epoch_loss)
            
            elapsed_time = time.time() - start_time
            times.append(elapsed_time)
            print('Elapsed time in epoch %d: %r' % (epoch, elapsed_time))
            
            sys.stdout.flush()
            
            # SAVING A CHECKPOINT
            if not checkpoints_folder.endswith('/'): checkpoints_folder += '/'
            checkpoints_file = checkpoints_folder + str(epoch) + '-epoch-chkpt.tar'
            print('Saving checkpoint file: %r \n' % (checkpoints_file))
            
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimiser.state_dict(),
                    'loss': epoch_loss
                    }, checkpoints_file)
            
            # Validation
            with torch.no_grad():
                focus_ixs = []
                context_ixs = []
                neg_ixs = []
                num_points = 0
                epoch_loss = 0.
                
                for row in validation:
                    focus_i = int(row[0])
                    context_i = int(row[1])
                    focus_ixs.append(focus_i)
                    context_ixs.append(context_i)
                    
                    num_points += 1
                    
                    if num_points % batch_size == 0 \
                        and num_points != 1:
                        # Restart gradients
                        optimiser.zero_grad()
                        
                        neg_samples = new_neg_sampling(sample_table, num_samples=num_neg_samples, batch_size=batch_size)
                        
                        loss = model(focus_ixs, context_ixs, neg_samples)
                        
                        epoch_loss += loss.item() / num_points
                        
                        focus_ixs = []
                        context_ixs = []
                
                print('\n********\nValidation loss at epoch %d: %.2f' % (epoch, epoch_loss))
                print('(num points:', num_points, ')')
                
                val_losses.append(epoch_loss)
            
    print('\n\nSaving model to ', model_file)
    # A common PyTorch convention is to save models using
    # either a .pt or .pth file extension.
    torch.save(model.state_dict(), model_file)
    #model.load_state_dict(torch.load(model_file))
    
    if emb_npy_file is not None:
        # Save input embeddings to a NPY file
        param_name = 'i_embedding'
        save_param_to_npy(model, param_name, emb_npy_file)
    
    avg_time = np.mean(times)
    
    print('Train losses:')
    print(losses)
    
    print('Validation losses:')
    print(val_losses)
    
    print('Average run time per epoch: ', avg_time)



''' # OLD TRAIN FUNCTION
def train_augm_w2v(data_file, vocab_file, syns_file, model_file, checkpoints_folder, validation_file, embedding_size=300, epochs=10, batch_size=10, num_neg_samples=5, learning_rate=0.01, w2v_init=True, w2v_path=None, syn_augm=True, emb_npy_file=None, data_augmentation_ratio=.25):
    """
    Augmented dataset Word2Vec training regime.
    Randomly cycle through the data and synonym
    files, so augmented examples are interleaved
    with natural ones. Processes the training
    data in batches to speed up performance, and
    backpropagates and optimises (SGD) after each
    batch. Reports the loss after each epoch
    (divided by the number of datapoints processed,
    which is important since this number varies
    between epochs). Finally, freezes the gradients
    and runs inference on the validation set to
    report a validation score. Also reports the
    training times.
    
    Requirements
    ------------
    import torch
    import torch.optim as optim
    import nn.SkipGram as SkipGram
    import utils.init_sample_table as init_sample_table
    import utils.new_neg_sampling as new_neg_sampling
    
    Parameters
    ----------
    data_file : str
        filepath to the lightweight natural
        dataset
    vocab_file : str
        filepath to the vocabulary file
    syns_file : str
        filepath to the lightweight augmented
        dataset
    model_file : str
        path to save the final model to
    checkpoints_folder : str
        path to save the checkpoints to
        after each epoch
    validation_file : str
        filepath to the lightweight validation
        dataset
    embedding_size : int, optional
        embedding dimensions (default: 300)
    epochs : int, optional
        number of epochs to train for
        (default: 10)
    batch_size : int, optional
        size of the training batches (default: 10)
    num_neg_samples : int, optional
        number of negative datapoints to sample
        (default: 5)
    learning_rate : float, optional
        learning rate for the selected optimisation
        (default: 0.01)
    w2v_init : bool, optional
        whether to initialise the embeddings
        with word2vec vectors (default: True)
    syn_augm : bool, optional
        whether to augment the training data
        with synonym information (default: True)
    emb_npy_file : str, optional
        if defined, path to save NPY input
        embedding to (default: None)
    data_augmentation_ratio : float, optional
        proportion of augmented data to use during
        training (default: .25)
    """
    
    # Generating random numbers:
    # IMPORTANT: keep track of the random seed
    # of all experiments that are run
    RANDOM_SEED = 1
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True # This option impacts speed

    if torch.cuda.is_available(): print('CUDA is available, running on GPU')

    # If CUDA is available, default all tensors to CUDA tensors
    torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)
    
    print('##### PARAMETERS \n')
    print('data_file', data_file)
    print('vocab_file', vocab_file)
    print('syns_file', syns_file)
    print('model_file', model_file)
    print('checkpoints_folder', checkpoints_folder)
    print('validation_file', validation_file)
    print('embedding_size', embedding_size)
    print('epochs', epochs)
    print('batch_size', batch_size)
    print('num_neg_samples', num_neg_samples)
    print('learning_rate', learning_rate)
    print('w2v_init', ('True' if w2v_init else 'False'))
    print('w2v_path', (w2v_path if w2v_path else 'None'))
    print('syn_augm', (syn_augm if syn_augm else 'None'))
    print('emb_npy_file', (emb_npy_file if emb_npy_file else 'None'))
    print('data_augmentation_ratio', data_augmentation_ratio)
    print('\n\n')
    
    with open(data_file, 'r', encoding='utf-8', errors='replace') as d, \
        open(vocab_file, 'r', encoding='utf-8', errors='replace') as v, \
        open(syns_file, 'r', encoding='utf-8', errors='replace') as s, \
        open(validation_file, 'r', encoding='utf-8', errors='replace') as val:
        
        start_time = time.time()
        
        data = csv.reader(d)
        vocab_reader = csv.reader(v)
        syns = csv.reader(s)
        
        validation = [w for w in csv.reader(val)]
        
        vocabulary = [w for w in vocab_reader]
        vocab_words = [w[0] for w in vocabulary]
        vocab_counts = [int(w[1]) for w in vocabulary]
        
        # Calculate the vocabulary ratios
        # Elevate counts to the 3/4th power
        pow_counts = np.array(vocab_counts)**0.75
        normaliser = sum(pow_counts)
        # Normalise the counts
        vocab_ratios = pow_counts / normaliser
        
        sample_table = init_sample_table(vocab_counts)
        
        print('Size of sample table: ', sample_table.size)
        
        print('Total distinct words: ', len(vocabulary))
        print('Samples from vocab: ', vocabulary[:5])
        
        model = SkipGram(len(vocabulary), embedding_size, w2v_init=w2v_init, w2v_path=w2v_path)
        
        if torch.cuda.is_available():
            model.cuda()
            
        optimiser = optim.SGD(model.parameters(),lr=learning_rate)
        
        losses = []
        val_losses = []
        times = []
        
        # RUN VALIDATION BEFORE ANY TRAINING
        # Validation
        with torch.no_grad():
            print('Epoch 0 validation')
            print(len(validation))
            
            focus_ixs = []
            context_ixs = []
            neg_ixs = []
            num_points = 0
            epoch_loss = 0.
            
            for row in validation:
                focus_i = int(row[0])
                context_i = int(row[1])
                focus_ixs.append(focus_i)
                context_ixs.append(context_i)
                
                num_points += 1
                
                if num_points % batch_size == 0 \
                    and num_points != 1:
                    # Restart gradients
                    optimiser.zero_grad()
                    
                    neg_samples = new_neg_sampling(sample_table, num_samples=num_neg_samples, batch_size=batch_size)
                    
                    loss = model(focus_ixs, context_ixs, neg_samples)
                    
                    epoch_loss += loss.item() / num_points
                    
                    focus_ixs = []
                    context_ixs = []
            
            print('\n********\nValidation loss at epoch %d: %.2f' % (0, epoch_loss))
            print('(num points:', num_points, ')')
            
            val_losses.append(epoch_loss)
        
        if syn_augm:
            syns_data = [row for row in syns]
            num_syns = len(syns_data)
            print("Calculating synonym indices. Number of synonyms:", num_syns)
            syns_ixs = np.random.choice(num_syns, num_syns, replace=False)
            syns_ixs = np.append(syns_ixs, np.random.choice(num_syns, num_syns, replace=False))
            syns_ixs = np.append(syns_ixs, np.random.choice(num_syns, num_syns, replace=False))
            print('Finished calculating. Number of random synonym indices:', len(syns_ixs))
            
            # print('Syns data:', len(syns_data))
            # print('Syns datapoint 1:', syns_data[0])
        
        for epoch in range(epochs):
            start_time = time.time()
            # Reset the reader position and skip
            # the header file
            d.seek(0)
            s.seek(0)
            
            i_break = 0
            epoch_loss = 0.
            num_points = 0
            syn_index = 0
            
            # Rough ratio of "natural" vs. augmented examples
            # Current dataset sizes:
            #   1.5G dataset.csv
            #   473M synonyms_vocab3_sw.csv
            data_ratio = 1 - data_augmentation_ratio #.75
            
            # Initial value for row, to emulate a do-while loop
            row = True
            
            focus_ixs = []
            context_ixs = []
            neg_ixs = []
            
            # With the next(_, False) function, row becomes
            # false when there is no more data to read (only
            # true when there is no more data in either dataset)
            while row:
                if syn_augm:
                    # Randomly select either a "natural" or an
                    # augmented example to process
                    if random.random() > data_ratio:
                        # While the synonym index is smaller
                        # than the number of synonyms use the
                        # first random indices array, otherwise
                        # use the second array. If it is greater
                        # than twice the number of synonyms exit
                        if syn_index < len(syns_ixs):
                            syn_ix = syns_ixs[syn_index]
                            syn_index += 1
                            row = syns_data[syn_ix]
                        else:
                            print('Out of synonym indices')
                            row = False
                        
                        # row = next(syns, False)
                    else:
                        row = next(data, False)
                else:
                    row = next(data, False)
                if row:
                    focus_i = int(row[0])
                    context_i = int(row[1])
                    focus_ixs.append(focus_i)
                    context_ixs.append(context_i)
                    
                    num_points += 1
                    
                    if num_points % batch_size == 0 \
                        and num_points != 1:
                        # Restart gradients
                        optimiser.zero_grad()
                        
                        neg_samples = new_neg_sampling(sample_table, num_samples=num_neg_samples, batch_size=batch_size)
                        
                        loss = model(focus_ixs, context_ixs, neg_samples)
                        
                        epoch_loss += loss.item() / num_points
                        loss.backward()
                        optimiser.step()
                        
                        focus_ixs = []
                        context_ixs = []
                        
            
            print('\n********\nLoss at epoch %d: %.2f' % (epoch, epoch_loss))
            print('(num points:', num_points, ')')
            print('(num synonyms:', syn_index, ')')
            
            losses.append(epoch_loss)
            
            elapsed_time = time.time() - start_time
            times.append(elapsed_time)
            print('Elapsed time in epoch %d: %r' % (epoch, elapsed_time))
            
            sys.stdout.flush()
            
            # SAVING A CHECKPOINT
            if not checkpoints_folder.endswith('/'): checkpoints_folder += '/'
            checkpoints_file = checkpoints_folder + str(epoch) + '-epoch-chkpt.tar'
            print('Saving checkpoint file: %r \n' % (checkpoints_file))
            
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimiser.state_dict(),
                    'loss': epoch_loss
                    }, checkpoints_file)
            
            # Validation
            with torch.no_grad():
                focus_ixs = []
                context_ixs = []
                neg_ixs = []
                num_points = 0
                epoch_loss = 0.
                
                for row in validation:
                    focus_i = int(row[0])
                    context_i = int(row[1])
                    focus_ixs.append(focus_i)
                    context_ixs.append(context_i)
                    
                    num_points += 1
                    
                    if num_points % batch_size == 0 \
                        and num_points != 1:
                        # Restart gradients
                        optimiser.zero_grad()
                        
                        neg_samples = new_neg_sampling(sample_table, num_samples=num_neg_samples, batch_size=batch_size)
                        
                        loss = model(focus_ixs, context_ixs, neg_samples)
                        
                        epoch_loss += loss.item() / num_points
                        
                        focus_ixs = []
                        context_ixs = []
                
                print('\n********\nValidation loss at epoch %d: %.2f' % (epoch, epoch_loss))
                print('(num points:', num_points, ')')
                
                val_losses.append(epoch_loss)
            
    print('\n\nSaving model to ', model_file)
    # A common PyTorch convention is to save models using
    # either a .pt or .pth file extension.
    torch.save(model.state_dict(), model_file)
    #model.load_state_dict(torch.load(model_file))
    
    if emb_npy_file is not None:
        # Save input embeddings to a NPY file
        param_name = 'i_embedding'
        save_param_to_npy(model, param_name, emb_npy_file)
    
    avg_time = np.mean(times)
    
    print('Train losses:')
    print(losses)
    
    print('Validation losses:')
    print(val_losses)
    
    print('Average run time per epoch: ', avg_time)
'''