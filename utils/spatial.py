###
#
# Embedding space analysis
# Support functions
#
###

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_distances

def pairwise_dists(embs_file, dists_save_file):
    """
    PERF. NOTE: Takes roughly 2mins on the server
    and 25% of memory usage (out of 32GB of
    available RAM) with the following parameters:
    
    Embeddings shape:  (45069, 300)
    
    Real elapsed time:  103.086623845
    """
    embs = np.load(embs_file)
    print('Embeddings shape: ', embs.shape)
    
    cos_dists = cosine_distances(embs)
    
    print('Saving distances to ', dists_save_file)
    np.save(dists_save_file, cos_dists)


def calculate_singular_vals(matrix_file, save_file):
    print('Loading matrix from ', matrix_file)
    
    A_tens = torch.tensor(np.load(matrix_file, mmap_mode='r'))
    
    print('Performing SVD...')
    
    # Slower implementation from Numpy
    # U, S, V_T = np.linalg.svd(A)
    
    _,S,_ = torch.svd(A_tens)
    
    np.save(save_file, S)
    

def explained_variance(S, num_s):
    '''
    Given a list of singular values, calculate
    the explained variance of each of them as
        S^2/sum(S^2)
    
    Reuirements
    -----------
    import numpy as np
    
    Parameters
    ----------
    S : list[float]
        list of singular values
    num_s : int
        number of singular values to consider
        
    Returns
    -------
    list[float]
        list of 
    '''
    S_2 = S**2
    S_2_sum = np.sum(S_2)
    return S_2[:num_s] / S_2_sum


def plot_exp_variance(S, plot=True):
    sing_vals = range(1, len(S)+1)
    exp_vars = explained_variance(S, len(S))
    cum_exp_vars = [np.sum(exp_vars[:i]) for i in range(1, len(S)+1)]
    
    print('Exp vars: ', exp_vars)
    print('Cumulative exp vars: ', cum_exp_vars)
    
    plt.plot(sing_vals, cum_exp_vars)
    
    if plot: plt.show()


def get_embedding_norms(embs_file, save_file, norm=2):
    embs = np.load(embs_file)
    norms = np.linalg.norm(embs, ord=norm, axis=1)
    
    np.save(save_file, norms)


