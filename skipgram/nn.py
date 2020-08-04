###
#
# SkipGram Neural Network
#
###

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_word2vec_vectors

class SkipGram(nn.Module):
    """
    SkipGram neural network class
    containing the model initialisation
    and the definition of the forward
    pass with negative sampling as
    described in Mikolov et al., 2013
    
    Requirements
    ------------
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    Attributes
    ----------
    vocab_size : int
        size of the vocabulary
    embed_size : int
        word embedding dimensions
    w2v_init : bool, optional
        whether to initialise the embeddings
        with word2vec vectors (default: True)
    
    Methods
    -------
    forward(focus:List[int], context:List[int],
            neg_indices:ndarray)
        embed the focus and context words into
        two BxD matrices and the negative samples
        into a BxKxD tensor. Use them to calculate
        the loss of the full batch as described in
        Mikolov et al., 2013
    """
    def __init__(self, vocab_size, embedding_size, w2v_init=True, w2v_path=None):
        super(SkipGram, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        
        self.i_embedding = nn.Embedding(vocab_size, embedding_size)
        self.o_embedding = nn.Embedding(vocab_size, embedding_size)
        
        if w2v_init:
            # Initialising weights to word2vec Google News 300 pretrained
            if w2v_path is None:
                self.i_embedding.weight.data.copy_(get_word2vec_vectors())
            else:
                self.i_embedding.weight.data.copy_(get_word2vec_vectors(w2v_path))
        
        # Sanity check: does first vector match
        # the first vector in Word2Vec?
        # print('First word vector')
        # print(self.i_embedding(torch.tensor(0)))
        
        self.logsigmoid = nn.LogSigmoid()
    
    
    def forward(self, focus, context, neg_indices):
        # BxD matrices (B=batch size, D=embedding dimensions)
        embed_focus = self.i_embedding(torch.tensor(focus))
        embed_context = self.o_embedding(torch.tensor(context))
        
        # BxKxD tensor (K=negative samples)
        embed_negs = self.o_embedding(torch.tensor(neg_indices))
        
        # Hadamard product of context and focus
        # embedding matrices followed by a sum
        # across rows (equivalent to stacking
        # vector dot products). Then we get the
        # log of the sigmoid of this vector and
        # finally add it all up
        product = embed_context * embed_focus
        sum_vector = torch.sum(product, dim=1)
        signal_vector = self.logsigmoid(sum_vector)
        
        # Calculating all negative samples together.
        # embed_negs is a tensor made up of B matrices,
        # each of which is made up of K vectors of D
        # dimensions.
        # torch.bmm() performs a 3D tensor multiplication
        # with the same number of matrices (B). For this,
        # we need to split the embed_focus matrix (BxD)
        # into a BxDx1 tensor where every focus embedding
        # is a Dx1 matrix
        # The result is a BxKx1 tensor, which is then
        # squeezed into a BxK matrix
        noise_product = torch.bmm(embed_negs, embed_focus.unsqueeze(2)).squeeze()
        noise_vector = self.logsigmoid(-noise_product)
        noise_sum_vector = torch.sum(noise_vector, dim=1)
        
        loss_vector = signal_vector + noise_sum_vector
        loss = loss_vector.sum()
        
        return -(loss)