####################################
## File: char_encoder.py           ##
## Author: Liu Yang                ##
## Email: liuyeung@stanford.edu    ##
####################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from char_embeddings import CharEmbeddings
from layers import RNNEncoder, BiDAFAttention

class CharQAMultualContext(nn.Module):
      '''
      The layer to compute question aware context representation
      '''
  
      def __init__(self, char_hidden_size, char_vector, drop_prob=0):
         '''
         Ags: 
           @param char_hidden_size (int): the hidden size of the char QA multual context
           @param char_vectors (tensor): the pretrain embeddings of the characters
           @param drop_prob (double): the dropout probabil
         '''
         super(CharQAMultualContext, self).__init__()
         self.char_hidden_size = char_hidden_size
         self.char_emb = CharEmbeddings(char_hidden_size, char_vector, drop_prob) 
         self.rnn = RNNEncoder(input_size=char_hidden_size, hidden_size=char_hidden_size, num_layers=2, drop_prob=drop_prob)
         self.biAtten = BiDAFAttention(hidden_size=2*char_hidden_size, drop_prob=drop_prob)

      def forward(self, cc_idxs, qc_idxs):
         '''
         Ags:
           @param cc_idxs (tensor): index of the context in characters. (batch_size, ctx_len, char_lims)
           @param qc_idxs (tensor): index of the question in characters. (batch_size, q_len, char_lims)
         Return:
           q_ctx (tensor): question aware representation at character level
         '''
         cc_indicator = cc_idxs.sum(-1) #(batch_size, ctx_len)
         qc_indicator = qc_idxs.sum(-1) #(batch_size, q_len)
         cc_mask = torch.zeros_like(cc_indicator) != cc_indicator #get the non-pad word mask in context 
         qc_mask = torch.zeros_like(qc_indicator) != qc_indicator #get the non-pad word mask in query
         cc_len = cc_mask.sum(-1) # (batch_size)
         qc_len = qc_mask.sum(-1) # (batch_size)

         cc_emb = self.char_emb(cc_idxs) #(batch_size, ctx_len, char_hidden_size)
         qc_emb = self.char_emb(qc_idxs) #(batch_size, q_len, char_hidden_size)
         cc_enc = self.rnn(cc_emb, cc_len) #(batch_size, ctx_len, 2*char_hidden_size)
         qc_enc = self.rnn(qc_emb, qc_len) #(batch_size, q_len, 2*char_hidden_size)

         q_ctx = self.biAtten(cc_enc, qc_enc, cc_mask, qc_mask) #(batch_size, ctx_len, 8*char_hidden_size)
         return q_ctx

class ConcatContext(nn.Module):
      '''
       The layer to concat the question aware context representions of word and character levels
      '''

      def __init__(self, w_ctx_dim, c_ctx_dim, reduce_factor=0.5):
         '''
         Args:
           @param: w_ctx_dim (int): dimension of word level context representation
           @param: c_ctx_dim (int): dimension of character level context representation 
         '''
         super(ConcatContext, self).__init__()
         self.input_dim = w_ctx_dim + c_ctx_dim
         self.output_dim = int(self.input_dim*reduce_factor)
         self.dim_reducer = nn.Linear(self.input_dim, self.output_dim)
         nn.init.xavier_uniform_(self.dim_reducer.weight)
         
      def forward(self, w_ctx, c_ctx):
         '''
         Args:
           @param: w_ctx (tensor): word level context. (batch_size, ctx_len, w_ctx_dim)
           @param: c_ctx (tensor): character level context. (batch_size, ctx_len, c_ctx_dim)
         Return:
           ctx
         ''' 
         cat_ctx = torch.cat((w_ctx, c_ctx), 2) #(batch_size, ctx_len, w_ctx_dim + c_ctx_dim)
         ctx = self.dim_reducer(cat_ctx)
         return ctx
