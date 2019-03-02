####################################
## File: char_encoder.py           ##
## Author: Liu Yang                ##
## Email: liuyeung@stanford.edu    ##
####################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import util
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

class AnswerablePredictor(nn.Module):
      '''
      This layer predict whether the Qestion is answerable
      '''
      
      def __init__(self, input_dim, lstm_hdim):
          '''
          Args:
            @param: input_dim (int): input dimension
            @param: lstm_hdim (int): the hidden dimension of the LSTM of location predictor
          '''
          super(AnswerablePredictor, self).__init__()
          self.input_dim = input_dim
          self.lstm_hdim = lstm_hdim
          self.att_weight = nn.Parameter(torch.zeros(input_dim, 1)) #weight to compute attention with respect to each input vector towards answer prediction
          self.h_proj = nn.Linear(input_dim, lstm_hdim) #the project layer to obtain initialization of next LSTM hidden state
          self.c_proj = nn.Linear(input_dim, lstm_hdim) #the project layer to obtain initialization of next LSTM memory state
          self.logits_proj = nn.Linear(input_dim, 2) #the project layer to obtain (no answer, answerable) logits

          for weight in (self.att_weight, self.h_proj.weight, self.c_proj.weight, self.logits_proj.weight):
            nn.init.xavier_uniform_(weight)


      def forward(self, M, G, ctx_mask):
          '''
          Args:
            @param: M (tensor): the output of the BiDAF model layer (batch_size, ctx_len, m_dim)
            @param: G (tensor): the output of the context condition on question (batch_size, ctx_len, g_dim)
            @param: ctx_mask (tensor): the mask of valid word (batch_size, ctx_len)
          Return:
            logits (tensor): logits of (0,1), 0 for no answer, 1 for answerable.
            (h0, c0) (tensor): initialization state of the start-loc LSTM  
          '''
          ctx = torch.cat((M, G), 2) #(batch_size, ctx_len, self.input_dim)
          att = torch.matmul(ctx, self.att_weight).transpose(1,2) #(batch_size, 1, ctx_len)
          att_prob = util.masked_softmax(att, ctx_mask.unsqueeze(1)) #(batch_size, 1, ctx_len)
          ctx_summary = torch.bmm(att_prob, ctx).squeeze(1) #(batch_size, self.input_dim)
          h0 = self.h_proj(ctx_summary) #(batch_size, self.lstm_hdim)
          c0 = self.c_proj(ctx_summary) #(batch_size, self.lstm_hdim)
          logits = self.logits_proj(ctx_summary) #(batch_size, self.lstm_hdim)
          
          return (h0, c0), logits

