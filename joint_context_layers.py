####################################
## File: joint_contex_layers.py    ##
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
         self.ctx_emb_dim = char_hidden_size*8 #the output embedding of the mutual context at character level

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
      
      def __init__(self, input_dim, lstm_hdim, drop_prob=0):
          '''
          Args:
            @param: input_dim (int): input dimension
            @param: lstm_hdim (int): the hidden dimension of the LSTM of location predictor
          '''
          super(AnswerablePredictor, self).__init__()
          self.input_dim = input_dim
          self.lstm_hdim = lstm_hdim
          self.drop_prob = drop_prob
          self.att_weight = nn.Parameter(torch.zeros(input_dim, 1)) #weight to compute attention with respect to each input vector towards answer prediction
          self.h_proj = nn.Linear(input_dim, 2*lstm_hdim) #the project layer to obtain initialization of next bi-LSTM hidden state
          self.c_proj = nn.Linear(input_dim, 2*lstm_hdim) #the project layer to obtain initialization of next bi-LSTM memory state
          self.logits_proj = nn.Linear(input_dim, 2) #the project layer to obtain (no answer, answerable) logits

          for weight in (self.att_weight, self.h_proj.weight, self.c_proj.weight, self.logits_proj.weight):
            nn.init.xavier_uniform_(weight)


      def forward(self, M, G, ctx_mask):
          '''
          Args:
            @param: M (tensor): output of the BiDAF model layer (batch_size, ctx_len, m_dim)
            @param: G (tensor): output of the context condition on question (batch_size, ctx_len, g_dim)
            @param: ctx_mask (tensor): mask of the valid word (batch_size, ctx_len)
          Return:
            logits (tensor): logits of (0,1), 0 for no answer, 1 for answerable.
            (h0, c0) (tensor): initialization state of the start-loc LSTM  
          '''
          ctx = torch.cat((G, M), 2) #(batch_size, ctx_len, self.input_dim)
          att = torch.matmul(ctx, self.att_weight).transpose(1,2) #(batch_size, 1, ctx_len)
          att_prob = util.masked_softmax(att, ctx_mask.unsqueeze(1)) #(batch_size, 1, ctx_len)
          ctx_summary = torch.bmm(att_prob, ctx).squeeze(1) #(batch_size, self.input_dim)
          h0 = self.h_proj(ctx_summary) #(batch_size, 2*self.lstm_hdim)
          h0 = h0.view(-1, 2, self.lstm_hdim).transpose(0,1) #(2, batch_size, self.lstm_hdim)
          c0 = self.c_proj(ctx_summary) #(batch_size, 2*self.lstm_hdim)
          c0 = c0.view(-1, 2, self.lstm_hdim).transpose(0,1) #(2, batch_size, self.lstm_hdim)

          logits = self.logits_proj(ctx_summary) #(batch_size, self.lstm_hdim)
          h0 = F.dropout(h0, self.drop_prob, self.training).contiguous()
          c0 = F.dropout(c0, self.drop_prob, self.training).contiguous()
          
          return logits, (h0, c0)

class StartLocationPredictor(nn.Module):
      '''
      This layer compute the conditional distribution P(start_loc | Answerable).
      '''

      def __init__(self, M_dim, G_dim, lstm_hdim, drop_prob=0):
          '''
          Args:
            @param: input_dim (int): input dimension
            @param: lstm_hdim (int): the hidden dimension of the LSTM
          '''
          super(StartLocationPredictor, self).__init__()
          self.M_dim = M_dim
          self.G_dim = G_dim
          self.lstm_hdim = lstm_hdim
          self.cat_dim = G_dim + 2*lstm_hdim
          self.rnn = RNNEncoder(M_dim, lstm_hdim, 1, drop_prob)
          self.loc_proj = nn.Linear(self.cat_dim, 1)
          self.h_proj = nn.Linear(self.cat_dim, 2*lstm_hdim)
          self.c_proj = nn.Linear(self.cat_dim, 2*lstm_hdim)
          self.drop_prob=drop_prob

          for weight in (self.loc_proj.weight, self.h_proj.weight, self.c_proj.weight):
            nn.init.xavier_uniform_(weight)


      def forward(self, M, G, ctx_mask, enc_init):
          '''
          Args:
            @param: M (tensor): output of the BiDAF model layer (batch_size, ctx_len, M_dim)
            @param: G (tensor): output of the context condition on question (batch_size, ctx_len, G_dim)
            @param: ctx_mask (tensor): mask of the valid word (batch_size, ctx_len)
            @param: enc_init (tensor): initiation state of the interal LSTM (2, batch_size, self.lstm_hdim)
          Rets:
            p_start (tensor): probability distribution of the start location (batch_size, ctx_len)
            (h0, c0) (tensor): initialization state of the end-loc LSTM
          '''
          lengths = ctx_mask.sum(-1) #(batch_size)
          M1 = self.rnn(M, lengths, enc_init) #M2: (batch_size, ctx_len, 2*self.lstm_hdim)
          ctx = torch.cat((G, M1), 2) #(batch_size, ctx_len, self.cat_dim)
          logits = self.loc_proj(ctx).squeeze(-1) #(batch_size, ctx_len)
          p_start = util.masked_softmax(logits, ctx_mask, log_softmax=True) #(batch_size, ctx_len)
          
          #use probability as attention explaining the start prediction
          att = p_start.unsqueeze(1) #(batch_size, 1, ctx_len)
          #use attention to obtain a summary of the context in respect to start prediction
          ctx_summary = torch.bmm(att, ctx).squeeze(1) #(batch_size, self.cat_dim)

          h0 = self.h_proj(ctx_summary) #(batch_size, 2*self.lstm_hdim)
          h0 = h0.view(-1, 2, self.lstm_hdim).transpose(0,1) #(2, batch_size, self.lstm_hdim)
          c0 = self.c_proj(ctx_summary) #(batch_size, 2*self.lstm_hdim)
          c0 = c0.view(-1, 2, self.lstm_hdim).transpose(0,1) #(2, batch_size, self.lstm_hdim)

          h0 = F.dropout(h0, self.drop_prob, self.training).contiguous()
          c0 = F.dropout(c0, self.drop_prob, self.training).contiguous()
         
          return p_start, (h0, c0)

class EndLocationPredictor(nn.Module):
      '''
      This layer compute the conditional distribution P(end_loc | Answerable, start_loc)
      '''
      
      def __init__(self, M_dim, G_dim, lstm_hdim, drop_prob=0.):
          '''
          Args:
            @param: input_dim (int): input dimension
            @param: lstm_hdim (int): the hidden dimension of the LSTM
          '''
          super(EndLocationPredictor, self).__init__()
          self.M_dim = M_dim
          self.G_dim = G_dim
          self.lstm_hdim = lstm_hdim
          self.drop_prob = drop_prob
          self.cat_dim = G_dim + 2*lstm_hdim
          self.rnn = RNNEncoder(M_dim, lstm_hdim, 1, drop_prob) 
          self.h_proj = nn.Linear(2*lstm_hdim, lstm_hdim)
          self.c_proj = nn.Linear(2*lstm_hdim, lstm_hdim)
          self.loc_proj = nn.Linear(self.cat_dim, 1)
          
          for weight in (self.h_proj.weight, self.c_proj.weight):
              nn.init.xavier_uniform_(weight)


      def forward(self, M, G, ctx_mask, enc_init_A, enc_init_start):
          '''
          Args:
            @param: M (tensor): output of the BiDAF model layer (batch_size, ctx_len, M_dim)
            @param: G (tensor): output of the context condition on question (batch_size, ctx_len, G_dim)
            @param: ctx_mask (tensor): mask of the valid word (batch_size, ctx_len)
            @param: enc_init_A (tensor): initialtion state computed by Answerable predictor (2, batch_size, self.lstm_hdim)
            @param: enc_init_start (tensor): initialtion state computed by start position predictor (2, batch_size, self.lstm_hdim)
          Rets:
            p_end (tensor): probability distribution of the end location (batch_size, ctx_len)
          '''
          lengths = ctx_mask.sum(-1) #(batch_size)
          (hA, cA) = enc_init_A
          (hs, cs) = enc_init_start
          h0 = self.h_proj(torch.cat((hA, hs), 2)) #(2, batch_size, self.lstm_hdim)
          c0 = self.h_proj(torch.cat((cA, cs), 2)) #(2, batch_size, self.lstm_hdim)
          M2 = self.rnn(M, lengths, (h0, c0)) #(batch_size, ctx_len, 2*lstm_hdim)
          ctx = torch.cat((G, M2), 2) #(batch_size, ctx_len, self.cat_dim)
          logits = self.loc_proj(ctx).squeeze(-1) #(batch_size, ctx_len)
          p_end = util.masked_softmax(logits, ctx_mask, log_softmax=True)#(batch_size, ctx_len)

          return p_end
          
          
















