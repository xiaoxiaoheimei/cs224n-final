####################################
## File: bert_word_embeddings.py   ##
## Author: Liu Yang                ##
## Email: liuyeung@stanford.edu    ##
####################################

import torch as t
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import modeling, BertTokenizer
import pdb

class BertWordEmbedding(nn.Module):
      '''
      The layer of generate Bert contextual representation
      '''

      def __init__(self, config, device=t.device("cpu")):
          '''
          Args:
            @config: configuration file of internal Bert layer
          '''
          super(BertWordEmbedding, self).__init__()
          self.bert = modeling.BertModel(config=config).from_pretrained('bert-base-uncased')
          self.device = device
          #self.bert.to('cuda')
          tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
          self.CLS = tokenizer.convert_tokens_to_ids(['[CLS]'])[0] #ID of the Bert [CLS] token
          self.SEP = tokenizer.convert_tokens_to_ids(['[SEP]'])[0] #ID of the Bert [SEP] token

      def make_bert_input(self, ques_idxs, ctx_idxs, ques_mask, ctx_mask):
          '''
          Args:
            @ques_idxs (tensor): Bert IDs of the question. (batch_size, q_len) Note that the q_len is a fixed number due to padding/clamping policy
            @ctx_idxs (tensor): Bert IDs of the context. (batch_size, ctx_len) Note that the ctx_len is a fixed number due to padding/clamping policy
            @ques_mask (tensor): elementwise 0,1 tensor to real token in question. (batch_size, q_len)
            @ctx_mask (tensor): elementwise 0,1 tensor to real token in context. (batch_size, ctx_len)
          Return:
            qa_idx (tensor): [CLS] question [SEP] context [SEP]. (batch_size, q_len + ctx_len + 3)
            segment_ids (tensor): 0, 1 identifiers ot the two question/context sentences respectively. (batch_size, q_len + ctx_len + 3)
            bert_att (tensor): attention of the real word in question/context sentences respectively. [0] ques_mask [0] ctx_mask [0]. (batch_size, q_len + ctx_len + 3)
          '''
          batch_size, q_len = ques_idxs.size()
          _, ctx_len = ctx_idxs.size()
          pad_CLS = t.tensor([[self.CLS]]*batch_size, dtype=ques_idxs.dtype, device=self.device) #(batch_size, 1)
          pad_SEP = t.tensor([[self.SEP]]*batch_size, dtype=ques_idxs.dtype, device=self.device) #(batch_size, 1)
          ques_pad_CLS = t.cat((pad_CLS, ques_idxs), dim=-1) #[CLS] ques_idxs : (batch_size, q_len+1)
          ques_pad = t.cat((ques_pad_CLS, pad_SEP), dim=-1) #[CLS] ques_idxs [SEP] : (batch_size, q_len+2)
          ctx_pad = t.cat((ctx_idxs, pad_SEP), dim=-1) # ctx_idxs [SEP] : (batch_size, ctx_len+1)
          qa_idxs = t.cat((ques_pad, ctx_pad), dim=-1) #[CLS] ques_idxs [SEP] ctx_idxs [SEP] : (batch_size, q_len + ctx_len + 3)
          segment_ids = t.cat((t.zeros_like(ques_pad), t.ones_like(ctx_pad)), dim=-1) # (batch_size, ctx_len + q_len + 3)

          att_pad = t.tensor([[0]]*batch_size, dtype=ques_mask.dtype, device=self.device) # (batch_size, 1)
          qmask_pad = t.cat((t.cat((att_pad, ques_mask), dim=-1), att_pad), dim=-1) # (batch_size, q_len+2)
          cxt_mask_pad = t.cat((ctx_mask, att_pad), dim=-1) # (batch_size, ctx_len+1)
          bert_att = t.cat((qmask_pad, cxt_mask_pad), dim=-1) # [0, ques_mask, 0, ctx_mask, 0]: (batch_size, ctx_len+q_len+3)

          return qa_idxs, segment_ids, bert_att

      def parse_context_rep(self, qa_encoder_rep, ctx_len):
          '''
          parse the context from the last layer encoder of the Bert.
          Args:
            @qa_encoder_rep (tensor): last layer encoder of the Bert (batch_size, q_len + ctx_len + 3, bert_hidden_size)
            @ctx_len (int): the context length. 
          Return:
            ctx_rep (tensor): contextual repersentation condition on question. (batch_size, ctx_len, bert_hidden_size)
          '''
          ctx_rep = qa_encoder_rep[:, -1-ctx_len:-1, :] #skip the last SEP
          return ctx_rep

      def forward(self, ques_idxs, ctx_idxs, ques_mask, ctx_mask):
          '''
          Args:
            @ques_idxs (tensor): Bert IDs of the question. (batch_size, q_len) Note that the q_len is a fixed number due to padding/clamping policy
            @ctx_idxs (tensor): Bert IDs of the context. (batch_size, ctx_len) Note that the ctx_len is a fixed number due to padding/clamping policy
            @ques_mask (tensor): elementwise 0,1 tensor to real token in question. (batch_size, q_len)
            @ctx_mask (tensor): elementwise 0,1 tensor to real token in context. (batch_size, ctx_len)
          Return:
            bert_ctx_emb (tensor): contextual embedding condition on question. (batch_size, ctx_len, bert_hidden_size)
          '''
          #pdb.set_trace()
          qa_idxs, segment_ids, bert_att = self.make_bert_input(ques_idxs, ctx_idxs, ques_mask, ctx_mask)
          #qa_idxs = qa_idxs.cuda()
          #segment_ids = segment_ids.cuda()
          #bert_att = bert_att.cuda()
          ctx_code, _ = self.bert(qa_idxs, segment_ids, bert_att, output_all_encoded_layers=False)
          bert_ctx_emb = self.parse_context_rep(ctx_code, ctx_idxs.size(1))
          return bert_ctx_emb
 
