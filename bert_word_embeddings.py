####################################
## File: bert_word_embeddings.py   ##
## Author: Liu Yang                ##
## Email: liuyeung@stanford.edu    ##
####################################

import torch as t
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import modeling, BertTokenizer

class BertWordEmbedding(nn.Module):
      '''
      The layer of generate Bert contextual representation
      '''

      def __init__(self, config):
          '''
          Args:
            @config: configuration file of internal Bert layer
          '''
          self.bert = modeling.BertModel(config=config)
          tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
          self.CLS = tokenizer.convert_tokens_to_ids(['[CLS]'])[0] #ID of the Bert [CLS] token
          self.SEP = tokenizer.convert_tokens_to_ids(['[SEP]'])[0] #ID of the Bert [SEP] token

      def make_bert_input(ques_idxs, ctx_idxs, ques_mask, ctx_mask):
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
          pad_CLS = t.tensor([[self.CLS]]*batch_size, dtype=ques_idxs.dtype) #(batch_size, 1)
          pad_SEP = t.tensor([[self.SEP]]*batch_size, dtype=ques_idxs.dtype) #(batch_size, 1)
          ques_pad_CLS = t.cat((pad_CLS, ques_idxs), dim=-1) #[CLS] ques_idxs : (batch_size, q_len+1)
          ques_pad = t.cat((ques_pad_CLS, pad_SEP), dim=-1) #[CLS] ques_idxs [SEP] : (batch_size, q_len+2)
          ctx_pad = t.cat((ctx_idxs, pad_SEP), dim=-1) # ctx_idxs [SEP] : (batch_size, ctx_len+1)
          qa_idxs = t.cat((ques_pad, ctx_pad), dim=-1) #[CLS] ques_idxs [SEP] ctx_idxs [SEP] : (batch_size, q_len + ctx_len + 3)
          segment_ids = t.cat((t.zeros_like(ques_pad), t.ones_like(ctx_pad)), dim=-1) # (batch_size, ctx_len + q_len + 3)

          att_pad = t.tensor([[0]]*batch_size, dtype=ques_mask.dtype) # (batch_size, 1)
          qmask_pad = t.cat((t.cat((att_pad, ques_mask), dim=-1), att_pad), dim=-1) # (batch_size, q_len+2)
          cxt_mask_pad = t.cat((ctx_mask, att_pad), dim=-1) # (batch_size, ctx_len+1)
          bert_att = t.cat((qmask_pad, cxt_mask_pad), dim=-1) # [0, ques_mask, 0, ctx_mask, 0]: (batch_size, ctx_len+q_len+3)

          return qa_idxs, segment_ids, bert_att


