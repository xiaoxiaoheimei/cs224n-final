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

      def make_bert_compact_input(self, ques_idxs, ctx_idxs, ques_mask, ctx_mask):
          '''
          Args:
           The same as make_bert_input
          Return:
           qa_idx, segment_ids, bert_att: same as make_bert_input
           q_restore_mask: 
           ctx_restore_mask:
          '''
          q_lens = ques_mask.sum(-1) #(batch_size,)
          ctx_lens = ctx_mask.sum(-1)#(batch_size,)
          batch_size, q_limits = ques_mask.size()
          ctx_limits = ctx_mask.size(1)
          qa_idx = t.zeros((batch_size, ctx_limits + q_limits + 3), dtype=t.long, device=self.device)
          q_restore_mask = t.zeros_like(ques_mask, dtype=t.long, device=self.device)
          ctx_restore_mask = t.zeros_like(ctx_mask, dtype=t.long, device=self.device)
          bert_att = t.zeros((batch_size, ctx_limits + q_limits + 3), dtype=t.long, device=self.device)
          segment_ids = t.zeros((batch_size, ctx_limits + q_limits + 3), dtype=t.long, device=self.device)
          for i in range(batch_size):
            qa_idx[i, 0] = self.CLS # CLS 
            qa_idx[i, 1:1+q_lens[i]] = ques_idxs[i, 0:q_lens[i]] # CLS <Question> 
            qa_idx[i, 1+q_lens[i]] = self.SEP # CLS <Question> SEP
            qa_idx[i, 2+q_lens[i]:2+q_lens[i]+ctx_lens[i]] = ctx_idxs[i, 0:ctx_lens[i]] # CLS <Question> SEP  <Context>
            qa_idx[i, 2+q_lens[i]+ctx_lens[i]] = self.SEP #CLS <Question> SEP  <Context> SEP
            segment_ids[i, 2+q_lens[i]:3+q_lens[i]+ctx_lens[i]] = 1
            bert_att[i, 0:3+q_lens[i]+ctx_lens[i]] = 1

            q_restore_mask[i, 0:q_lens[i]] = t.tensor(list(range(1, 1+q_lens[i])), dtype=t.long, device=self.device)
            ctx_restore_mask[i, 0:ctx_lens[i]] = t.tensor(list(range(2+q_lens[i], 2+q_lens[i]+ctx_lens[i])), dtype=t.long, device=self.device)

          ###test###
          assert t.all((t.gather(qa_idx, 1, q_restore_mask) == ques_idxs).long().sum(-1) == q_lens)
          assert t.all((t.gather(qa_idx, 1, ctx_restore_mask) == ctx_idxs).long().sum(-1) == ctx_lens)
          #########

          return qa_idx, segment_ids, bert_att, q_restore_mask, ctx_restore_mask

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

      def parse_compact_QA_rep(self, qa_encoder_rep, q_restore_mask, ctx_restore_mask):
          '''
          parse the context from the last layer encoder of the Bert.
          Args:
            @qa_encoder_rep (tensor): last layer encoder of the Bert (batch_size, q_len + ctx_len + 3, bert_hidden_size)
            @q_restore_mask (tensor): (batch_size, q_len)
            @ctx_restore_mask (tensor): (batch_size, ctx_len)
          Return:
            q_ctx_rep (tensor): contextual repersentation condition on question. (batch_size, q_len, bert_hidden_size)
            ctx_rep (tensor): contextual repersentation condition on question. (batch_size, ctx_len, bert_hidden_size)
          '''
          _, _, bert_hidden_size = qa_encoder_rep.size()
          q_restore_idx = q_restore_mask.unsqueeze(-1).repeat(1, 1, bert_hidden_size)
          ctx_restore_idx = ctx_restore_mask.unsqueeze(-1).repeat(1, 1, bert_hidden_size)
          q_ctx_rep = t.gather(qa_encoder_rep, 1, q_restore_idx)
          ctx_rep = t.gather(qa_encoder_rep, 1, ctx_restore_idx)
          return q_ctx_rep, ctx_rep

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
          qa_idxs, segment_ids, bert_att, q_restore_mask, ctx_restore_mask = self.make_bert_compact_input(ques_idxs, ctx_idxs, ques_mask, ctx_mask)
          #qa_idxs = qa_idxs.cuda()
          #segment_ids = segment_ids.cuda()
          #bert_att = bert_att.cuda()
          ctx_code, _ = self.bert(qa_idxs, segment_ids, bert_att, output_all_encoded_layers=False)
          _, bert_ctx_emb = self.parse_compact_QA_rep(ctx_code, q_restore_mask, ctx_restore_mask)
          return bert_ctx_emb
 
