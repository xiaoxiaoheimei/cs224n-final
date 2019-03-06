####################################
## File: joint_context_models.py   ##
## Author: Liu Yang                ##
## Email: liuyeung@stanford.edu    ##
####################################
import torch
import torch.nn as nn
import torch.nn.functional as F
import util
from joint_context_layers import CharQAMultualContext, ConcatContext, AnswerablePredictor, StartLocationPredictor, EndLocationPredictor
from char_embeddings import CharEmbeddings
from bert_word_embeddings import BertWordEmbedding
from layers import RNNEncoder

class JointContextQA(nn.Module):
      '''
      The model integrates question-aware context at both character level and word level.
      The model conducts answerable/start/end predictions with our proposed Joint Probability Predictior. 
      '''

      def __init__(self, bert_config, char_emb_dim, char_vector, drop_prob=0.3,
                         word_emb_dim=768, cat_reduce_factor=0.5, lstm_dim_bm=768, lstm_dim_pred=768, device=torch.device("cpu")):
          '''
          Args:
            @param bert_config (BertConfig): the configuration for Bert word embedding layer.
            @param char_emb_dim (int): the dimension of word embedding at charactor level.
            @param char_vector (ndarray): the charactor embedding.
            @param drop_prob (float): the dropout probability.
            @param word_emb_dim (int): the dimension of word embedding of Bert.
            @param cat_reduce_factor (float): the fraction of dimension we keep when doing charEmb/wordEmb concat.
            @param lstm_dim_bm (int): the LSTM hidden dimension of BiDAF module layer.
            @param lstm_dim_pred (int): the LSTM hidden dimension of Joint Probability Predictor.
            @param device (torch.device): target platform.
          '''
          super(JointContextQA, self).__init__()
          self.word_emb_dim = word_emb_dim
          self.char_emb_dim = char_emb_dim
          self.charCtx = CharQAMultualContext(char_emb_dim, char_vector, drop_prob)
          self.bertCtx = BertWordEmbedding(bert_config, device)
          self.ctxConcat = ConcatContext(word_emb_dim, self.charCtx.ctx_emb_dim, cat_reduce_factor)
          self.mod = RNNEncoder(input_size=self.ctxConcat.output_dim, hidden_size=lstm_dim_bm, num_layers=2, drop_prob=drop_prob)
          M_dim = 2*lstm_dim_bm
          G_dim = self.ctxConcat.output_dim
          self.predAnswer = AnswerablePredictor(input_dim=(M_dim+G_dim), lstm_hdim=lstm_dim_pred, drop_prob=drop_prob)
          self.predStart = StartLocationPredictor(M_dim=M_dim, G_dim=G_dim, lstm_hdim=lstm_dim_pred, drop_prob=drop_prob)
          self.predEnd = EndLocationPredictor(M_dim=M_dim, G_dim=G_dim, lstm_hdim=lstm_dim_pred, drop_prob=drop_prob)

      def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs):
          '''
          Args:
            @param cw_idxs (tensor): the Bert IDs of the context words. (batch_size, ctx_len)
            @param qw_idxs (tensor): the Bert IDs of the question words. (batch_size, q_len)
            @param cc_idxs (tensor): the character IDs of the context characters. (batch_size, ctx_len, char_limit)
            @param qc_idxs (tensor): the character IDs of the question characters. (batch_size, q_len, char_limit)
          Return:
            logits (tensor): logits of (0,1), 0 for no answer, 1 for answerable. (batch_size, 2)
            p_start (tensor): the distribution of the start location. (batch_size, ctx_len)
            p_end (tensor): the distribution of the end location. (batch_size, q_len)
          '''
          ctx_mask = torch.zeros_like(cw_idxs) != cw_idxs #(batch_size, ctx_len)
          q_mask = torch.zeros_like(qw_idxs) != qw_idxs  #(batch_size, ctx_len)
          char_ctx_on_q = self.charCtx(cc_idxs, qc_idxs) #(batch_size, ctx_len, self.char_emb_dim)
          word_ctx_on_q = self.bertCtx(qw_idxs, cw_idxs, q_mask, ctx_mask) #(batch_size, ctx_len, self.word_emb_dim)
          G = self.ctxConcat(char_ctx_on_q, word_ctx_on_q) #(batch_size, ctx_len, G_dim)
          #print("G.size:{}".format(G.size()))
          lengths = ctx_mask.sum(-1)
          M = self.mod(G, lengths) #(batch_size, M_dim)
          #print("M.size:{}".format(M.size()))
          ans_logits, init_enc_start = self.predAnswer(M, G, ctx_mask) #(batch_size, 2), (batch_size, lstm_dim_pred)
          p_start, init_enc_end = self.predStart(M, G, ctx_mask, init_enc_start) #(batch_size, ctx_len), (batch_size, lstm_dim_pred)
          p_end = self.predEnd(M, G, ctx_mask, init_enc_start, init_enc_end) #(batch_size, ctx_len)
 
          return ans_logits, p_start, p_end

def compute_loss(ans_logits, p_start, p_end, y1, y2):
    '''
      Function:
      compute the joint loss.
    ''' 
    ans_label = (y1 != -1).long() # 0 for no-answer, 1 for has-answer
    weight = torch.tensor([2., 1.], device=torch.device("cuda")) #punish 0 prediction more to avoid the tendency to predict 1 in which case accounts the location loss
    ans_loss = F.cross_entropy(ans_logits, ans_label, weight=weight, size_average=False)

    start_weight = -1. * ans_label.float() * p_start[:, y1].diag() #only keep element with ans_label = 1
    end_weight = -1. * ans_label.float() * p_end[:, y2].diag() #only keep element with ans_label = 1
    loc_loss = ans_loss + start_weight.sum(-1) + end_weight.sum(-1)
         
    loss = ans_loss + loc_loss
    return loss



