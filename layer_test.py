####################################
## File: char_encoder.py           ##
## Author: Liu Yang                ##
## Email: liuyeung@stanford.edu    ##
####################################
from char_embeddings import CharEmbeddings
import torch as t
import util
from joint_context_layers import CharQAMultualContext, ConcatContext, AnswerablePredictor, StartLocationPredictor, EndLocationPredictor
import pdb

def test_char_embeddings(data_item):
    char_vectors = util.torch_from_json("./data/char_emb.json")
    print(char_vectors.size())
    context_ch_ind = 1
    question_ch_ind = 3
    context_char_idx = data_item[context_ch_ind].unsqueeze(1) #add batch dim
    question_char_idx = data_item[question_ch_ind].unsqueeze(1)
    emb_size = 48
    charEmb = CharEmbeddings(emb_size, char_vectors) 
    context_char_emb = charEmb(context_char_idx)
    question_char_emb = charEmb(question_char_idx)
    assert tuple(context_char_emb.size()) == (context_char_idx.size(0), 1, emb_size)
    assert tuple(question_char_emb.size()) == (question_char_idx.size(0), 1,  emb_size)
    print("Pass the test of char cnn embeddings!")

def test_qa_multual_context(dataeset):
    batch_size = 10
    hidden_size = 48
    char_vectors = util.torch_from_json("./data/char_emb.json")
    cc_idx = dataset[:batch_size][1]
    cq_idx = dataset[:batch_size][3]
    #pdb.set_trace()
    cqaContext = CharQAMultualContext(hidden_size, char_vectors)
    q_ctx = cqaContext(cc_idx, cq_idx)
    assert tuple(q_ctx.size()) == (batch_size, cc_idx.size(1), 8*hidden_size)
    print("Pass the test of building qa multual context!")

def test_concat_context():
    batch_size = 10
    ctx_len = 100
    w_ctx_dim = 300
    c_ctx_dim = 98
    reduce_factor = 0.8
    reducer = ConcatContext(w_ctx_dim, c_ctx_dim, reduce_factor)
    w_ctx = t.randn(batch_size, ctx_len, w_ctx_dim)
    c_ctx = t.randn(batch_size, ctx_len, c_ctx_dim)
    ctx = reducer(w_ctx, c_ctx)
    assert tuple(ctx.size()) == (batch_size, ctx_len, int((w_ctx_dim + c_ctx_dim)*reduce_factor))
    print("Pass the test of merging word/character contexts!")

def test_answerable_predictor():
    batch_size = 10
    M_dim = 20
    G_dim = 100
    ctx_len = 150
    lstm_hdim = 50
    input_dim = M_dim + G_dim
    M = t.randn(batch_size, ctx_len, M_dim)
    G = t.randn(batch_size, ctx_len, G_dim)
    c_mask = t.ones(batch_size, ctx_len)
    c_mask[:, ctx_len-30:] = 0.
    predictor = AnswerablePredictor(input_dim, lstm_hdim)
    logits, (h0, c0) = predictor(M, G, c_mask)
    assert tuple(h0.size()) == (2, batch_size, lstm_hdim)
    assert tuple(c0.size()) == (2, batch_size, lstm_hdim)
    assert tuple(logits.size()) == (batch_size, 2)
    print("Pass the test of answerable predictor layer!")

def test_start_predictor():
    batch_size = 10
    M_dim = 20
    G_dim = 100
    ctx_len = 150
    lstm_hdim = 50
    input_dim = M_dim + G_dim
    M = t.randn(batch_size, ctx_len, M_dim)
    G = t.randn(batch_size, ctx_len, G_dim)
    c_mask = t.ones(batch_size, ctx_len)
    c_mask[:, ctx_len-30:] = 0.
    enc_init = (t.randn(2, batch_size, lstm_hdim), t.randn(2, batch_size, lstm_hdim))
    predictor = StartLocationPredictor(M_dim, G_dim, lstm_hdim, drop_prob=0.1)
    p_start, (h0, c0) = predictor(M, G, c_mask, enc_init)
    assert tuple(h0.size()) == (2, batch_size, lstm_hdim)
    assert tuple(c0.size()) == (2, batch_size, lstm_hdim)
    assert tuple(p_start.size()) == (batch_size, ctx_len)
    print("Pass the test of start predictor layer!")

def test_end_predictor():
    batch_size = 10
    M_dim = 20
    G_dim = 100
    ctx_len = 150
    lstm_hdim = 50
    input_dim = M_dim + G_dim
    M = t.randn(batch_size, ctx_len, M_dim)
    G = t.randn(batch_size, ctx_len, G_dim)
    c_mask = t.ones(batch_size, ctx_len)
    c_mask[:, ctx_len-30:] = 0.
    enc_init_A = (t.randn(2, batch_size, lstm_hdim), t.randn(2, batch_size, lstm_hdim))
    enc_init_start = (t.randn(2, batch_size, lstm_hdim), t.randn(2, batch_size, lstm_hdim))
    predictor = EndLocationPredictor(M_dim, G_dim, lstm_hdim, drop_prob=0.1)
    p_end  = predictor(M, G, c_mask, enc_init_A, enc_init_start)
    assert tuple(p_end.size()) == (batch_size, ctx_len)
    print("Pass the test of end predictor layer!")

def main(dataset):
    test_char_embeddings(dataset[0]);
    test_qa_multual_context(dataset)
    test_concat_context()
    test_answerable_predictor()
    test_start_predictor()
    test_end_predictor()

if __name__ == '__main__':
   dataset = util.SQuAD('./data/train.npz', True)
   main(dataset)
