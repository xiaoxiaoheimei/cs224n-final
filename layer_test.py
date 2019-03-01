####################################
## File: char_encoder.py           ##
## Author: Liu Yang                ##
## Email: liuyeung@stanford.edu    ##
####################################
from char_embeddings import CharEmbeddings
import torch as t
import util
from joint_context_layers import CharQAMultualContext, ConcatContext
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


def main(dataset):
    test_char_embeddings(dataset[0]);
    test_qa_multual_context(dataset)
    test_concat_context()

if __name__ == '__main__':
   dataset = util.SQuAD('./data/train.npz', True)
   main(dataset)
