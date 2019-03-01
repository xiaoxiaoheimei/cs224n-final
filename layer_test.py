####################################
## File: char_encoder.py           ##
## Author: Liu Yang                ##
## Email: liuyeung@stanford.edu    ##
####################################
from char_embeddings import CharEmbeddings
import torch as t
import util

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


def main(dataset):
    test_char_embeddings(dataset[0]);

if __name__ == '__main__':
   dataset = util.SQuAD('./data/train.npz', True)
   main(dataset)
