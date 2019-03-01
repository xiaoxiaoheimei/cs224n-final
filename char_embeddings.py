####################################
## File: char_encoder.py           ##
## Author: Liu Yang                ##
## Email: liuyeung@stanford.edu    ##
####################################

import torch
from layers import HighwayEncoder
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
   def __init__(self, in_c, out_c, k_size=5):
       """
       @param in_c (int): input channel (chracter embed dim)
       @param out_c (int): output channel (feature dim)
       @k_size (int): kernel size
       """
       super(CNN, self).__init__()
       self.conv = nn.Conv1d(in_c, out_c, kernel_size=k_size)
       nn.init.xavier_uniform_(self.conv.weight)

   def forward(self, x):
       """
       @param x (tensor): input (batch_size, character_embed_dim, word_length)
       @return x_co (tensor): MaxPool(Relu(Conv1d(x))) (batch_size, character_embed_dim)
       """
       x_conv = self.conv(x)
       x_relu = F.relu(x_conv)
       x_co = F.max_pool1d(x_relu, kernel_size=x_relu.size(-1)) #(batch_size, character_embed_dim, 1)
       x_co = x_co.view(x_co.size(0), -1) #(character_size, character_embem_dim)
       return x_co

class CharEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, char_vector, drop_prob=0.3):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(CharEmbeddings, self).__init__()
        self.embed_size = embed_size
        self.embeddings = nn.Embedding.from_pretrained(char_vector) #initialize the char embeddings from pretrain
        self.cnn = CNN(self.embeddings.embedding_dim, embed_size)
        self.hy = HighwayEncoder(1, embed_size)
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        #print(input.size(), input, self.embeddings)
        input_cvec = self.embeddings(input)
        input_cvec = torch.transpose(input_cvec, 2, 3) #(sentence_lenght, batch_size, embed_size, max_word_length)
        x_embed = []
        for word_batch in input_cvec:
           wb_conv = self.cnn(word_batch) #(batch_size, embed_size)
           wb_hy = self.hy(wb_conv) #(betch_size, embed_size)
           wb_dropout = self.dropout(wb_hy)
           x_embed.append(wb_dropout)
        x_embed = torch.stack(x_embed, dim=0)
        return x_embed


