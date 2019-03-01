"""Following code inspired by:
(1) [pytorch-pretrained-BERT]
https://github.com/huggingface/pytorch-pretrained-BERT#doc

"""

import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel

class BertEmbedding(nn.Module):
    def __init__(self):
        super(BertEmbedding, self).__init__()
        self.model = BertModel.from_pretrained("bert-base-uncased")
    
    def forward(self, tokens_tensor, segments_tensors):
    """
    @param input_ids: a torch.LongTensor of shape [batch_size, sequence_length] with the word token indices in the vocabulary.
    @param token_type_ids: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token types indices selected in [0, 1]. 
        Type 0 corresponds to a sentence A and type 1 corresponds to a sentence B token.
    @param attention_mask: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices selected in [0, 1]. 
        It's a mask to be used if some input sequence lengths are smaller than the max input sequence length of the current batch. 
        It's the mask that we typically use for attention when a batch has varying length sentences.
    @param output_all_encoded_layers: outputs a list of the encoded-hidden-states at the end of each attention block. Default: True.

    @returns encoded_layers
    @returns pooled_output (Tensor [batch_size, hidden_size]) 

    """
        # Load pre-trained model (weights)
        self.model.eval()

        # If you have a GPU, put everything on cuda
        tokens_tensor = tokens_tensor.to('cuda')
        segments_tensors = segments_tensors.to('cuda')
        model.to('cuda')
        
        # Predict hidden states features for each layer
        with torch.no_grad():
            encoded_layers, _ = model(tokens_tensor, segments_tensors)
        # We have a hidden states for each of the 12 layers in model bert-base-uncased
        assert len(encoded_layers) == 12

        return encoded_layers