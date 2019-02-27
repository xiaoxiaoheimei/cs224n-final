"""Following code inspired by [BERT FineTuning with Cloud TPU: Sentence and Sentence-Pair Classification Tasks]
https://colab.research.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb#scrollTo=tYkaAlJNfhul"""

import os
import sys
import layers

# !test -d pytorch-pretrained-BERT || git clone https://github.com/huggingface/pytorch-pretrained-BERT
if not 'pytorch-pretrained-BERT' in sys.path:
    sys.path += ['pytorch-pretrained-BERT']

cwd = os.getcwd()
DataPath = cwd + "\\bert\\"

# Available pretrained model checkpoints:
#   uncased_L-12_H-768_A-12: uncased BERT base model
#   uncased_L-24_H-1024_A-16: uncased BERT large model
#   cased_L-12_H-768_A-12: cased BERT large model
BERT_MODEL = 'uncased_L-12_H-768_A-12' #@param {type:"string"}
BERT_PRETRAINED_DIR = DataPath + BERT_MODEL
print('***** BERT pretrained directory: {} *****'.format(BERT_PRETRAINED_DIR))


class BertEmbedding(nn.Module):
    def __init__(self, word_vectors, hidden_size, drop_prob=0.):
        super(BiDAF, self).__init__()