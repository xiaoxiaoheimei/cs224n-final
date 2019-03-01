
### In Construction 

# CS224n-final with BERT pre-trained embeddings and BiDAF attention flow

## 0. Introduction

This repository is for Liu and Lijing's final project of cs224n for default SQuAD 2.0 task based on [PyTorch reimplementation](https://github.com/huggingface/pytorch-pretrained-BERT) of [Google's TensorFlow repository for the BERT model](https://github.com/google-research/bert) with the paper: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805).

### Implementation

You can check what was the last saved dir and specify like this:

$ python ./test.py  --split dev -n baseline --load_path save/train/baseline-03/best.pth.tar