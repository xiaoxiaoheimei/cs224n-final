"""Download and pre-process SQuAD and GloVe.

Usage:
    > source activate squad
    > python setup.py

Pre-processing code adapted from:
    > https://github.com/HKUST-KnowComp/R-Net/blob/master/prepro.py

Author:
    Chris Chute (chute@stanford.edu)
"""

import numpy as np
import os
import spacy
import ujson as json
import urllib.request
from pytorch_pretrained_bert import BertTokenizer

from args import get_setup_args
from codecs import open
from collections import Counter
from subprocess import run
from tqdm import tqdm
from zipfile import ZipFile

import re


def download_url(url, output_path, show_progress=True):
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    if show_progress:
        # Download with a progress bar
        with DownloadProgressBar(unit='B', unit_scale=True,
                                 miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url,
                                       filename=output_path,
                                       reporthook=t.update_to)
    else:
        # Simple download with no progress bar
        urllib.request.urlretrieve(url, output_path)


def url_to_data_path(url):
    return os.path.join('./data/', url.split('/')[-1])


def download(args):
    downloads = [
        # Can add other downloads here (e.g., other word vectors)
        ('GloVe word vectors', args.glove_url),
    ]

    for name, url in downloads:
        output_path = url_to_data_path(url)
        if not os.path.exists(output_path):
            print('Downloading {}...'.format(name))
            download_url(url, output_path)

        if os.path.exists(output_path) and output_path.endswith('.zip'):
            extracted_path = output_path.replace('.zip', '')
            if not os.path.exists(extracted_path):
                print('Unzipping {}...'.format(name))
                with ZipFile(output_path, 'r') as zip_fh:
                    zip_fh.extractall(extracted_path)

    print('Downloading spacy language model...')
    run(['python', '-m', 'spacy', 'download', 'en'])

#############################################
# is_whitespace: code from bert squad example
#############################################
def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
       return True
    return False

###################################################################
# Seperate sent (str) into document token and its [start, end) span
###################################################################
def separate_words(sent):
    doc_tokens = []
    char_to_word_offset = []
    prev_is_whitespace = True
    doc_token_locs = [] #tuple of [start, end) for each token
    start = end = 0
    for idx, c in enumerate(sent):
      if is_whitespace(c):
         prev_is_whitespace = True
         if start < end:
            doc_token_locs.append((start, end))
            start = end
      else:
         if prev_is_whitespace:
            doc_tokens.append(c)
            start = idx
            end = start + 1
         else:
            doc_tokens[-1] += c
            end += 1
         prev_is_whitespace = False
      char_to_word_offset.append(len(doc_tokens) - 1)
    if start < end: #in case the last word has not been added
      doc_token_locs.append((start, end))

    assert len(doc_tokens) == len(doc_token_locs)
    return doc_tokens, doc_token_locs

########################################################################################
# is_word_piece: to determine whether the bert-token is a word peice starting with '##' #
########################################################################################
def is_word_piece(token):
    if re.match('##', token) is not None:
       return True
    else:
       return False

###############################################################################
# Generate bert tokens and the corresponding spans in the raw text            #
###############################################################################
def word_char_bert_tokenize(bert_tokenizer, doc_tokens, doc_token_locs, word_counter, char_counter, increment=1):
    bert_word_tokens = []
    bert_char_tokens = [] #character tokens
    spans = []
    for i, dtoken in enumerate(doc_tokens):
      sub_tokens = bert_tokenizer.tokenize(dtoken) 
      start, end = doc_token_locs[i]
      for sub_token in sub_tokens:
        if not is_word_piece(sub_token):
           end = start + len(sub_token)
           spans.append((start, end))
           bert_char_tokens.append(list(sub_token)) 
        else: 
           word_piece = sub_token[2:] #skip split '##'
           end = start + len(word_piece)
           spans.append((start, end))
           bert_char_tokens.append(list(word_piece)) #only add effictive word part, skip the split '##'
        start = end
        bert_word_tokens.append(sub_token)
        word_counter[sub_token] += increment
        for c in bert_char_tokens[-1]:
           char_counter[c] += increment
 
    return bert_word_tokens, bert_char_tokens, spans

def word_tokenize(sent):
    tokens = tokenizer.tokenize(sent) #obtain bert tokens
    return tokens


def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


def process_file(filename, bert_tokenizer, data_type, word_counter, char_counter):
    print("Pre-processing {} examples...".format(data_type))
    examples = []
    eval_examples = {}
    total = 0
    with open(filename, "r") as fh:
        source = json.load(fh)
        for article in tqdm(source["data"]):
            for para in article["paragraphs"]:
                context = para["context"].replace(
                    "''", '" ').replace("``", '" ')
                ctx_doc_tokens, ctx_doc_token_locs = separate_words(context)
                context_tokens, context_chars, spans = word_char_bert_tokenize(bert_tokenizer, ctx_doc_tokens,
                                                                               ctx_doc_token_locs, word_counter,
                                                                               char_counter, increment=len(para["qas"]))
                for qa in para["qas"]:
                    total += 1
                    ques = qa["question"].replace(
                        "''", '" ').replace("``", '" ')
                    ques_doc_tokens, ques_doc_token_locs = separate_words(ques)
                    ques_tokens, ques_chars, _ = word_char_bert_tokenize(bert_tokenizer, ques_doc_tokens, 
                                                                          ques_doc_token_locs, word_counter,
                                                                          char_counter, increment=1)
                    y1s, y2s = [], []
                    answer_texts = []
                    for answer in qa["answers"]:
                        answer_text = answer["text"]
                        answer_start = answer['answer_start']
                        answer_end = answer_start + len(answer_text)
                        answer_texts.append(answer_text)
                        answer_span = []
                        for idx, span in enumerate(spans):
                            if not (answer_end <= span[0] or answer_start >= span[1]):
                                answer_span.append(idx)
                        y1, y2 = answer_span[0], answer_span[-1]
                        y1s.append(y1)
                        y2s.append(y2)
                    example = {"context_tokens": context_tokens,
                               "context_chars": context_chars,
                               "ques_tokens": ques_tokens,
                               "ques_chars": ques_chars,
                               "y1s": y1s,
                               "y2s": y2s,
                               "id": total}
                    examples.append(example)
                    eval_examples[str(total)] = {"context": context,
                                                 "question": ques,
                                                 "spans": spans,
                                                 "answers": answer_texts,
                                                 "uuid": qa["id"]}
        print("{} questions in total".format(len(examples)))
    return examples, eval_examples


def get_embedding(counter, data_type, limit=-1, emb_file=None, vec_size=None, num_vectors=None):
    print("Pre-processing {} vectors...".format(data_type))
    embedding_dict = {}
    filtered_elements = [k for k, v in counter.items() if v > limit]
    if emb_file is not None:
        assert vec_size is not None
        with open(emb_file, "r", encoding="utf-8") as fh:
            for line in tqdm(fh, total=num_vectors):
                array = line.split()
                word = "".join(array[0:-vec_size])
                vector = list(map(float, array[-vec_size:]))
                if word in counter and counter[word] > limit:
                    embedding_dict[word] = vector
        print("{} / {} tokens have corresponding {} embedding vector".format(
            len(embedding_dict), len(filtered_elements), data_type))
    else:
        assert vec_size is not None
        for token in filtered_elements:
            embedding_dict[token] = [np.random.normal(
                scale=0.1) for _ in range(vec_size)]
        print("{} tokens have corresponding {} embedding vector".format(
            len(filtered_elements), data_type))

    NULL = "--NULL--"
    OOV = "--OOV--"
    token2idx_dict = {token: idx for idx, token in enumerate(embedding_dict.keys(), 2)}
    token2idx_dict[NULL] = 0
    token2idx_dict[OOV] = 1
    embedding_dict[NULL] = [0. for _ in range(vec_size)]
    embedding_dict[OOV] = [0. for _ in range(vec_size)]
    idx2emb_dict = {idx: embedding_dict[token]
                    for token, idx in token2idx_dict.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    return emb_mat, token2idx_dict


def convert_to_features(args, data, word2idx_dict, char2idx_dict, is_test):
    example = {}
    context, question = data
    context = context.replace("''", '" ').replace("``", '" ')
    question = question.replace("''", '" ').replace("``", '" ')
    example['context_tokens'] = word_tokenize(context)
    example['ques_tokens'] = word_tokenize(question)
    example['context_chars'] = [list(token) for token in example['context_tokens']]
    example['ques_chars'] = [list(token) for token in example['ques_tokens']]

    para_limit = args.test_para_limit if is_test else args.para_limit
    ques_limit = args.test_ques_limit if is_test else args.ques_limit
    char_limit = args.char_limit

    def filter_func(example):
        return len(example["context_tokens"]) > para_limit or \
               len(example["ques_tokens"]) > ques_limit

    if filter_func(example):
        raise ValueError("Context/Questions lengths are over the limit")

    context_idxs = np.zeros([para_limit], dtype=np.int32)
    context_char_idxs = np.zeros([para_limit, char_limit], dtype=np.int32)
    ques_idxs = np.zeros([ques_limit], dtype=np.int32)
    ques_char_idxs = np.zeros([ques_limit, char_limit], dtype=np.int32)

    def _get_word(word):
        for each in (word, word.lower(), word.capitalize(), word.upper()):
            if each in word2idx_dict:
                return word2idx_dict[each]
        return 1

    def _get_char(char):
        if char in char2idx_dict:
            return char2idx_dict[char]
        return 1

    for i, token in enumerate(example["context_tokens"]):
        context_idxs[i] = _get_word(token)

    for i, token in enumerate(example["ques_tokens"]):
        ques_idxs[i] = _get_word(token)

    for i, token in enumerate(example["context_chars"]):
        for j, char in enumerate(token):
            if j == char_limit:
                break
            context_char_idxs[i, j] = _get_char(char)

    for i, token in enumerate(example["ques_chars"]):
        for j, char in enumerate(token):
            if j == char_limit:
                break
            ques_char_idxs[i, j] = _get_char(char)

    return context_idxs, context_char_idxs, ques_idxs, ques_char_idxs


def is_answerable(example):
    return len(example['y2s']) > 0 and len(example['y1s']) > 0


def build_features(args, examples, bert_tokenizer, data_type, out_file, char2idx_dict, is_test=False):
    para_limit = args.test_para_limit if is_test else args.para_limit
    ques_limit = args.test_ques_limit if is_test else args.ques_limit
    ans_limit = args.ans_limit
    char_limit = args.char_limit

    def drop_example(ex, is_test_=False):
        if is_test_:
            drop = False
        else:
            drop = len(ex["context_tokens"]) > para_limit or \
                   len(ex["ques_tokens"]) > ques_limit or \
                   (is_answerable(ex) and
                    ex["y2s"][0] - ex["y1s"][0] > ans_limit)

        return drop

    print("Converting {} examples to indices...".format(data_type))
    total = 0
    total_ = 0
    meta = {}
    context_idxs = []
    context_char_idxs = []
    ques_idxs = []
    ques_char_idxs = []
    y1s = []
    y2s = []
    ids = []
    for n, example in tqdm(enumerate(examples)):
        total_ += 1

        if drop_example(example, is_test):
            continue

        total += 1

        def _get_char(char):
            if char in char2idx_dict:
                return char2idx_dict[char]
            return 1

        context_idx = np.array(bert_tokenizer.convert_tokens_to_ids(example["context_tokens"]), dtype=np.int32)
        context_idx = np.pad(context_idx, (0, para_limit - len(context_idx)), 'constant', constant_values=0)
        context_char_idx = np.zeros([para_limit, char_limit], dtype=np.int32)
        ques_idx = np.array(bert_tokenizer.convert_tokens_to_ids(example["ques_tokens"]), dtype=np.int32)
        ques_idx = np.pad(ques_idx, (0, ques_limit - len(ques_idx)), 'constant', constant_values=0)
        ques_char_idx = np.zeros([ques_limit, char_limit], dtype=np.int32)

        context_idxs.append(context_idx)
        ques_idxs.append(ques_idx)

        for i, token in enumerate(example["context_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                context_char_idx[i, j] = _get_char(char)
        context_char_idxs.append(context_char_idx)

        for i, token in enumerate(example["ques_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                ques_char_idx[i, j] = _get_char(char)
        ques_char_idxs.append(ques_char_idx)

        if is_answerable(example):
            start, end = example["y1s"][-1], example["y2s"][-1]
        else:
            start, end = -1, -1

        y1s.append(start)
        y2s.append(end)
        ids.append(example["id"])

    np.savez(out_file,
             context_idxs=np.array(context_idxs),
             context_char_idxs=np.array(context_char_idxs),
             ques_idxs=np.array(ques_idxs),
             ques_char_idxs=np.array(ques_char_idxs),
             y1s=np.array(y1s),
             y2s=np.array(y2s),
             ids=np.array(ids))
    print("Built {} / {} instances of features in total".format(total, total_))
    meta["total"] = total
    return meta


def save(filename, obj, message=None):
    if message is not None:
        print("Saving {}...".format(message))
        with open(filename, "w") as fh:
            json.dump(obj, fh)


def pre_process(args, tokenizer):
    # Process training set and use it to decide on the word/character vocabularies
    word_counter, char_counter = Counter(), Counter()
    train_examples, train_eval = process_file(args.train_file, tokenizer, "train", word_counter, char_counter)
    char_emb_mat, char2idx_dict = get_embedding(
        char_counter, 'char', emb_file=None, vec_size=args.char_dim)

    # Process dev and test sets
    dev_examples, dev_eval = process_file(args.dev_file, tokenizer, "dev", word_counter, char_counter)
    build_features(args, train_examples, tokenizer, "train", args.train_bert_record_file, char2idx_dict)
    dev_meta = build_features(args, dev_examples, tokenizer, "dev", args.dev_bert_record_file, char2idx_dict)
    if args.include_test_examples:
        test_examples, test_eval = process_file(args.test_file, tokenizer, "test", word_counter, char_counter)
        save(args.test_bert_eval_file, test_eval, message="test eval")
        test_meta = build_features(args, test_examples, tokenizer, "test",
                                   args.test_bert_record_file, char2idx_dict, is_test=True)
        save(args.test_bert_meta_file, test_meta, message="test meta")

    save(args.char_bert_emb_file, char_emb_mat, message="char embedding")
    save(args.train_bert_eval_file, train_eval, message="train eval")
    save(args.dev_bert_eval_file, dev_eval, message="dev eval")
    save(args.char2idx_bert_file, char2idx_dict, message="char dictionary")
    save(args.dev_bert_meta_file, dev_meta, message="dev meta")


if __name__ == '__main__':
    # Get command-line args
    args_ = get_setup_args()


    # Import spacy language model
    bert_tokenizer= BertTokenizer.from_pretrained('bert-base-uncased')

    # Preprocess dataset
    args_.train_file = url_to_data_path(args_.train_url)
    args_.dev_file = url_to_data_path(args_.dev_url)
    if args_.include_test_examples:
        args_.test_file = url_to_data_path(args_.test_url)
    pre_process(args_, bert_tokenizer)
