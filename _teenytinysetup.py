import numpy as np
import os
import spacy
import ujson as json
import urllib.request

from args import get_setup_args
from codecs import open
from collections import Counter
from subprocess import run
from tqdm import tqdm
from zipfile import ZipFile


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

def process_file(filename, data_type, word_counter, char_counter):
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
                context_tokens = word_tokenize(context)
                context_chars = [list(token) for token in context_tokens]
                spans = convert_idx(context, context_tokens)
                for token in context_tokens:
                    word_counter[token] += len(para["qas"])
                    for char in token:
                        char_counter[char] += len(para["qas"])
                for qa in para["qas"]:
                    total += 1
                    ques = qa["question"].replace(
                        "''", '" ').replace("``", '" ')
                    ques_tokens = word_tokenize(ques)
                    ques_chars = [list(token) for token in ques_tokens]
                    for token in ques_tokens:
                        word_counter[token] += 1
                        for char in token:
                            char_counter[char] += 1
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


def build_features(args, examples, data_type, out_file, word2idx_dict, char2idx_dict, is_test=False):
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

        def _get_word(word):
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in word2idx_dict:
                    return word2idx_dict[each]
            return 1

        def _get_char(char):
            if char in char2idx_dict:
                return char2idx_dict[char]
            return 1

        context_idx = np.zeros([para_limit], dtype=np.int32)
        context_char_idx = np.zeros([para_limit, char_limit], dtype=np.int32)
        ques_idx = np.zeros([ques_limit], dtype=np.int32)
        ques_char_idx = np.zeros([ques_limit, char_limit], dtype=np.int32)

        for i, token in enumerate(example["context_tokens"]):
            context_idx[i] = _get_word(token)
        context_idxs.append(context_idx)

        for i, token in enumerate(example["ques_tokens"]):
            ques_idx[i] = _get_word(token)
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


def word_tokenize(sent):
    nlp = spacy.blank("en")

    doc = nlp(sent)
    return [token.text for token in doc]

def save(filename, obj, message=None):
    if message is not None:
        print("Saving {}...".format(message))
        with open(filename, "w") as fh:
            json.dump(obj, fh)


if __name__ == '__main__':
    # Process training set and use it to decide on the word/character vocabularies
    word_counter, char_counter = Counter(), Counter()
    train_examples, train_eval = process_file("./data/teenytiny.json", "train", word_counter, char_counter)
    """
    word_emb_mat, word2idx_dict = get_embedding(
        word_counter, 'word', emb_file=args.glove_file, vec_size=args.glove_dim, num_vectors=args.glove_num_vecs)
    char_emb_mat, char2idx_dict = get_embedding(
        char_counter, 'char', emb_file=None, vec_size=args.char_dim)

    save('./teenytiny/word_emb.json', word_emb_mat, message="word embedding")
    save('./teenytiny/char_emb.json', char_emb_mat, message="char embedding")
    """

    save('./teenytiny/train_eval.json', train_eval, message="train eval")

    """
    save('./data/word2idx.json', word2idx_dict, message="word dictionary")
    save('./data/char2idx.json', char2idx_dict, message="char dictionary")
    """