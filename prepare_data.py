# coding: utf-8
# created by deng on 2019-01-18

import numpy as np
import joblib
import os
from gensim.models import KeyedVectors

from utils.path_util import from_project_root
import utils.json_util as ju

np.random.seed(233)
LABEL_IDS = {"DNA": 1, "RNA": 2, "protein": 3, "cell_line": 4, "cell_type": 5}

PRETRAINED_URL = from_project_root("data/PubMed-shuffle-win-30.bin")


def gen_vocab_from_data(data_url, pretrained_url, binary=True, update=False):
    """ generate vocabulary and embeddings from data file, generated vocab files will be saved in
        data dir

    Args:
        data_url: url to data file
        pretrained_url: url to pretrained embedding file
        binary: binary for load word2vec
        update: force to update even vocab file exists

    Returns:
        generated word embedding url
    """

    data_dir = os.path.dirname(data_url)
    vocab_url = os.path.join(data_dir, "vocab.json")
    char_vocab_url = os.path.join(data_dir, "char_vocab.json")
    embedding_url = os.path.join(data_dir, "embeddings.npy") if pretrained_url else None

    if (not update) and os.path.exists(vocab_url):
        print("vocab file already exists")
        return embedding_url

    vocab = set()
    char_vocab = set()
    print("generating vocab from", data_url)
    with open(data_url, 'r', encoding='utf-8') as data_file:
        for row in data_file:
            if row == '\n':
                continue
            vocab.add(row.split()[0])
            char_vocab = char_vocab.union(row.split()[0])

    # sorting vocab according alpha
    vocab = sorted(vocab)
    char_vocab = sorted(char_vocab)

    # generate word embeddings for vocab
    if pretrained_url is not None:
        print("generating pretrained embedding from", pretrained_url)
        kvs = KeyedVectors.load_word2vec_format(pretrained_url, binary=binary)
        embeddings = list()
        for word in vocab:
            if word in kvs:
                embeddings.append(kvs[word])
            else:
                embeddings.append(np.random.uniform(-0.25, 0.25, kvs.vector_size)),

    char_vocab = ['<pad', '<unk>'] + char_vocab
    vocab = ['<pad>', '<unk>'] + vocab
    ju.dump(ju.list_to_dict(vocab), vocab_url)
    ju.dump(ju.list_to_dict(char_vocab), char_vocab_url)

    if pretrained_url is None:
        return
    embeddings = np.vstack([np.zeros(kvs.vector_size),  # for <pad>
                            np.random.uniform(-0.25, 0.25, kvs.vector_size),  # for <unk>
                            embeddings])
    np.save(embedding_url, embeddings)
    return embedding_url


def infer_records(columns):
    """ inferring all entity records of a sentence

    Args:
        columns: columns of a sentence in iob2 format

    Returns:
        entity record in gave sentence

    """
    records = dict()
    for col in columns:
        start = 0
        while start < len(col):
            end = start + 1
            if col[start][0] == 'B':
                while end < len(col) and col[end][0] == 'I':
                    end += 1
                records[(start, end)] = col[start][2:]
            start = end
    return records


def load_raw_data(data_url, update=False):
    """ load data into sentences and records

    Args:
        data_url: url to data file
        update: whether force to update
    Returns:
        sentences(raw), records
    """

    # load from pickle
    save_url = data_url.replace('.bio', '.raw.pkl').replace('.iob2', '.raw.pkl')
    if not update and os.path.exists(save_url):
        return joblib.load(save_url)

    sentences = list()
    records = list()
    with open(data_url, 'r', encoding='utf-8') as iob_file:
        first_line = iob_file.readline()
        n_columns = first_line.count('\t')
        # JNLPBA dataset don't contains the extra 'O' column
        if 'jnlpba' in data_url:
            n_columns += 1
        columns = [[x] for x in first_line.split()]
        for line in iob_file:
            if line != '\n':
                line_values = line.split()
                for i in range(n_columns):
                    columns[i].append(line_values[i])

            else:  # end of a sentence
                sentence = columns[0]
                sentences.append(sentence)
                records.append(infer_records(columns[1:]))
                columns = [list() for i in range(n_columns)]
    joblib.dump((sentences, records), save_url)
    return sentences, records


def prepare_vocab(data_url, pretrained_url=PRETRAINED_URL, update=True):
    """ prepare vocab and embedding

    Args:
        data_url: url to data file for preparing vocab
        pretrained_url: url to pretrained embedding file
        update: force to update

    """
    binary = pretrained_url.endswith('.bin')
    gen_vocab_from_data(data_url, pretrained_url, binary=binary, update=update)


def main():
    # load_data(data_url, update=False)
    data_url = from_project_root("data/genia/genia.iob2")
    prepare_vocab(data_url, update=False)
    pass


if __name__ == '__main__':
    main()
