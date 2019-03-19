# coding: utf-8
# created by deng on 2019-03-13

import numpy as np
import torch
from os.path import dirname

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from prepare_data import load_raw_data

import utils.json_util as ju


def gen_sentence_tensors(sentence_list, device, data_url):
    """ generate input tensors from sentence list

    Args:
        sentence_list: list of raw sentence
        device: torch device
        data_url: raw data url to locate the vocab url

    Returns:
        sentences, tensor
        sentence_lengths, tensor
        sentence_words, list of tensor
        sentence_word_lengths, list of tensor
        sentence_word_indices, list of tensor

    """
    vocab = ju.load(dirname(data_url) + '/vocab.json')
    char_vocab = ju.load(dirname(data_url) + '/char_vocab.json')

    sentences = list()
    sentence_words = list()
    sentence_word_lengths = list()
    sentence_word_indices = list()

    unk_idx = 1
    for sent in sentence_list:
        # word to word id
        sentence = torch.LongTensor([vocab[word] if word in vocab else unk_idx
                                     for word in sent]).to(device)

        # char of word to char id
        words = list()
        for word in sent:
            words.append([char_vocab[ch] if ch in char_vocab else unk_idx
                          for ch in word])

        # save word lengths
        word_lengths = torch.LongTensor([len(word) for word in words]).to(device)

        # sorting lengths according to length
        word_lengths, word_indices = torch.sort(word_lengths, descending=True)

        # sorting word according word length
        words = np.array(words)[word_indices.cpu().numpy()]
        word_indices = word_indices.to(device)
        words = [torch.LongTensor(word).to(device) for word in words]

        # padding char tensor of words
        words = pad_sequence(words, batch_first=True).to(device)
        # (max_word_len, sent_len)

        sentences.append(sentence)
        sentence_words.append(words)
        sentence_word_lengths.append(word_lengths)
        sentence_word_indices.append(word_indices)

    # record sentence length and padding sentences
    sentence_lengths = [len(sentence) for sentence in sentences]
    # (batch_size)
    sentences = pad_sequence(sentences, batch_first=True).to(device)
    # (batch_size, max_sent_len)

    return sentences, sentence_lengths, sentence_words, sentence_word_lengths, sentence_word_indices


class ExhaustiveDataset(Dataset):
    label_ids = {"neither": 0, "DNA": 1, "RNA": 2, "protein": 3, "cell_line": 4, "cell_type": 5, "padding": 6}

    def __init__(self, data_url, device, max_region=10):
        super().__init__()
        self.x, self.y = load_raw_data(data_url)
        self.data_url = data_url
        self.max_region = max_region
        self.device = device

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

    def collate_func(self, data_list):
        data_list = sorted(data_list, key=lambda tup: len(tup[0]), reverse=True)
        sentence_list, records_list = zip(*data_list)  # un zip
        max_sent_len = len(sentence_list[0])
        sentence_tensors = gen_sentence_tensors(sentence_list, self.device, self.data_url)
        # (sentences, sentence_lengths, sentence_words, sentence_word_lengths, sentence_word_indices)

        region_labels = list()
        for records, length in zip(records_list, sentence_tensors[1]):
            labels = list()
            for region_size in range(1, self.max_region + 1):
                for start in range(0, max_sent_len - region_size + 1):
                    if start + region_size > length:
                        labels.append(6)
                    elif (start, start + region_size) in records:
                        labels.append(self.label_ids[records[start, start + region_size]])
                    else:
                        labels.append(0)
            region_labels.append(labels)
        region_labels = torch.LongTensor(region_labels).to(self.device)
        # (batch_size, n_regions)

        return sentence_tensors, region_labels


def main():
    pass


if __name__ == '__main__':
    main()
