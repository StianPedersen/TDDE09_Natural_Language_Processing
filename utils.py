import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

COLOR = {
    'G':'\033[0;32m',
    'bG':'\033[1;32m',
    'Y':'\033[0;33m',
    'bY':'\033[1;33m',
    'C': '\033[0m'
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PAD = '<pad>'
UNK = '<unk>'

class Dataset():

    ROOT = ('<root>', '<root>', 0)  # Pseudo-root

    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        with open(self.filename, 'rt', encoding='utf-8') as lines:
            tmp = [Dataset.ROOT]
            for line in lines:
                if not line.startswith('#'):  # Skip lines with comments
                    line = line.rstrip()
                    if line:
                        columns = line.split('\t')
                        if columns[0].isdigit():  # Skip range tokens
                            tmp.append(
                                (columns[1], columns[3], int(columns[6])))
                    else:
                        yield tmp
                        tmp = [Dataset.ROOT]

class TaggedDataset():

    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        with open(self.filename, 'rt', encoding='utf-8') as lines:
            tmp = []
            for line in lines:
                if not line.startswith('#'):  # Skip lines with comments
                    line = line.rstrip()
                    if line:
                        columns = line.split('\t')
                        if columns[0].isdigit():  # Skip range tokens
                            tmp.append(columns)
                    else:
                        yield tmp
                        tmp = []

def make_vocabs(gold_data):
    vocab = {PAD: 0, UNK: 1}
    tags = {PAD: 0}
    for sentence in gold_data:
        for pair in sentence:
            word = pair[0]
            tag = pair[1]

            if word not in vocab:
                vocab[word] = len(vocab)

            if tag not in tags:
                tags[tag] = len(tags)

    return vocab, tags

def accuracy(tagger, gold_data):
    nr_correct = 0
    nr_words = 0

    for sentence in gold_data:
        words = [tokens[0] for tokens in sentence]

        nr_words += len(words)

        correct_tags = [tokens[1] for tokens in sentence]
        predicted_tags = tagger.predict(words)

        for i in range(len(words)):
            if predicted_tags[i] == correct_tags[i]:
                nr_correct += 1

    acc = nr_correct / nr_words

    return acc

def uas(parser, gold_data):
    nr_correct = 0
    nr_words = 0

    for sentence in gold_data:
        words = [tokens[0] for tokens in sentence]
        tags = [tokens[1] for tokens in sentence]
        correct_head = [tokens[2] for tokens in sentence]
        # Do not include pseudo-root
        nr_words += (len(words) - 1)

        predicted_head = parser.predict(words, tags)

        # skip pseudo-root
        for i in range(1, len(words)):
            if predicted_head[i] == correct_head[i]:
                nr_correct += 1

    acc = nr_correct / nr_words
    return acc

def find_highest_move(scores, legal_transitions):
    _, sorted_indexes = torch.sort(scores, descending=True)
    # find valid move with highest score (SH, LA, RA)
    if len(legal_transitions) > 0:
        sorted_move_list = sorted_indexes.tolist()[0]
        # choose first valid move as default move
        t_p = legal_transitions[0]
        for move in sorted_move_list:
            if move in legal_transitions:
                return t_p