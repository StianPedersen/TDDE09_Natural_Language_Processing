# Skeleton code for lab L1X

# Do not use any imports other than the ones already in this file.

import numpy as np


# The next four functions are adapted from CBOW_Classifier.ipynb


def load_data(filename, max_length=20):
    items = []
    with open(filename, 'rt', encoding='utf-8') as fp:
        for line in fp:
            sentence, label = line.rstrip().split('\t')
            items.append((sentence.split()[:max_length], int(label)))
    return items


def make_vocab(data):
    vocab = {'<pad>': 0}
    for sentence, label in data:
        for t in sentence:
            if t not in vocab:
                vocab[t] = len(vocab)
    return vocab


def vectorize(vocab, data):
    max_length = max(len(s) for s, _ in data)
    xs = []
    ys = []
    for sentence, label in data:
        x = [0] * max_length
        for i, w in enumerate(sentence):
            if w in vocab:
                x[i] = vocab[w]
        xs.append(x)
        ys.append(label)
    return np.array(xs), np.array(ys)


def accuracy(y_pred, y):
    return np.mean((y_pred == y).astype(np.float))


def minibatches(x, y, batch_size):
    random_indices = np.random.permutation(x.shape[0])
    for i in range(0, x.shape[0] - batch_size + 1, batch_size):
        batch_indices = random_indices[i:i+batch_size]
        yield x[batch_indices], y[batch_indices]


# Efficient implementation of the softmax function; see
# http://stackoverflow.com/a/39558290

def softmax(X):
    E = np.exp(X - np.max(X, axis=1, keepdims=True))
    return E / E.sum(axis=1, keepdims=True)


# Your code below

class CBOW(object):

    def __init__(self, num_embeddings, output_dim, embedding_dim=30):
        # TODO: Replace the next lines with your own code
        self.E = np.zeros((num_embeddings, embedding_dim))
        self.W = np.zeros((embedding_dim, output_dim))
        self.b = np.zeros((1, output_dim))

    def forward(self, features):
        # TODO: Replace the next line with your own code
        return np.zeros((len(features), self.W.shape[1]))


def train(vocab, train_x, train_y, n_epochs=20, batch_size=24, lr=1e-1):
    # TODO: Replace the next line with your own code
    return CBOW(len(vocab), 5)


# MAIN METHOD


def main():
    import sys

    train_data = load_data('sst-5-train.txt')
    dev_data = load_data('sst-5-dev.txt')

    vocab = make_vocab(train_data)
    train_x, train_y = vectorize(vocab, train_data)
    dev_x, dev_y = vectorize(vocab, dev_data)

    accuracies = []
    for _ in range(10):
        model = train(vocab, train_x, train_y)
        dev_output = model.forward(dev_x)
        dev_y_pred = np.argmax(dev_output, axis=1)
        accuracies.append(accuracy(dev_y_pred, dev_y))
    print('Average accuracy: {:.4f}'.format(sum(accuracies) / len(accuracies)))


if __name__ == '__main__':
    main()
