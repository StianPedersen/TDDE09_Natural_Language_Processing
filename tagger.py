import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from utils import device, UNK, PAD

class FixedWindowTaggerModel(nn.Module):

    def __init__(self, embedding_specs, hidden_dim, output_dim):
        super().__init__()
        # Extract embedding_specs
        emb_spec_words = embedding_specs[0]
        emb_spec_tags = embedding_specs[1]

        n_words = emb_spec_words[0]
        vocab_size = emb_spec_words[1]
        word_dim = emb_spec_words[2]

        n_tags = emb_spec_tags[0]
        tags_size = emb_spec_tags[1]
        tag_dim = emb_spec_tags[2]

        # Create embeddings
        self.embeddings = nn.ModuleDict([
            ['word_embs', nn.Embedding(vocab_size, word_dim, padding_idx=0)],
            ['tag_embs', nn.Embedding(tags_size, tag_dim, padding_idx=0)]])

        # Create hidden layers
        self.hidden = nn.Linear(
            n_words * word_dim + n_tags * tag_dim, hidden_dim)  # 3 * 50 + 1 * 10,

        # Create RELU
        self.activation = nn.ReLU()

        # Create output layers
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, features):
        batch_size = len(features)

        # Extract words and tags
        words = features[:, :-1]
        tags = features[:, -1]

        # Get the word and tag embeddings
        word_embs = self.embeddings['word_embs'](words)  # 3 * 50
        tag_embs = self.embeddings['tag_embs'](tags)  # 1 * 10

        concat_words = word_embs.view(batch_size, -1)

        concat_embs = torch.cat([concat_words, tag_embs], dim=1)

        hidden = self.hidden(concat_embs)

        relu = self.activation(hidden)

        output = self.output(relu)

        return output

class Tagger(object):

    def predict(self, sentence):
        raise NotImplementedError

class FixedWindowTagger(Tagger):

    def __init__(self, vocab_words, vocab_tags, word_dim=50, tag_dim=10, hidden_dim=100):
        embedding_specs = [(3, len(vocab_words), word_dim),
                           (1, len(vocab_tags), tag_dim)]
        self.model = FixedWindowTaggerModel(
            embedding_specs, hidden_dim, len(vocab_tags)).to(device)
        self.vocab_words = vocab_words
        self.vocab_tags = vocab_tags

    def featurize(self, words, i, pred_tags):
        feature = []
        if len(words) == 1:
            feature = [words[i], 0, 0, 0]

        elif i == 0:  # first word
            # Wi, PAD, PAD, PAD
            feature = [words[i], words[i+1], 0, 0]
        elif i == len(words)-1:  # last word
            # Wi, Wi+1, PAD, PAD
            feature = [words[i], 0, words[i-1], pred_tags[i-1]]
        else:
            # Wi, Wi+1, Wi-1, Ti-1
            feature = [words[i], words[i+1], words[i-1], pred_tags[i-1]]
        return torch.tensor([feature]).to(device)

    def predict(self, words):
        # find word indexes for given words
        words_idxs = []
        for word in words:
            if not word in self.vocab_words:
                words_idxs.append(self.vocab_words[UNK])
            else:
                words_idxs.append(self.vocab_words[word])

        # predict tags
        pred_tags_idxs = [0] * len(words)
        for i in range(0, len(words_idxs)):
            feature = self.featurize(words_idxs, i, pred_tags_idxs)
            pred_tags = self.model.forward(feature)
            # Find tag index with highest probability
            pred_tags_idxs[i] = torch.argmax(pred_tags).item()

        # convert tag indexes
        pred_tags = []
        for tag_idx in pred_tags_idxs:
            tag = [k for k, v in self.vocab_tags.items() if v == tag_idx][0]
            pred_tags.append(tag)

        return pred_tags