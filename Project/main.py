import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
from tqdm import tqdm

from utils import device, make_vocabs, Dataset, TaggedDataset, accuracy, uas, COLOR
from tagger import FixedWindowTagger, FixedWindowTaggerModel
from train import train_parser_static, train_parser_dynamic, train_fixed_window_tagger

class Pipeline():

    def __init__(self, alg, oracle, data):
        self.alg = alg
        self.oracle = oracle
        self.data = data

    def tag_data(self):
        print("Hello NLP World!")
        print('Device:', device, ' and Torch Version:', torch.__version__)

        # Load the training and development data
        train_data = Dataset('data/no_nynorsk-ud-train-projectivized.conllu')
        dev_data = Dataset('data/no_nynorsk-ud-dev-projectivized.conllu')

        # Create the vocabularies
        vocab, tags = make_vocabs(train_data)
        print('Words vocab len: ', len(vocab))
        print('Tags vocab len: ', len(tags))

  
        # Train the tagger and do prediction
        # print('Training the tagger:')
        # tagger = train_fixed_window_tagger(train_data)
        # print('Tagging accuracy: {:.4f}'.format(accuracy(tagger, dev_data)))
        # torch.save(tagger.model, "latin_model")
        # Import pre trained tagger
        print("Importing pretrained tagger")
        tagger = FixedWindowTagger(vocab, tags)
        tagger.model = torch.load('nynorsk_tagger_model', map_location=device)
        print('Tagger accuracy: {:.4f}'.format(accuracy(tagger, dev_data)))

        if self.data == 'retagged':
            # Use tagger to create predicted part-of-speech tags dataset for parser
            print('Use trained tagger to create predicted part-of-speech tags dataset for parser')
            with open('data/no_nynorsk-ud-train-projectivized-retagged.conllu', 'wt', encoding="utf-8") as target:
                for sentence in TaggedDataset('data/no_nynorsk-ud-train-projectivized.conllu'):
                    words = [columns[1] for columns in sentence]
                    for i, t in enumerate(tagger.predict(words)):
                        sentence[i][3] = t
                    for columns in sentence:
                        print('\t'.join(c for c in columns), file=target)
                    print(file=target)

            with open('data/no_nynorsk-ud-dev-retagged.conllu', 'wt', encoding="utf-8") as target:
                for sentence in TaggedDataset('data/no_nynorsk-ud-dev.conllu'):
                    words = [columns[1] for columns in sentence]
                    for i, t in enumerate(tagger.predict(words)):
                        sentence[i][3] = t
                    for columns in sentence:
                        print('\t'.join(c for c in columns), file=target)
                    print(file=target)
            print('Predicted part-of-speech tags dataset for parser is ready!')

    def benchmark(self):
        # Load gold or retagged training and development data
        if self.data == 'gold':
            train_data = Dataset('data/no_nynorsk-ud-train-projectivized.conllu')
            dev_data = Dataset('data/no_nynorsk-ud-dev-projectivized.conllu')
        else:
            train_data = Dataset(
                'data/no_nynorsk-ud-train-projectivized-retagged.conllu')
            dev_data = Dataset('data/no_nynorsk-ud-dev-retagged.conllu')

        # Train the parser and do prediction
        print(f'{COLOR["bY"]}Training parser ({self.alg}, {self.oracle} , {self.data}):{COLOR["C"]}')

        if self.oracle == 'static': 
            parser = train_parser_static(train_data, alg=self.alg, n_epochs=1)
        elif self.oracle == 'dynamic':
            parser = train_parser_dynamic(train_data, alg=self.alg, n_epochs=1)

        print(f'{COLOR["bG"]}Evaluating...{COLOR["C"]}')
        print('Parser UAS: {:.4f}'.format(uas(parser, dev_data)))


if __name__ == "__main__":
    # define flags
    arg_parse = argparse.ArgumentParser(description="Available flags",
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    arg_parse.add_argument("-g", "--gold", action="store_true", help="gold-standard dataset")

    # fetch arguments
    args = arg_parse.parse_args()
    config = vars(args)

    if config['gold'] == True:
        data = 'gold'
    else:
        data = 'retagged'

    # Arc-standard with static oracle
    x1 = Pipeline('standard', 'static', data)

    # Re-tag data, doing it only once saves time between benchmarks
    x1.tag_data()

    x1.benchmark()
    
    # Arc-hybrid with static oracle
    x2 = Pipeline('hybrid', 'static', data).benchmark()

    # the dynamic oracle for the arc-standard algorithm requires a dynamic programming algorithm, so it is skipped 
    # x3 = Pipeline('standard', 'dynamic', data).benchmark()

    # Arc-hybrid with dynamic oracle
    x4 = Pipeline('hybrid', 'dynamic', data).benchmark()