import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

PAD = '<pad>'
UNK = '<unk>'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
                            tmp.append((columns[1], columns[3], int(columns[6])))
                    else:
                        yield tmp
                        tmp = [Dataset.ROOT]

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
        self.hidden = nn.Linear(n_words * word_dim + n_tags * tag_dim, hidden_dim) # 3 * 50 + 1 * 10,

        # Create RELU
        self.activation = nn.ReLU()

        # Create output layers
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, features):
        batch_size = len(features)
        
        # Extract words and tags 
        words = features[:,:-1]
        tags = features[:,-1]

        # Get the word and tag embeddings
        word_embs = self.embeddings['word_embs'](words) # 3 * 50
        tag_embs = self.embeddings['tag_embs'](tags) # 1 * 10
        
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
        embedding_specs = [(3, len(vocab_words), word_dim), (1, len(vocab_tags), tag_dim)]
        self.model = FixedWindowTaggerModel(embedding_specs, hidden_dim, len(vocab_tags)).to(device)
        self.vocab_words = vocab_words
        self.vocab_tags = vocab_tags

    def featurize(self, words, i, pred_tags):
        feature = []
        if len(words) == 1:
            feature = [words[i], 0, 0, 0]

        elif i == 0: # first word
            # Wi, PAD, PAD, PAD
            feature = [words[i], words[i+1], 0, 0]
        elif i == len(words)-1: # last word
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

def training_examples_tagger(vocab_words, vocab_tags, gold_data, tagger, batch_size=100):
    batch = []
    gold_label = []
    sentence_idx = 0
    for sentence in gold_data:
        sentence_idx += 1
        all_words_idx = []
        all_tags_idx = []

        for word, tag, _ in sentence:
            all_words_idx.append(vocab_words[word])
            all_tags_idx.append(vocab_tags[tag])

        for i in range(0, len(all_words_idx)):
            batch.append(tagger.featurize(all_words_idx, i, all_tags_idx))
            gold_label.append(all_tags_idx[i])

            # Yield batch
            if len(batch) == batch_size:
                batch_tensor = torch.Tensor(batch_size, 4).long().to(device)
                bx = torch.cat(batch, out=batch_tensor).to(device)
                by = torch.Tensor(gold_label).long().to(device)
                yield bx, by
                batch = []
                gold_label = []

        # Yield remaining batch
        if sentence_idx == len(list(gold_data))-1:
            remainder = len(batch)
            batch_tensor = torch.Tensor(remainder, 4).long().to(device)
            bx = torch.cat(batch, out=batch_tensor).to(device)
            by = torch.Tensor(gold_label).long().to(device)
            yield bx, by

def var_init(model, std=0.01):
    for name, param in model.named_parameters():
        param.data.normal_(mean=0.0, std=std)

def train_fixed_window_tagger(train_data, n_epochs=1, batch_size=100, lr=1e-2):
    vocab_words, vocab_tags =  make_vocabs(train_data)

    tagger = FixedWindowTagger(vocab_words, vocab_tags)
    
    # Initialize embedding weights
    var_init(tagger.model)

    optimizer = optim.Adam(tagger.model.parameters(), lr=lr)

    nr_words = 0

    for sentence in train_data:
        words = [tokens[0] for tokens in sentence]
        nr_words += len(words)

    try:    
        for epoch in range(n_epochs):
            # Begin training
            with tqdm(total=nr_words) as pbar:
                batch = 0
                tagger.model.train()
                for bx, by in training_examples_tagger(vocab_words, vocab_tags, train_data, tagger, batch_size):
                    curr_batch_size = len(bx)

                    score = tagger.model.forward(bx)
                    optimizer.zero_grad()
                    loss = F.cross_entropy(score, by)
                    loss.backward()
                    optimizer.step()

                    pbar.set_postfix(loss=(loss.item()), batch=batch+1)
                    pbar.update(curr_batch_size)
                    batch += 1
                
    except KeyboardInterrupt:
        pass
    
    return tagger

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

def uas(parser, gold_data):
    nr_correct = 0
    nr_words = 0

    for sentence in gold_data:
        words = [tokens[0] for tokens in sentence]
        tags = [tokens[1] for tokens in sentence]
        # Do not include pseudo-root
        nr_words += (len(words) - 1)

        correct_head = [tokens[2] for tokens in sentence]
        predicted_head = parser.predict(words, tags)

        # skip pseudo-root
        for i in range(1, len(words)):
            if predicted_head[i] == correct_head[i]:
                nr_correct += 1

    acc = nr_correct / nr_words
    return acc

class Parser(object):

    def predict(self, words, tags):
        raise NotImplementedError

class ArcStandardParser(Parser):

    MOVES = tuple(range(3))

    SH, LA, RA = MOVES  # Parser moves are specified as integers.

    @staticmethod
    def initial_config(num_words):
        return (0, [], [0] * num_words)

    @staticmethod
    def valid_moves(config):
        # TODO: Replace the next line with your own code
        valid_moves = []
        buffer, stack, heads = config

        if buffer < len(heads):
            valid_moves.append(ArcStandardParser.SH)
        if len(stack) > 2:
            valid_moves.append(ArcStandardParser.LA)
        if len(stack) > 1:
            valid_moves.append(ArcStandardParser.RA)
        return valid_moves

    @staticmethod
    def next_config(config, move):
        buffer, stack, heads = config
        # SHIFT
        if move == ArcStandardParser.SH:
            stack.append(buffer)
            buffer += 1
        # LEFT ARC
        elif move == ArcStandardParser.LA:
            heads[stack[-2]] = stack[-1]
            top = stack[-1]
            stack = stack[:-2] 
            stack.append(top)
        # RIGHT ARC
        elif move == ArcStandardParser.RA:
            heads[stack[-1]] = stack[-2]
            stack = stack[:-1]
            
        return (buffer, stack, heads)

    @staticmethod
    def is_final_config(config):
        buffer, stack, heads = config
        return buffer == len(heads) and len(stack) == 1 and stack[0] == 0

def oracle_moves(gold_heads):
    parser = ArcStandardParser()
    config = parser.initial_config(len(gold_heads))
    buffer, stack, heads = config
    SH, LA, RA = parser.MOVES
    dependants = {}

    # For each word, count how many other words are dependant on it
    for head in gold_heads:
        if head not in dependants:
            dependants[head] = 1    
        else:
            dependants[head] += 1
    
    # If we haven't reached our final configuration, keep looking
    while not parser.is_final_config(config):
        if len(stack) >= 2:
            top = stack[-1]
            second_top = stack[-2]
            
            # LEFT ARC
            # Does the top of the stack match the gold_head[second_top] and does the 2nd top not have any dependants left?
            # Since second_top will be pushed off the stack, we need to have processed all of it's dependants
            if top == gold_heads[second_top] and dependants.get(second_top, 0) == 0:
                yield config, LA
                config = parser.next_config(config, LA)
                buffer, stack, heads = config
                dependants[top] -= 1 # 1 dependant processed

            # RIGHT ARC
            # Does the second_top of the stack match the gold_head[top] and does the top not have any dependants left?
            # Since top will be pushed off the stack, we need to have processed all of it's dependants
            elif second_top == gold_heads[top] and dependants.get(top, 0) == 0:
                yield config, RA
                config = parser.next_config(config, RA)
                buffer, stack, heads = config
                dependants[second_top] -= 1 # 1 dependant processed
            
            # SHIFT
            # If neither LA or RA is the right move we have to keep shifting
            else:
                yield config, SH
                config = parser.next_config(config, SH)
        
        # SHIFT
        # Shift more words from buffer onto the stack
        else:
            yield config, SH
            config = parser.next_config(config, SH)

class FixedWindowParserModel(nn.Module):

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
        self.embeddings = nn.ModuleDict([['word_embs', nn.Embedding(vocab_size, word_dim, padding_idx=0)],
                                         ['tag_embs', nn.Embedding(tags_size, tag_dim, padding_idx=0)]])

        # Create hidden layers
        self.hidden = nn.Linear(n_words * word_dim + n_tags * tag_dim, hidden_dim) # 3 * 50 + 3 * 10,

        # Create ReLU
        self.activation = nn.ReLU()

        # Create output layers
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, features):
        batch_size = len(features)
        
        # Extract words and tags for buffer 1, stack 1, stack 2
        words, tags = torch.split(features, 3, dim=1)

        # Get the word and tag embeddings
        word_embs = self.embeddings['word_embs'](words) # 3 * 50
        tag_embs = self.embeddings['tag_embs'](tags) # 3 * 10
        
        concat_words = word_embs.view(batch_size, -1)
        concat_tags = tag_embs.view(batch_size, -1)
        
        concat_embs = torch.cat([concat_words, concat_tags], dim=1)

        hidden = self.hidden(concat_embs)

        relu = self.activation(hidden)

        output = self.output(relu)

        return output

class FixedWindowParser(ArcStandardParser):

    def __init__(self, vocab_words, vocab_tags, word_dim=50, tag_dim=10, hidden_dim=180):
        num_moves = len(ArcStandardParser.MOVES)
        embedding_specs = [(3, len(vocab_words), word_dim), (3, len(vocab_tags), tag_dim)]
        self.model = FixedWindowParserModel(embedding_specs, hidden_dim, num_moves).to(device)
        self.vocab_words = vocab_words
        self.vocab_tags = vocab_tags

    def featurize(self, words, tags, config):
        buffer, stack, heads = config

        # stack might be empty or not have enough words, set words and tags to PAD
        word_2 = self.vocab_words[PAD]
        tag_2 = self.vocab_tags[PAD]
        word_3 = self.vocab_words[PAD]
        tag_3 = self.vocab_tags[PAD]

        if buffer < len(heads):
            word_1 = words[buffer]
            tag_1 = tags[buffer]
        else:
            word_1 = self.vocab_words[PAD]
            tag_1 = self.vocab_tags[PAD]
        
        if len(stack) >= 2 and len(stack) <= len(words):
            word_2 = words[stack[-1]]
            tag_2 = tags[stack[-1]]
            word_3 = words[stack[-2]]
            tag_3 = tags[stack[-2]]

        elif len(stack) == 1:
            word_2 = words[stack[-1]]
            tag_2 = tags[stack[-1]]

        # next word in buffer, topmost word on stack, 2nd topmost word on stack,
        # tag of next word in buffer, tag of topmost word on stack, tag of 2nd topmost word on stack
        feature = [word_1, word_2, word_3, tag_1, tag_2, tag_3]
        return torch.tensor([feature]).to(device)

    def predict(self, words, tags):
        # find word indexes for given words
        words_idxs = []
        for word in words:
            if word in self.vocab_words:
                words_idxs.append(self.vocab_words[word])
            else:
                words_idxs.append(self.vocab_words[UNK])

        # find tag indexes for given tags
        tags_idxs = []
        for tag in tags:
            if tag in self.vocab_tags:
                tags_idxs.append(self.vocab_tags[tag])
            else:
                tags_idxs.append(self.vocab_tags[PAD])

        config = self.initial_config(len(words))

        while not self.is_final_config(config):
            valid_moves = self.valid_moves(config)
            feature = self.featurize(words_idxs, tags_idxs, config)
            pred_moves = self.model.forward(feature)
            _, sorted_indexes = torch.sort(pred_moves, descending=True)
            # find valid move with highest score (SH, LA, RA)
            if len(valid_moves) > 0:
                sorted_move_list = sorted_indexes.tolist()[0]
                # choose first valid move as default move
                new_move = valid_moves[0]
                for move in sorted_move_list:
                    if move in valid_moves:
                        new_move = move
                        break
                config = self.next_config(config, new_move)

        return config[2]

def training_examples_parser(vocab_words, vocab_tags, gold_data, parser, batch_size=100):
    batch = []
    moves = []
    sentence_idx = 0
    for sentence in gold_data:
        sentence_idx += 1
        all_words_idx = []
        all_tags_idx = []
        all_heads = []

        for word, tag, head in sentence:
            all_words_idx.append(vocab_words[word])
            all_tags_idx.append(vocab_tags[tag])
            all_heads.append(head)

        for c, m in oracle_moves(all_heads):
            batch.append(parser.featurize(all_words_idx, all_tags_idx, c))
            moves.append(m)

            # Yield batch
            if len(batch) == batch_size:
                batch_tensor = torch.Tensor(batch_size, 6).long().to(device)
                bx = torch.cat(batch, out=batch_tensor).to(device)
                by = torch.Tensor(moves).long().to(device)
                yield bx, by
                batch = []
                moves = []

    # Yield remaining batch
    if sentence_idx == len(list(gold_data))-1:
        remainder = len(batch)
        batch_tensor = torch.Tensor(remainder, 6).long().to(device)
        bx = torch.cat(batch, out=batch_tensor).to(device)
        by = torch.Tensor(moves).long().to(device)
        yield bx, by

def train_fixed_window_parser(train_data, n_epochs=1, batch_size=100, lr=1e-2):
    vocab_words, vocab_tags =  make_vocabs(train_data)

    parser = FixedWindowParser(vocab_words, vocab_tags)

    optimizer = optim.Adam(parser.model.parameters(), lr=lr, weight_decay=1e-5)

    nr_examples = 421700

    try:    
        for epoch in range(n_epochs):
            # Begin training
            with tqdm(total=nr_examples) as pbar:
                batch = 1
                train_loss = 0

                parser.model.train()
                for bx, by in training_examples_parser(vocab_words, vocab_tags, train_data, parser, batch_size):
                    curr_batch_size = len(bx)

                    score = parser.model.forward(bx)
                    optimizer.zero_grad()
                    loss = F.cross_entropy(score, by)
                    train_loss += loss.item()
                    loss.backward()
                    optimizer.step()

                    pbar.set_postfix(loss=(train_loss/batch), batch=batch)
                    pbar.update(curr_batch_size)
                    batch += 1
                
    except KeyboardInterrupt:
        pass
    
    return parser

def main():
    print("Hello NLP World!")
    print('Device:', device, ' and Version:', torch.__version__)

    # We load the training data and the development data
    train_data = Dataset('en_ewt-ud-train-projectivized.conllu')
    dev_data = Dataset('en_ewt-ud-dev.conllu')

    # Create the vocabularies
    vocab, tags = make_vocabs(train_data)
    print('Words vocab len', len(vocab))
    print('Tags vocab len',len(tags))

    # Train the tagger and do prediction
    tagger = train_fixed_window_tagger(train_data)
    print('{:.4f}'.format(accuracy(tagger, dev_data)))

    # Use tagger to create predicted part-of-speech tags dataset for parser
    with open('en_ewt-ud-train-projectivized-retagged.conllu', 'wt') as target:
        for sentence in TaggedDataset('en_ewt-ud-train-projectivized.conllu'):
            words = [columns[1] for columns in sentence]
            for i, t in enumerate(tagger.predict(words)):
                sentence[i][3] = t
            for columns in sentence:
                print('\t'.join(c for c in columns), file=target)
            print(file=target)
    
    with open('en_ewt-ud-dev-retagged.conllu', 'wt') as target:
        for sentence in TaggedDataset('en_ewt-ud-dev.conllu'):
            words = [columns[1] for columns in sentence]
            for i, t in enumerate(tagger.predict(words)):
               sentence[i][3] = t
            for columns in sentence:
              print('\t'.join(c for c in columns), file=target)
            print(file=target)
    
    train_data_retaged = Dataset('en_ewt-ud-train-projectivized-retagged.conllu')
    dev_data_retaged = Dataset('en_ewt-ud-dev-retagged.conllu')

    # Train the parser and do prediction
    parser = train_fixed_window_parser(train_data_retaged, n_epochs=1)
    print('{:.4f}'.format(uas(parser, dev_data_retaged)))

if __name__ == "__main__":
    main()