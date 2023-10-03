import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

from tqdm import tqdm

from utils import make_vocabs, find_highest_move
from tagger import FixedWindowTagger
from neural_parser import standard_static_oracle, hybrid_static_oracle, dynamic_oracle
from neural_parser import FixedWindowParser, FixedWindowParserHybrid, ArcHybridParser, ArcStandardParser

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


def train_fixed_window_tagger(train_data, n_epochs=1, batch_size=100, lr=1e-2):
    vocab_words, vocab_tags = make_vocabs(train_data)

    tagger = FixedWindowTagger(vocab_words, vocab_tags)

    optimizer = optim.Adam(tagger.model.parameters(), lr=lr)

    nr_iterations = 0

    for sentence in train_data:
        words = [tokens[0] for tokens in sentence]
        nr_iterations += len(words)

    try:
        for epoch in range(n_epochs):
            # Begin training
            with tqdm(total=nr_iterations) as pbar:
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

def training_examples_parser(vocab_words, vocab_tags, gold_data, parser, alg, batch_size=100):
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

        if alg == 'standard':
            oracle = standard_static_oracle(all_heads)
        elif alg == 'hybrid':
            oracle = hybrid_static_oracle(all_heads)

        for c, m in oracle:
            batch.append(parser.featurize(all_words_idx, all_tags_idx, list(c[2]), c))
            moves.append(m)

            # Yield batch
            if len(batch) == batch_size:
                batch_tensor = torch.Tensor(batch_size, 24).long().to(device)
                bx = torch.cat(batch, out=batch_tensor).to(device)
                by = torch.Tensor(moves).long().to(device)
                yield bx, by
                batch = []
                moves = []

    # Yield remaining batch
    if sentence_idx == len(list(gold_data))-1:
        remainder = len(batch)
        batch_tensor = torch.Tensor(remainder, 24).long().to(device)
        bx = torch.cat(batch, out=batch_tensor).to(device)
        by = torch.Tensor(moves).long().to(device)
        yield bx, by

def train_parser_static(train_data, alg, n_epochs=1, batch_size=100, lr=1e-2):
    vocab_words, vocab_tags = make_vocabs(train_data)

    if alg == 'standard':
        parser = FixedWindowParser(vocab_words, vocab_tags)
    elif alg == 'hybrid':
        parser = FixedWindowParserHybrid(vocab_words, vocab_tags)

    optimizer = optim.Adam(parser.model.parameters(), lr=lr)

    nr_iterations = 0

    for sentence in train_data:
        nr_iterations += 2 * len(sentence) - 1

    try:
        for epoch in range(n_epochs):
            # Begin training
            with tqdm(total=nr_iterations) as pbar:
                batch = 1
                train_loss = 0

                parser.model.train()
                for bx, by in training_examples_parser(vocab_words, vocab_tags, train_data, parser, alg, batch_size):
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

def train_parser_dynamic(train_data, alg, n_epochs=1, batch_size=100, lr=1e-2):
    vocab_words, vocab_tags =  make_vocabs(train_data)

    parser = FixedWindowParserHybrid(vocab_words, vocab_tags)

    if alg == 'standard':
        arc_parser = ArcStandardParser()
    elif alg == 'hybrid':
        arc_parser = ArcHybridParser()

    optimizer = optim.Adam(parser.model.parameters(), lr=lr)

    nr_iterations = 0

    for sentence in train_data:
        nr_iterations += 1

    try:    
        for epoch in range(n_epochs):
            # Begin training
            with tqdm(total=nr_iterations) as pbar:
                total_moves = 1
                train_loss = 0
                batch_loss = []
                batch_iter = 0

                parser.model.train()
                for sentence in train_data:
                    all_words_idx = []
                    all_tags_idx = []
                    all_heads = []

                    for word, tag, head in sentence:
                        all_words_idx.append(vocab_words[word])
                        all_tags_idx.append(vocab_tags[tag])
                        all_heads.append(head)
                    
                    config = arc_parser.initial_config(len(all_heads))

                    while not arc_parser.is_final_config(config):
                        # Get all legal moves
                        legal_transitions = arc_parser.valid_moves(config)

                        # Compute which move should be taken next
                        scores = parser.model.forward(parser.featurize(all_words_idx, all_tags_idx, list(config[2]), config)).to(device)

                        # Get legal move with highest probability
                        t_p = find_highest_move(scores, legal_transitions)

                        # Extract scores to list
                        scores_list = scores.tolist()[0]

                        # Compute which moves are zero cost
                        zero_cost_moves = dynamic_oracle(all_heads, config, legal_transitions, arc_parser)
                        
                        # Get the best legal zero cost move
                        t_o = max(zero_cost_moves, key=lambda p: scores_list[p])
                        # Target vector    
                        y = torch.tensor([t_o]).long().to(device)

                        loss = F.cross_entropy(scores, y)
                        batch_loss.append(loss)
                        train_loss += loss.item()

                        # If predicted transition is not in the zero cost moves, update weights.
                        if t_p not in zero_cost_moves:
                            # choose random transition from zero_cost. Might be bad move but such is life.
                            config = parser.next_config(config, random.choice(zero_cost_moves))
                        else:
                            config = parser.next_config(config, t_p)

                        pbar.set_postfix(loss=(train_loss/total_moves), configs=total_moves)
                        total_moves += 1
                        batch_iter += 1

                        # Update the parameters
                        if len(batch_loss) > 0 and batch_iter >= batch_size:
                            optimizer.zero_grad()
                            loss = sum(batch_loss)
                            loss.backward()
                            optimizer.step()
                            batch_loss = []
                            batch_iter = 0
                    
                    pbar.update(1)
                
    except KeyboardInterrupt:
        pass
    
    return parser