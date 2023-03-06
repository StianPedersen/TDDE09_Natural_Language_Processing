import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm

from utils import PAD, UNK, device

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
        if len(stack) > 1:
            valid_moves.append(ArcStandardParser.LA)
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
    

class ArcHybridParser(Parser):

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
            valid_moves.append(ArcHybridParser.SH)
        if len(stack) > 1 and buffer < len(config[2]):
            valid_moves.append(ArcHybridParser.LA)
        if len(stack) > 1:
            valid_moves.append(ArcHybridParser.RA)
        return valid_moves

    @staticmethod
    def next_config(config, move):
        buffer, stack, heads = config
        # SHIFT
        if move == ArcHybridParser.SH:
            stack.append(buffer)
            buffer += 1
        # LEFT ARC
        elif move == ArcHybridParser.LA:
            heads[stack[-1]] = buffer
            stack = stack[:-1]
        # RIGHT ARC
        elif move == ArcHybridParser.RA:
            heads[stack[-1]] = stack[-2]
            stack = stack[:-1]
            
        return (buffer, stack, heads)

    @staticmethod
    def is_final_config(config):
        buffer, stack, heads = config
        return buffer == len(heads) and len(stack) == 1 and stack[0] == 0
    
    # Buffer = 0
    # stack = 1
    # heads = 2
    @staticmethod
    def zero_cost_shift(current_config, gold_config):
        if current_config[0] == len(current_config[2]):
            return False
        if len(current_config[1]) == 0:
            return True
        item = current_config[0]

        # SH 1
        if item in gold_config:
            for d in current_config[1][0:-1]:
                if item == gold_config[d]:
                    return False

        # SH 2
        for h in current_config[1][0:-1]:
            if h == gold_config[item]:
                return False

        return True
    
    # Buffer = 0
    # stack = 1
    # heads = 2
    @staticmethod
    def zero_cost_la(current_config, gold_config):
        s0 = current_config[1][-1]
        if len(current_config[1]) < 2:
            s1 = None
        else:
            s1 = current_config[1][-2]

        # LA 1
        for buffer_item in range(current_config[0],len(current_config[2])):
            if s0 == gold_config[buffer_item]:
                return False 
    
        # LA 2
        if s1 == gold_config[s0]:
            return False
         
        # LA 3
        for buffer_item in range(current_config[0]+1,len(current_config[2])):
            if buffer_item == gold_config[s0]:
                return False
        return True


    # Buffer = 0
    # stack = 1
    # heads = 2
    @staticmethod
    def zero_cost_ra(current_config, gold_config):
        s0 = current_config[1][-1]

        # RA 1
        for buffer_item in range(current_config[0],len(current_config[2])):
            if s0 == gold_config[buffer_item]:
                return False

        # RA 2
        for buffer_item in range(current_config[0],len(current_config[2])):
            if buffer_item == gold_config[s0]:
                return False
        return True


def standard_static_oracle(gold_heads):
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
                dependants[top] -= 1  # 1 dependant processed

            # RIGHT ARC
            # Does the second_top of the stack match the gold_head[top] and does the top not have any dependants left?
            # Since top will be pushed off the stack, we need to have processed all of it's dependants
            elif second_top == gold_heads[top] and dependants.get(top, 0) == 0:
                yield config, RA
                config = parser.next_config(config, RA)
                buffer, stack, heads = config
                dependants[second_top] -= 1  # 1 dependant processed

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

def hybrid_static_oracle(gold_heads):
    parser = ArcHybridParser()
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
        if len(stack) >= 1:
            top = stack[-1]
            buf = buffer
            # LEFT ARC
            # Does the buffer match the gold_head[top] and does the top of the stack not have any dependants left?
            # Since top will be pushed off the stack, we need to have processed all of it's dependants
            if buf == gold_heads[top] and dependants.get(top, 0) == 0:
                yield config, LA
                config = parser.next_config(config, LA)
                buffer, stack, heads = config
                dependants[buf] -= 1 # 1 dependant processed
            
            elif len(stack) >= 2:
                second_top = stack[-2]
                # RIGHT ARC
                # Does the second_top of the stack match the gold_head[top] and does the top not have any dependants left?
                # Since top will be pushed off the stack, we need to have processed all of it's dependants
                if second_top == gold_heads[top] and dependants.get(top, 0) == 0:
                    yield config, RA
                    config = parser.next_config(config, RA)
                    buffer, stack, heads = config
                    dependants[second_top] -= 1 # 1 dependant processed
                # SHIFT
                # If neither LA or RA is the right move we have to keep shifting
                elif buffer < len(heads):
                    yield config, SH
                    config = parser.next_config(config, SH)
                    buffer, stack, heads = config
            # SHIFT
            # If neither LA or RA is the right move we have to keep shifting
            else:
                yield config, SH
                config = parser.next_config(config, SH)
                buffer, stack, heads = config

        # If neither LA or RA is the right move we have to keep shifting
        else:
            yield config, SH
            config = parser.next_config(config, SH)
            buffer, stack, heads = config


def dynamic_oracle(gold_config, current_config, legal_transition, parser):
    SHIFT, LA, RA = 0,1,2
    moves = []
    if SHIFT in legal_transition and parser.zero_cost_shift(current_config, gold_config):
        moves.append(SHIFT)
    if LA in legal_transition and parser.zero_cost_la(current_config, gold_config):
        moves.append(LA)
    if RA in legal_transition and parser.zero_cost_ra(current_config, gold_config):
        moves.append(RA)
    return moves


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
        self.hidden = nn.Linear(n_words * word_dim + n_tags * tag_dim, hidden_dim)  # 12 * 50 + 12 * 10,

        # Create ReLU
        self.activation = nn.ReLU()

        # Create output layers
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, features):
        batch_size = len(features)

        # Extract words and tags
        words, tags = torch.split(features, 12, dim=1)

        # Get the word and tag embeddings
        word_embs = self.embeddings['word_embs'](words)  # 12 * 50
        tag_embs = self.embeddings['tag_embs'](tags)  # 12 * 10

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
        embedding_specs = [(12, len(vocab_words), word_dim),
                           (12, len(vocab_tags), tag_dim)]
        self.model = FixedWindowParserModel(
            embedding_specs, hidden_dim, num_moves).to(device)
        self.vocab_words = vocab_words
        self.vocab_tags = vocab_tags

    def featurize(self, words, tags, gold_heads, config):
        buffer, stack, heads = config

        s0_w = self.vocab_words[PAD]
        s0_t = self.vocab_tags[PAD]
        s1_w = self.vocab_words[PAD]
        s1_t = self.vocab_tags[PAD]
        s2_w = self.vocab_words[PAD]
        s2_t = self.vocab_tags[PAD]

        b0_w = self.vocab_words[PAD]
        b0_t = self.vocab_tags[PAD]
        b1_w = self.vocab_words[PAD]
        b1_t = self.vocab_tags[PAD]
        b2_w = self.vocab_words[PAD]
        b2_t = self.vocab_tags[PAD]

        if buffer < len(heads):
            b0_w = words[buffer]
            b0_t = tags[buffer]
            if buffer + 1 < len(heads):
                b1_w = words[buffer + 1]
                b1_t = tags[buffer + 1]
                if buffer + 2 < len(heads):
                    b2_w = words[buffer + 2]
                    b2_t = tags[buffer + 2]

        if len(stack) >= 1:
            s0_w = words[stack[-1]]
            s0_t = tags[stack[-1]]
            if len(stack) >= 2:
                s1_w = words[stack[-2]]
                s1_t = tags[stack[-2]]
                if len(stack) >= 3:
                    s2_w = words[stack[-3]]
                    s2_t = tags[stack[-3]]

        s0_b1_w = self.vocab_words[PAD]
        s0_b2_w = self.vocab_words[PAD]
        s0_b1_t = self.vocab_tags[PAD]
        s0_b2_t = self.vocab_tags[PAD]
        for idx, head in enumerate(gold_heads[0:s0_w]):
            if head == s0_w and s0_b1_w == self.vocab_tags[PAD]:
                s0_b1_w = words[idx]
                s0_b1_t = tags[idx]
            if head == s0_w and s0_b2_w == self.vocab_tags[PAD]:
                s0_b2_w = words[idx]
                s0_b2_t = tags[idx]

        s0_f1_w = self.vocab_words[PAD]
        s0_f2_w = self.vocab_words[PAD]
        s0_f1_t = self.vocab_tags[PAD]
        s0_f2_t = self.vocab_tags[PAD]
        if len(stack) >= 1:
            for idx, head in enumerate(gold_heads[s0_w:]):
                if head == s0_w and s0_f1_w == self.vocab_tags[PAD]:
                    s0_f1_w = words[idx]
                    s0_f1_t = tags[idx]
                if head == s0_w and s0_f2_w == self.vocab_tags[PAD]:
                    s0_f2_w = tags[idx]
                    s0_f2_t = tags[idx]

        n0_b1_w = self.vocab_words[PAD]
        n0_b2_w = self.vocab_words[PAD]
        n0_b1_t = self.vocab_tags[PAD]
        n0_b2_t = self.vocab_tags[PAD]
        for idx, head in enumerate(gold_heads[0:b0_w]):
            if head == b0_w and n0_b1_w == self.vocab_tags[PAD]:
                n0_b1_w = words[idx]
                n0_b1_t = tags[idx]
            if head == b0_w and n0_b2_w == self.vocab_tags[PAD]:
                n0_b2_w = words[idx]
                n0_b2_t = tags[idx]

        feature = [b0_w, b1_w, b2_w, s0_w, s1_w, s2_w,
                   s0_b1_w, s0_b2_w, s0_f1_w, s0_f2_w, n0_b1_w, n0_b2_w,
                   b0_t, b1_t, b2_t, s0_t, s1_t, s2_t,
                   s0_b1_t, s0_b2_t, s0_f1_t, s0_f2_t, n0_b1_t, n0_b2_t]
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
            feature = self.featurize(words_idxs, tags_idxs, list(config[2]), config)
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
    

class FixedWindowParserHybrid(ArcHybridParser):

    def __init__(self, vocab_words, vocab_tags, word_dim=50, tag_dim=10, hidden_dim=180):
        num_moves = len(ArcHybridParser.MOVES)
        embedding_specs = [(12, len(vocab_words), word_dim), (12, len(vocab_tags), tag_dim)]
        self.model = FixedWindowParserModel(embedding_specs, hidden_dim, num_moves).to(device)
        self.vocab_words = vocab_words
        self.vocab_tags = vocab_tags

    def featurize(self, words, tags, gold_heads, config):
        buffer, stack, heads = config
    
        s0_w = self.vocab_words[PAD]
        s0_t = self.vocab_tags[PAD]
        s1_w = self.vocab_words[PAD]
        s1_t = self.vocab_tags[PAD]
        s2_w = self.vocab_words[PAD]
        s2_t = self.vocab_tags[PAD]

        b0_w = self.vocab_words[PAD]
        b0_t = self.vocab_tags[PAD]
        b1_w = self.vocab_words[PAD]
        b1_t = self.vocab_tags[PAD]
        b2_w = self.vocab_words[PAD]
        b2_t = self.vocab_tags[PAD]

        if buffer < len(heads):
            b0_w = words[buffer]
            b0_t = tags[buffer]
            if buffer + 1 < len(heads):
                b1_w = words[buffer + 1]
                b1_t = tags[buffer + 1]
                if buffer + 2 < len(heads):
                    b2_w = words[buffer + 2]
                    b2_t = tags[buffer + 2]
        
        if len(stack) >= 1:
            s0_w = words[stack[-1]]
            s0_t = tags[stack[-1]]
            if len(stack) >= 2:
                s1_w = words[stack[-2]]
                s1_t = tags[stack[-2]]
                if len(stack) >= 3:
                    s2_w = words[stack[-3]]
                    s2_t = tags[stack[-3]]
        
        s0_b1_w = self.vocab_words[PAD]
        s0_b2_w = self.vocab_words[PAD]
        s0_b1_t = self.vocab_tags[PAD]
        s0_b2_t = self.vocab_tags[PAD]
        for idx, head in enumerate(gold_heads[0:s0_w]):
            if head == s0_w and s0_b1_w == self.vocab_tags[PAD]:
                s0_b1_w = words[idx]
                s0_b1_t = tags[idx]
            if head == s0_w and s0_b2_w == self.vocab_tags[PAD]:
                s0_b2_w = words[idx]
                s0_b2_t = tags[idx]


        s0_f1_w = self.vocab_words[PAD]
        s0_f2_w = self.vocab_words[PAD]
        s0_f1_t = self.vocab_tags[PAD]
        s0_f2_t = self.vocab_tags[PAD]
        if len(stack) >= 1:
            for idx, head in enumerate(gold_heads[s0_w:]):
                if head == s0_w and s0_f1_w == self.vocab_tags[PAD]:
                    s0_f1_w = words[idx]
                    s0_f1_t = tags[idx]
                if head == s0_w and s0_f2_w == self.vocab_tags[PAD]:
                    s0_f2_w = tags[idx]
                    s0_f2_t = tags[idx]


        n0_b1_w = self.vocab_words[PAD]
        n0_b2_w = self.vocab_words[PAD]
        n0_b1_t = self.vocab_tags[PAD]
        n0_b2_t = self.vocab_tags[PAD]
        for idx, head in enumerate(gold_heads[0:b0_w]):
            if head == b0_w and n0_b1_w == self.vocab_tags[PAD]:
                n0_b1_w = words[idx]
                n0_b1_t = tags[idx]
            if head == b0_w and n0_b2_w == self.vocab_tags[PAD]:
                n0_b2_w = words[idx]
                n0_b2_t = tags[idx]


        feature = [b0_w, b1_w, b2_w, s0_w, s1_w, s2_w,
                   s0_b1_w, s0_b2_w, s0_f1_w, s0_f2_w, n0_b1_w, n0_b2_w,
                   b0_t, b1_t, b2_t, s0_t, s1_t, s2_t,
                   s0_b1_t, s0_b2_t, s0_f1_t, s0_f2_t, n0_b1_t, n0_b2_t]
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
            feature = self.featurize(words_idxs, tags_idxs, list(config[2]), config)
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