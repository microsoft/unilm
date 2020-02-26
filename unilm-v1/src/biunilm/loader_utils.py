from random import randint, shuffle
from random import random as rand
import numpy as np

import torch
import torch.utils.data


def get_random_word(vocab_words):
    i = randint(0, len(vocab_words)-1)
    return vocab_words[i]


def batch_list_to_batch_tensors(batch):
    batch_tensors = []
    for x in zip(*batch):
        if x[0] is None:
            batch_tensors.append(None)
        elif isinstance(x[0], torch.Tensor):
            batch_tensors.append(torch.stack(x))
        else:
            batch_tensors.append(torch.tensor(x, dtype=torch.long))
    return batch_tensors


class TrieNode(object):
    def __init__(self):
        self.children = {}
        self.is_leaf = False

    def try_get_children(self, key):
        if key not in self.children:
            self.children[key] = TrieNode()
        return self.children[key]


class TrieTree(object):
    def __init__(self):
        self.root = TrieNode()

    def add(self, tokens):
        r = self.root
        for token in tokens:
            r = r.try_get_children(token)
        r.is_leaf = True

    def get_pieces(self, tokens, offset):
        pieces = []
        r = self.root
        token_id = 0
        last_valid = 0
        match_count = 0
        while last_valid < len(tokens):
            if token_id < len(tokens) and tokens[token_id] in r.children:
                r = r.children[tokens[token_id]]
                match_count += 1
                if r.is_leaf:
                    last_valid = token_id
                token_id += 1
            else:
                pieces.append(
                    list(range(token_id - match_count + offset, last_valid + 1 + offset)))
                last_valid += 1
                token_id = last_valid
                r = self.root
                match_count = 0

        return pieces


def _get_word_split_index(tokens, st, end):
    split_idx = []
    i = st
    while i < end:
        if (not tokens[i].startswith('##')) or (i == st):
            split_idx.append(i)
        i += 1
    split_idx.append(end)
    return split_idx


def _expand_whole_word(tokens, st, end):
    new_st, new_end = st, end
    while (new_st >= 0) and tokens[new_st].startswith('##'):
        new_st -= 1
    while (new_end < len(tokens)) and tokens[new_end].startswith('##'):
        new_end += 1
    return new_st, new_end


class Pipeline():
    """ Pre-process Pipeline Class : callable """

    def __init__(self):
        super().__init__()
        self.skipgram_prb = None
        self.skipgram_size = None
        self.pre_whole_word = None
        self.mask_whole_word = None
        self.word_subsample_prb = None
        self.sp_prob = None
        self.pieces_dir = None
        self.vocab_words = None
        self.pieces_threshold = 10
        self.trie = None
        self.call_count = 0
        self.offline_mode = False
        self.skipgram_size_geo_list = None
        self.span_same_mask = False

    def init_skipgram_size_geo_list(self, p):
        if p > 0:
            g_list = []
            t = p
            for _ in range(self.skipgram_size):
                g_list.append(t)
                t *= (1-p)
            s = sum(g_list)
            self.skipgram_size_geo_list = [x/s for x in g_list]

    def create_trie_tree(self, pieces_dir):
        print("sp_prob = {}".format(self.sp_prob))
        print("pieces_threshold = {}".format(self.pieces_threshold))
        if pieces_dir is not None:
            self.trie = TrieTree()
            pieces_files = [pieces_dir]
            for token in self.vocab_words:
                self.trie.add([token])
            for piece_file in pieces_files:
                print("Load piece file: {}".format(piece_file))
                with open(piece_file, mode='r', encoding='utf-8') as reader:
                    for line in reader:
                        parts = line.split('\t')
                        if int(parts[-1]) < self.pieces_threshold:
                            pass
                        tokens = []
                        for part in parts[:-1]:
                            tokens.extend(part.split(' '))
                        self.trie.add(tokens)

    def __call__(self, instance):
        raise NotImplementedError

    # pre_whole_word: tokenize to words before masking
    # post whole word (--mask_whole_word): expand to words after masking
    def get_masked_pos(self, tokens, n_pred, add_skipgram=False, mask_segment=None, protect_range=None):
        if self.pieces_dir is not None and self.trie is None:
            self.create_trie_tree(self.pieces_dir)
        if self.pre_whole_word:
            if self.trie is not None:
                pieces = self.trie.get_pieces(tokens, 0)

                new_pieces = []
                for piece in pieces:
                    if len(new_pieces) > 0 and tokens[piece[0]].startswith("##"):
                        new_pieces[-1].extend(piece)
                    else:
                        new_pieces.append(piece)
                del pieces
                pieces = new_pieces

                pre_word_split = list(_[-1] for _ in pieces)
                pre_word_split.append(len(tokens))
            else:
                pre_word_split = _get_word_split_index(tokens, 0, len(tokens))
            index2piece = None
        else:
            pre_word_split = list(range(0, len(tokens)+1))

            if self.trie is not None:
                pieces = self.trie.get_pieces(tokens, 0)

                index2piece = {}
                for piece in pieces:
                    for index in piece:
                        index2piece[index] = (piece[0], piece[-1])
            else:
                index2piece = None

        span_list = list(zip(pre_word_split[:-1], pre_word_split[1:]))

        # candidate positions of masked tokens
        cand_pos = []
        special_pos = set()
        if mask_segment:
            for i, sp in enumerate(span_list):
                sp_st, sp_end = sp
                if (sp_end-sp_st == 1) and tokens[sp_st].endswith('SEP]'):
                    segment_index = i
                    break
        for i, sp in enumerate(span_list):
            sp_st, sp_end = sp
            if (sp_end-sp_st == 1) and (tokens[sp_st].endswith('CLS]') or tokens[sp_st].endswith('SEP]')):
                special_pos.add(i)
            else:
                if mask_segment:
                    if ((i < segment_index) and ('a' in mask_segment)) or ((i > segment_index) and ('b' in mask_segment)):
                        cand_pos.append(i)
                else:
                    cand_pos.append(i)
        shuffle(cand_pos)

        masked_pos = set()
        for i_span in cand_pos:
            if len(masked_pos) >= n_pred:
                break
            cand_st, cand_end = span_list[i_span]
            if len(masked_pos)+cand_end-cand_st > n_pred:
                continue
            if any(p in masked_pos for p in range(cand_st, cand_end)):
                continue

            n_span = 1
            if index2piece is not None:
                p_start, p_end = index2piece[i_span]
                if p_start < p_end and (rand() < self.sp_prob):
                    # n_span = p_end - p_start + 1
                    st_span, end_span = p_start, p_end + 1
                else:
                    st_span, end_span = i_span, i_span + 1
            else:
                rand_skipgram_size = 0
                # ngram
                if self.skipgram_size_geo_list:
                    # sampling ngram size from geometric distribution
                    rand_skipgram_size = np.random.choice(
                        len(self.skipgram_size_geo_list), 1, p=self.skipgram_size_geo_list)[0] + 1
                else:
                    if add_skipgram and (self.skipgram_prb > 0) and (self.skipgram_size >= 2) and (rand() < self.skipgram_prb):
                        rand_skipgram_size = min(
                            randint(2, self.skipgram_size), len(span_list)-i_span)
                for n in range(2, rand_skipgram_size+1):
                    tail_st, tail_end = span_list[i_span+n-1]
                    if (tail_end-tail_st == 1) and (tail_st in special_pos):
                        break
                    if len(masked_pos)+tail_end-cand_st > n_pred:
                        break
                    n_span = n
                st_span, end_span = i_span, i_span + n_span

            if self.mask_whole_word:
                # pre_whole_word==False: position index of span_list is the same as tokens
                st_span, end_span = _expand_whole_word(
                    tokens, st_span, end_span)

            # subsampling according to frequency
            if self.word_subsample_prb:
                skip_pos = set()
                if self.pre_whole_word:
                    w_span_list = span_list[st_span:end_span]
                else:
                    split_idx = _get_word_split_index(
                        tokens, st_span, end_span)
                    w_span_list = list(
                        zip(split_idx[:-1], split_idx[1:]))
                for i, sp in enumerate(w_span_list):
                    sp_st, sp_end = sp
                    if sp_end-sp_st == 1:
                        w_cat = tokens[sp_st]
                    else:
                        w_cat = ''.join(tokens[sp_st:sp_end])
                    if (w_cat in self.word_subsample_prb) and (rand() < self.word_subsample_prb[w_cat]):
                        for k in range(sp_st, sp_end):
                            skip_pos.add(k)
            else:
                skip_pos = None

            for sp in range(st_span, end_span):
                for mp in range(span_list[sp][0], span_list[sp][1]):
                    if not(skip_pos and (mp in skip_pos)) and (mp not in special_pos) and not(protect_range and (protect_range[0] <= mp < protect_range[1])):
                        masked_pos.add(mp)

        if len(masked_pos) < n_pred:
            shuffle(cand_pos)
            for pos in cand_pos:
                if len(masked_pos) >= n_pred:
                    break
                if pos not in masked_pos:
                    masked_pos.add(pos)
        masked_pos = list(masked_pos)
        if len(masked_pos) > n_pred:
            # shuffle(masked_pos)
            masked_pos = masked_pos[:n_pred]
        return masked_pos

    def replace_masked_tokens(self, tokens, masked_pos):
        if self.span_same_mask:
            masked_pos = sorted(list(masked_pos))
        prev_pos, prev_rand = None, None
        for pos in masked_pos:
            if self.span_same_mask and (pos-1 == prev_pos):
                t_rand = prev_rand
            else:
                t_rand = rand()
            if t_rand < 0.8:  # 80%
                tokens[pos] = '[MASK]'
            elif t_rand < 0.9:  # 10%
                tokens[pos] = get_random_word(self.vocab_words)
            prev_pos, prev_rand = pos, t_rand
