import numpy as np

from random import randint, shuffle, choice
from random import random as rand
import math
import logging
import torch
import torch.utils.data


logger = logging.getLogger(__name__)


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

# Input file format :
# 1. One sentence per line. These should ideally be actual sentences,
#    not entire paragraphs or arbitrary spans of text. (Because we use
#    the sentence boundaries for the "next sentence prediction" task).
# 2. Blank lines between documents. Document boundaries are needed
#    so that the "next sentence prediction" task doesn't span between documents.


def truncate_tokens_pair(tokens_a, tokens_b, max_len, max_len_a=0, max_len_b=0, trunc_seg=None, always_truncate_tail=False):
    num_truncated_a = [0, 0]
    num_truncated_b = [0, 0]
    while True:
        if len(tokens_a) + len(tokens_b) <= max_len:
            break
        if (max_len_a > 0) and len(tokens_a) > max_len_a:
            trunc_tokens = tokens_a
            num_truncated = num_truncated_a
        elif (max_len_b > 0) and len(tokens_b) > max_len_b:
            trunc_tokens = tokens_b
            num_truncated = num_truncated_b
        elif trunc_seg:
            # truncate the specified segment
            if trunc_seg == 'a':
                trunc_tokens = tokens_a
                num_truncated = num_truncated_a
            else:
                trunc_tokens = tokens_b
                num_truncated = num_truncated_b
        else:
            # truncate the longer segment
            if len(tokens_a) > len(tokens_b):
                trunc_tokens = tokens_a
                num_truncated = num_truncated_a
            else:
                trunc_tokens = tokens_b
                num_truncated = num_truncated_b
        # whether always truncate source sequences
        if (not always_truncate_tail) and (rand() < 0.5):
            del trunc_tokens[0]
            num_truncated[0] += 1
        else:
            trunc_tokens.pop()
            num_truncated[1] += 1
    return num_truncated_a, num_truncated_b


class Seq2SeqDataset(torch.utils.data.Dataset):
    """ Load sentence pair (sequential or random order) from corpus """

    def __init__(self, file_src, file_tgt, batch_size, tokenizer, max_len, file_oracle=None, short_sampling_prob=0.1, sent_reverse_order=False, bi_uni_pipeline=[]):
        super().__init__()
        self.tokenizer = tokenizer  # tokenize function
        self.max_len = max_len  # maximum length of tokens
        self.short_sampling_prob = short_sampling_prob
        self.bi_uni_pipeline = bi_uni_pipeline
        self.batch_size = batch_size
        self.sent_reverse_order = sent_reverse_order

        # read the file into memory
        self.ex_list = []
        if file_oracle is None:
            with open(file_src, "r", encoding='utf-8') as f_src, open(file_tgt, "r", encoding='utf-8') as f_tgt:
                for src, tgt in zip(f_src, f_tgt):
                    src_tk = tokenizer.tokenize(src.strip())
                    tgt_tk = tokenizer.tokenize(tgt.strip())
                    assert len(src_tk) > 0
                    assert len(tgt_tk) > 0
                    self.ex_list.append((src_tk, tgt_tk))
        else:
            with open(file_src, "r", encoding='utf-8') as f_src, \
                    open(file_tgt, "r", encoding='utf-8') as f_tgt, \
                    open(file_oracle, "r", encoding='utf-8') as f_orc:
                for src, tgt, orc in zip(f_src, f_tgt, f_orc):
                    src_tk = tokenizer.tokenize(src.strip())
                    tgt_tk = tokenizer.tokenize(tgt.strip())
                    s_st, labl = orc.split('\t')
                    s_st = [int(x) for x in s_st.split()]
                    labl = [int(x) for x in labl.split()]
                    self.ex_list.append((src_tk, tgt_tk, s_st, labl))
        print('Load {0} documents'.format(len(self.ex_list)))

    def __len__(self):
        return len(self.ex_list)

    def __getitem__(self, idx):
        instance = self.ex_list[idx]
        proc = choice(self.bi_uni_pipeline)
        instance = proc(instance)
        return instance

    def __iter__(self):  # iterator to load data
        for __ in range(math.ceil(len(self.ex_list) / float(self.batch_size))):
            batch = []
            for __ in range(self.batch_size):
                idx = randint(0, len(self.ex_list)-1)
                batch.append(self.__getitem__(idx))
            # To Tensor
            yield batch_list_to_batch_tensors(batch)


class Seq2seqWithPseudoMaskPipeline(object):
    def __init__(self, vocab_words, indexer, max_source_len=256,
                 max_target_len=128, cls_token='[CLS]', sep_token='[SEP]', two_sep=False):
        self.vocab_words = vocab_words
        self.indexer = indexer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len

        self.cls_token = cls_token
        self.sep_token = sep_token
        self.two_sep = two_sep
        self.target_delta = 2 if two_sep else 1

    def convert_and_pad(self, tokens, max_len, pad_id=0):
        assert len(tokens) <= max_len
        token_ids = self.indexer(tokens)
        return token_ids + [pad_id] * (max_len - len(token_ids))

    def __call__(self, tokens_a, tokens_b):
        if len(tokens_a) > self.max_source_len - 2:
            tokens_a = tokens_a[:self.max_source_len - 2]

        if len(tokens_b) > self.max_target_len - self.target_delta:
            tokens_b = tokens_b[:self.max_target_len - self.target_delta]

        source_tokens = [self.cls_token] + tokens_a + [self.sep_token]

        target_tokens = tokens_b + [self.sep_token]
        if self.two_sep:
            target_tokens = [self.sep_token] + target_tokens

        num_source_tokens = len(source_tokens)
        num_target_tokens = len(target_tokens)

        source_ids = self.convert_and_pad(source_tokens, self.max_source_len)
        target_ids = self.convert_and_pad(target_tokens, self.max_target_len)

        return {
            "source": source_ids,
            "target": target_ids,
            "num_source_tokens": num_source_tokens,
            "num_target_tokens": num_target_tokens,
        }


class Preprocess4Seq2seq(Pipeline):
    """ Pre-processing steps for pretraining transformer """

    def __init__(self, max_pred, mask_prob, vocab_words, indexer, max_len=512,
                 skipgram_prb=0, skipgram_size=0, block_mask=False, mask_whole_word=False,
                 new_segment_ids=False, truncate_config={}, mask_source_words=False, mode="s2s",
                 has_oracle=False, num_qkv=0, s2s_special_token=False, s2s_add_segment=False,
                 s2s_share_segment=False, pos_shift=False, require_pseudo_ids=False,
                 rand_prob=0.1, keep_prob=0.1, cls_token='[CLS]', sep_token='[SEP]', mask_token='[MASK]'):
        super().__init__()
        self.max_len = max_len
        self.max_pred = max_pred  # max tokens of prediction
        self.mask_prob = mask_prob  # masking probability
        self.vocab_words = vocab_words  # vocabulary (sub)words
        self.indexer = indexer  # function from token to token index
        self.max_len = max_len
        self._tril_matrix = torch.tril(torch.ones(
            (max_len, max_len), dtype=torch.long))
        self.skipgram_prb = skipgram_prb
        self.skipgram_size = skipgram_size
        self.mask_whole_word = mask_whole_word
        self.new_segment_ids = new_segment_ids
        self.always_truncate_tail = truncate_config.get(
            'always_truncate_tail', False)
        self.max_len_a = truncate_config.get('max_len_a', None)
        self.max_len_b = truncate_config.get('max_len_b', None)
        self.trunc_seg = truncate_config.get('trunc_seg', None)
        self.task_idx = 3   # relax projection layer for different tasks
        self.mask_source_words = mask_source_words
        assert mode in ("s2s", "l2r")
        self.mode = mode
        self.has_oracle = has_oracle
        self.num_qkv = num_qkv
        self.s2s_special_token = s2s_special_token
        self.s2s_add_segment = s2s_add_segment
        self.s2s_share_segment = s2s_share_segment
        self.pos_shift = pos_shift
        self.require_pseudo_ids = require_pseudo_ids
        self.rand_prob = rand_prob
        self.keep_prob = keep_prob

        self.cls_token = cls_token
        self.sep_token = sep_token
        self.mask_token = mask_token

    def __call__(self, instance):
        tokens_a, tokens_b = instance[:2]

        if self.pos_shift:
            tokens_b = [self.sep_token] + tokens_b

        # -3  for special tokens [CLS], [SEP], [SEP]
        num_truncated_a, _ = truncate_tokens_pair(tokens_a, tokens_b, self.max_len - 3, max_len_a=self.max_len_a,
                                                  max_len_b=self.max_len_b, trunc_seg=self.trunc_seg, always_truncate_tail=self.always_truncate_tail)

        # Add Special Tokens
        if self.s2s_special_token:
            tokens = [self.cls_token] + tokens_a + \
                [self.sep_token] + tokens_b + [self.sep_token]
        else:
            tokens = [self.sep_token] + tokens_a + [self.sep_token] + tokens_b + [self.sep_token]

        if self.new_segment_ids:
            if self.mode == "s2s":
                if self.s2s_add_segment:
                    if self.s2s_share_segment:
                        segment_ids = [0] + [1] * \
                            (len(tokens_a)+1) + [5]*(len(tokens_b)+1)
                    else:
                        segment_ids = [4] + [6] * \
                            (len(tokens_a)+1) + [5]*(len(tokens_b)+1)
                else:
                    segment_ids = [4] * (len(tokens_a)+2) + \
                        [5]*(len(tokens_b)+1)
            else:
                segment_ids = [2] * (len(tokens))
        else:
            segment_ids = [0]*(len(tokens_a)+2) + [1]*(len(tokens_b)+1)

        if self.pos_shift:
            n_pred = min(self.max_pred, len(tokens_b))
            masked_pos = [len(tokens_a)+2+i for i in range(len(tokens_b))]
            masked_weights = [1]*n_pred
            masked_ids = self.indexer(tokens_b[1:]+[self.sep_token])
            if len(masked_ids) > n_pred:
                masked_pos = masked_pos[:n_pred]
                masked_ids = masked_ids[:n_pred]
        else:
            # For masked Language Models
            # the number of prediction is sometimes less than max_pred when sequence is short
            effective_length = len(tokens_b)
            if self.mask_source_words:
                effective_length += len(tokens_a)
            n_pred = min(self.max_pred, max(
                1, int(round(effective_length*self.mask_prob))))
            # candidate positions of masked tokens
            cand_pos = []
            special_pos = set()
            for i, tk in enumerate(tokens):
                # only mask tokens_b (target sequence)
                # we will mask [SEP] as an ending symbol
                if (i >= len(tokens_a)+2) and (tk != self.cls_token):
                    cand_pos.append(i)
                elif self.mask_source_words and (i < len(tokens_a)+2) and (tk != self.cls_token) and (not tk.startswith(self.sep_token)):
                    cand_pos.append(i)
                else:
                    special_pos.add(i)
            shuffle(cand_pos)

            masked_pos = set()
            max_cand_pos = max(cand_pos)
            for pos in cand_pos:
                if len(masked_pos) >= n_pred:
                    break
                if pos in masked_pos:
                    continue

                def _expand_whole_word(st, end):
                    new_st, new_end = st, end
                    while (new_st >= 0) and tokens[new_st].startswith('##'):
                        new_st -= 1
                    while (new_end < len(tokens)) and tokens[new_end].startswith('##'):
                        new_end += 1
                    return new_st, new_end

                if (self.skipgram_prb > 0) and (self.skipgram_size >= 2) and (rand() < self.skipgram_prb):
                    # ngram
                    cur_skipgram_size = randint(2, self.skipgram_size)
                    if self.mask_whole_word:
                        st_pos, end_pos = _expand_whole_word(
                            pos, pos + cur_skipgram_size)
                    else:
                        st_pos, end_pos = pos, pos + cur_skipgram_size
                else:
                    # directly mask
                    if self.mask_whole_word:
                        st_pos, end_pos = _expand_whole_word(pos, pos + 1)
                    else:
                        st_pos, end_pos = pos, pos + 1

                for mp in range(st_pos, end_pos):
                    if (0 < mp <= max_cand_pos) and (mp not in special_pos):
                        masked_pos.add(mp)
                    else:
                        break

            masked_pos = list(masked_pos)
            if len(masked_pos) > n_pred:
                shuffle(masked_pos)
                masked_pos = masked_pos[:n_pred]

            masked_tokens = [tokens[pos] for pos in masked_pos]
            for pos in masked_pos:
                if rand() < 0.8:  # 80%
                    tokens[pos] = '[MASK]'
                elif rand() < 0.5:  # 10%
                    tokens[pos] = get_random_word(self.vocab_words)
            # when n_pred < max_pred, we only calculate loss within n_pred
            masked_weights = [1]*len(masked_tokens)

            # Token Indexing
            masked_ids = self.indexer(masked_tokens)
        # Token Indexing
        input_ids = self.indexer(tokens)

        # Zero Padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0]*n_pad)
        segment_ids.extend([0]*n_pad)

        if self.num_qkv > 1:
            mask_qkv = [0]*(len(tokens_a)+2) + [1] * (len(tokens_b)+1)
            mask_qkv.extend([0]*n_pad)
        else:
            mask_qkv = None

        input_mask = torch.zeros(self.max_len, self.max_len, dtype=torch.long)
        if self.mode == "s2s":
            input_mask[:, :len(tokens_a)+2].fill_(1)
            second_st, second_end = len(
                tokens_a)+2, len(tokens_a)+len(tokens_b)+3
            input_mask[second_st:second_end, second_st:second_end].copy_(
                self._tril_matrix[:second_end-second_st, :second_end-second_st])
        else:
            st, end = 0, len(tokens_a) + len(tokens_b) + 3
            input_mask[st:end, st:end].copy_(self._tril_matrix[:end, :end])

        # Zero Padding for masked target
        if self.max_pred > n_pred:
            n_pad = self.max_pred - n_pred
            if masked_ids is not None:
                masked_ids.extend([0]*n_pad)
            if masked_pos is not None:
                masked_pos.extend([0]*n_pad)
            if masked_weights is not None:
                masked_weights.extend([0]*n_pad)

        oracle_pos = None
        oracle_weights = None
        oracle_labels = None
        if self.has_oracle:
            s_st, labls = instance[2:]
            oracle_pos = []
            oracle_labels = []
            for st, lb in zip(s_st, labls):
                st = st - num_truncated_a[0]
                if st > 0 and st < len(tokens_a):
                    oracle_pos.append(st)
                    oracle_labels.append(lb)
            oracle_pos = oracle_pos[:20]
            oracle_labels = oracle_labels[:20]
            oracle_weights = [1] * len(oracle_pos)
            if len(oracle_pos) < 20:
                x_pad = 20 - len(oracle_pos)
                oracle_pos.extend([0] * x_pad)
                oracle_labels.extend([0] * x_pad)
                oracle_weights.extend([0] * x_pad)

            return (input_ids, segment_ids, input_mask, mask_qkv, masked_ids,
                    masked_pos, masked_weights, -1, self.task_idx,
                    oracle_pos, oracle_weights, oracle_labels)

        if len(masked_ids) > self.max_len_b:
            print("FC")

        return (input_ids, segment_ids, input_mask, mask_qkv, masked_ids, masked_pos, masked_weights, -1, self.task_idx)


class Preprocess4Seq2seqDecoder(Pipeline):
    """ Pre-processing steps for pretraining transformer """

    def __init__(self, vocab_words, indexer, max_len=512, max_tgt_length=128, new_segment_ids=False, mode="s2s",
                 num_qkv=0, s2s_special_token=False, s2s_add_segment=False, s2s_share_segment=False, pos_shift=False,
                 cls_token='[CLS]', sep_token='[SEP]', pad_token='[PAD]'):
        super().__init__()
        self.max_len = max_len
        self.vocab_words = vocab_words  # vocabulary (sub)words
        self.indexer = indexer  # function from token to token index
        self.max_len = max_len
        self._tril_matrix = torch.tril(torch.ones(
            (max_len, max_len), dtype=torch.long))
        self.new_segment_ids = new_segment_ids
        self.task_idx = 3   # relax projection layer for different tasks
        assert mode in ("s2s", "l2r")
        self.mode = mode
        self.max_tgt_length = max_tgt_length
        self.num_qkv = num_qkv
        self.s2s_special_token = s2s_special_token
        self.s2s_add_segment = s2s_add_segment
        self.s2s_share_segment = s2s_share_segment
        self.pos_shift = pos_shift

        self.cls_token = cls_token
        self.sep_token = sep_token
        self.pad_token = pad_token

        self.cc = 0

    def __call__(self, instance):
        tokens_a, max_a_len = instance

        # Add Special Tokens
        if self.s2s_special_token:
            padded_tokens_a = [self.cls_token] + tokens_a + [self.sep_token]
        else:
            padded_tokens_a = [self.cls_token] + tokens_a + [self.sep_token]
        assert len(padded_tokens_a) <= max_a_len + 2
        if max_a_len + 2 > len(padded_tokens_a):
            padded_tokens_a += [self.pad_token] * \
                (max_a_len + 2 - len(padded_tokens_a))
        assert len(padded_tokens_a) == max_a_len + 2
        max_len_in_batch = min(self.max_tgt_length +
                               max_a_len + 2, self.max_len)
        tokens = padded_tokens_a
        if self.new_segment_ids:
            if self.mode == "s2s":
                _enc_seg1 = 0 if self.s2s_share_segment else 4
                if self.s2s_add_segment:
                    if self.s2s_share_segment:
                        segment_ids = [
                            0] + [1]*(len(padded_tokens_a)-1) + [5]*(max_len_in_batch - len(padded_tokens_a))
                    else:
                        segment_ids = [
                            4] + [6]*(len(padded_tokens_a)-1) + [5]*(max_len_in_batch - len(padded_tokens_a))
                else:
                    segment_ids = [4]*(len(padded_tokens_a)) + \
                        [5]*(max_len_in_batch - len(padded_tokens_a))
            else:
                segment_ids = [2]*max_len_in_batch
        else:
            segment_ids = [0]*(len(padded_tokens_a)) \
                + [1]*(max_len_in_batch - len(padded_tokens_a))

        if self.num_qkv > 1:
            mask_qkv = [0]*(len(padded_tokens_a)) + [1] * \
                (max_len_in_batch - len(padded_tokens_a))
        else:
            mask_qkv = None

        position_ids = []
        for i in range(len(tokens_a) + 2):
            position_ids.append(i)
        for i in range(len(tokens_a) + 2, max_a_len + 2):
            position_ids.append(0)
        for i in range(max_a_len + 2, max_len_in_batch):
            position_ids.append(i - (max_a_len + 2) + len(tokens_a) + 2)

        # Token Indexing
        input_ids = self.indexer(tokens)

        self.cc += 1
        if self.cc < 20:
            logger.info("Input src = %s" % " ".join(self.vocab_words[tk_id] for tk_id in input_ids))

        # Zero Padding
        input_mask = torch.zeros(
            max_len_in_batch, max_len_in_batch, dtype=torch.long)
        if self.mode == "s2s":
            input_mask[:, :len(tokens_a)+2].fill_(1)
        else:
            st, end = 0, len(tokens_a) + 2
            input_mask[st:end, st:end].copy_(
                self._tril_matrix[:end, :end])
            input_mask[end:, :len(tokens_a)+2].fill_(1)
        second_st, second_end = len(padded_tokens_a), max_len_in_batch

        input_mask[second_st:second_end, second_st:second_end].copy_(
            self._tril_matrix[:second_end-second_st, :second_end-second_st])

        return (input_ids, segment_ids, position_ids, input_mask, mask_qkv, self.task_idx)
