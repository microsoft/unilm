from __future__ import absolute_import, division, print_function

import logging
import os
import json
import random
import glob
import re

import torch
import tqdm
import torch.utils.data

logger = logging.getLogger(__name__)


class Seq2seqDatasetForBert(torch.utils.data.Dataset):
    def __init__(
            self, features, max_source_len, max_target_len,
            vocab_size, cls_id, sep_id, pad_id, mask_id,
            random_prob, keep_prob, offset, num_training_instances,
            span_len=1, span_prob=1.0):
        self.features = features
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.offset = offset
        if offset > 0:
            logger.info("  ****  Set offset %d in Seq2seqDatasetForBert ****  ", offset)
        self.cls_id = cls_id
        self.sep_id = sep_id
        self.pad_id = pad_id
        self.random_prob = random_prob
        self.keep_prob = keep_prob
        self.mask_id = mask_id
        self.vocab_size = vocab_size
        self.num_training_instances = num_training_instances
        self.span_len = span_len
        self.span_prob = span_prob

    def __len__(self):
        return int(self.num_training_instances)

    def __trunk(self, ids, max_len):
        if len(ids) > max_len - 1:
            ids = ids[:max_len - 1]
        ids = ids + [self.sep_id]
        return ids

    def __pad(self, ids, max_len):
        if len(ids) < max_len:
            return ids + [self.pad_id] * (max_len - len(ids))
        else:
            assert len(ids) == max_len
            return ids

    def __getitem__(self, idx):
        idx = (self.offset + idx) % len(self.features)
        feature = self.features[idx]
        source_ids = self.__trunk([self.cls_id] + feature["source_ids"], self.max_source_len)
        target_ids = self.__trunk(feature["target_ids"], self.max_target_len)
        pseudo_ids = []
        for tk_id in target_ids:
            p = random.random()
            if p < self.keep_prob:
                pseudo_ids.append(tk_id)
            elif p < self.keep_prob + self.random_prob:
                pseudo_ids.append(random.randint(0, self.vocab_size - 1))
            else:
                pseudo_ids.append(self.mask_id)

        num_source_tokens = len(source_ids)
        num_target_tokens = len(target_ids)

        source_ids = self.__pad(source_ids, self.max_source_len)
        target_ids = self.__pad(target_ids, self.max_target_len)
        pseudo_ids = self.__pad(pseudo_ids, self.max_target_len)

        if self.span_len > 1:
            span_ids = []
            span_id = 1
            while len(span_ids) < num_target_tokens:
                p = random.random()
                if p < self.span_prob:
                    span_len = random.randint(2, self.span_len)
                    span_len = min(span_len, num_target_tokens - len(span_ids))
                else:
                    span_len = 1
                span_ids.extend([span_id] * span_len)
                span_id += 1
            span_ids = self.__pad(span_ids, self.max_target_len)
            return source_ids, target_ids, pseudo_ids, num_source_tokens, num_target_tokens, span_ids
        else:
            return source_ids, target_ids, pseudo_ids, num_source_tokens, num_target_tokens


# DONE: finish this!!! the 2D input id settings.
class Seq2seqDatasetForLayoutlm(torch.utils.data.Dataset):
    def __init__(
            self, features, max_source_len, max_target_len,
            vocab_size, cls_id, sep_id, pad_id, mask_id,
            random_prob, keep_prob, offset, num_training_instances, layout_flag=True,
            span_len=1, span_prob=1.0):

        self.layout_flag = layout_flag

        self.features = features
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.offset = offset
        if offset > 0:
            logger.info("  ****  Set offset %d in Seq2seqDatasetForBert ****  ", offset)
        self.cls_id = cls_id
        self.sep_id = sep_id
        self.pad_id = pad_id
        self.random_prob = random_prob
        self.keep_prob = keep_prob
        self.mask_id = mask_id
        self.vocab_size = vocab_size
        self.num_training_instances = num_training_instances
        self.span_len = span_len
        self.span_prob = span_prob

        self.index_sp_id = 0

    def __len__(self):
        return int(self.num_training_instances)

    def __clip_index(self, ids):
        replace_value = 0
        for i in range(len(ids)):
            if ids[i] > self.max_source_len - 1:
                ids[i] = replace_value
        return ids

    def __trunk(self, ids, max_len, simple=False, value=None):
        trunk_value = value if value is not None else self.sep_id
        if len(ids) > max_len - 1:
            ids = ids[:max_len - 1]
        if simple:
            ids = ids + [trunk_value]
        else:
            ids = ids + [[trunk_value, 1000, 1000, 1000, 1000]]
        return ids

    def __pad(self, ids, max_len, simple=False, value=None):
        pad_value = value if value is not None else self.pad_id
        if len(ids) < max_len:
            if simple:
                return ids + [pad_value] * (max_len - len(ids))
            else:
                return ids + [[pad_value, 0, 0, 0, 0]] * (max_len - len(ids))
        else:
            assert len(ids) == max_len
            return ids

    def __getitem__(self, idx):
        if self.layout_flag:
            return self.__getitem_layout__(idx)
        else:
            return self.__getitem_bert__(idx)

    def __getitem_bert__(self, idx):
        idx = (self.offset + idx) % len(self.features)
        feature = self.features[idx]
        source_ids = self.__trunk([self.cls_id] + feature["source_ids"], self.max_source_len, simple=True)
        target_ids = self.__trunk(feature["target_ids"], self.max_target_len, simple=True)
        target_index = self.__trunk(feature['target_index'], self.max_target_len, simple=True, value=self.index_sp_id)

        pseudo_ids = []
        for tk_id in target_ids:
            p = random.random()
            if p < self.keep_prob:
                pseudo_ids.append(tk_id)
            elif p < self.keep_prob + self.random_prob:
                pseudo_ids.append(random.randint(0, self.vocab_size - 1))
            else:
                pseudo_ids.append(self.mask_id)

        num_source_tokens = len(source_ids)
        num_target_tokens = len(target_ids)

        source_ids = self.__pad(source_ids, self.max_source_len, simple=True)
        target_ids = self.__pad(target_ids, self.max_target_len, simple=True)
        pseudo_ids = self.__pad(pseudo_ids, self.max_target_len, simple=True)
        target_index = self.__pad(target_index, self.max_target_len, simple=True, value=self.index_sp_id)
        target_index = self.__clip_index(target_index)

        if self.span_len > 1:
            span_ids = []
            span_id = 1
            while len(span_ids) < num_target_tokens:
                p = random.random()
                if p < self.span_prob:
                    span_len = random.randint(2, self.span_len)
                    span_len = min(span_len, num_target_tokens - len(span_ids))
                else:
                    span_len = 1
                span_ids.extend([span_id] * span_len)
                span_id += 1
            span_ids = self.__pad(span_ids, self.max_target_len)
            return source_ids, target_ids, pseudo_ids, num_source_tokens, num_target_tokens, span_ids, target_index
        else:
            return source_ids, target_ids, pseudo_ids, num_source_tokens, num_target_tokens, target_index

    def __getitem_layout__(self, idx):
        # TODO: how to initialize the random and masked tokens' pos emb
        #   Simple Solution: only mask the text
        idx = (self.offset + idx) % len(self.features)
        feature = self.features[idx]
        source_ids = self.__trunk([[self.cls_id, 0, 0, 0, 0]] + feature["source_ids"], self.max_source_len)
        target_ids = self.__trunk(feature["target_ids"], self.max_target_len)
        target_index = self.__trunk(feature['target_index'], self.max_target_len, simple=True, value=self.index_sp_id)

        pseudo_ids = []
        for tk_id in target_ids:
            p = random.random()
            if p < self.keep_prob:
                pseudo_ids.append(tk_id)
            elif p < self.keep_prob + self.random_prob:
                pseudo_ids.append([random.randint(0, self.vocab_size - 1)] + [0, 0, 0, 0])  # tk_id[1:])
            else:
                pseudo_ids.append([self.mask_id] + [0, 0, 0, 0])  # tk_id[1:])

        num_source_tokens = len(source_ids)
        num_target_tokens = len(target_ids)

        source_ids = self.__pad(source_ids, self.max_source_len)
        target_ids = self.__pad(target_ids, self.max_target_len)
        pseudo_ids = self.__pad(pseudo_ids, self.max_target_len)
        target_index = self.__pad(target_index, self.max_target_len, simple=True, value=self.index_sp_id)
        target_index = self.__clip_index(target_index)

        if self.span_len > 1:
            span_ids = []
            span_id = 1
            while len(span_ids) < num_target_tokens:
                p = random.random()
                if p < self.span_prob:
                    span_len = random.randint(2, self.span_len)
                    span_len = min(span_len, num_target_tokens - len(span_ids))
                else:
                    span_len = 1
                span_ids.extend([span_id] * span_len)
                span_id += 1
            span_ids = self.__pad(span_ids, self.max_target_len)
            return source_ids, target_ids, pseudo_ids, num_source_tokens, num_target_tokens, span_ids, target_index
        else:
            return source_ids, target_ids, pseudo_ids, num_source_tokens, num_target_tokens, target_index


def batch_list_to_batch_tensors(batch):
    batch_tensors = []
    for x in zip(*batch):
        if isinstance(x[0], torch.Tensor):
            batch_tensors.append(torch.stack(x))
        else:
            batch_tensors.append(torch.tensor(x, dtype=torch.long))
    return batch_tensors


def get_max_epoch_model(output_dir):
    fn_model_list = glob.glob(os.path.join(output_dir, "model.*.bin"))
    fn_optim_list = glob.glob(os.path.join(output_dir, "optim.*.bin"))
    if (not fn_model_list) or (not fn_optim_list):
        return None
    os.path.basename(output_dir)
    both_set = set([int(os.path.basename(fn).split('.')[1]) for fn in fn_model_list]
                   ) & set([int(os.path.basename(fn).split('.')[1]) for fn in fn_optim_list])
    if both_set:
        return max(both_set)
    else:
        return None


def load_and_cache_examples(
        example_file, tokenizer, local_rank, cached_features_file, shuffle=True):
    # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()

    if cached_features_file is not None and os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", example_file)

        examples = []
        with open(example_file, mode="r", encoding="utf-8") as reader:
            for i, line in enumerate(reader):
                if i == 100:
                    break
                examples.append(json.loads(line))
        features = []

        for example in tqdm.tqdm(examples):
            if isinstance(example["src"], list):
                source_tokens = example["src"]
                target_tokens = example["tgt"]
            else:
                source_tokens = tokenizer.tokenize(example["src"])
                target_tokens = tokenizer.tokenize(example["tgt"])
            features.append({
                "source_ids": tokenizer.convert_tokens_to_ids(source_tokens),
                "target_ids": tokenizer.convert_tokens_to_ids(target_tokens),
            })

        if shuffle:
            random.shuffle(features)

        if local_rank in [-1, 0] and cached_features_file is not None:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    if local_rank == 0:
        torch.distributed.barrier()

    return features


def load_and_cache_line_order_examples(
        example_path, tokenizer, local_rank, cached_features_file, max_src_length=1024,
        layout_flag=True, shuffle=True,
        src_shuffle_rate=0,
        file_info_flag=False,
        ):
    # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()

    if cached_features_file is not None and os.path.exists(cached_features_file) and False:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset at %s", example_path)

        examples = []

        with open(example_path, 'r') as layout_reader:
            logger.info(f'Start loading {example_path}')
            for i, line in enumerate(layout_reader):
                examples.append(json.loads(line))

        features = []

        for layout in tqdm.tqdm(examples):
            bleu = layout['bleu']

            if random.random() < src_shuffle_rate:
                # print('Random!!!')
                # DONE: the random src! here has bug! index also need shuffle
                src_layout = layout['src']
                tgt_index = layout['tgt_index']

                source_length = len(src_layout)
                shuffle_index = list(range(source_length))
                random.shuffle(shuffle_index)

                shuffle_layout = ['' for _ in range(source_length)]
                for i, j in enumerate(shuffle_index):
                    # NOTE: map i-th token to j-th token
                    shuffle_layout[j] = src_layout[i]

                shuffle_target_index = [shuffle_index[i] for i in tgt_index]

                layout['tgt_index'] = shuffle_target_index
                layout['src'] = shuffle_layout

            mask = tokenizer.mask_token_id
            src_ids = [tokenizer.convert_tokens_to_ids([str(tmp_i)])[:1] + src_layout for tmp_i, src_layout in enumerate(layout['src'])]
            tgt_ids = [tokenizer.convert_tokens_to_ids([str(tmp_i)])[:1] + tgt_layout for tmp_i, tgt_layout in enumerate(layout['tgt'])]
            tgt_index = layout['tgt_index']

            feature = {
                "source_ids": src_ids,
                "target_ids": tgt_ids,
                "target_index": tgt_index,
                'bleu': bleu
            }

            if file_info_flag:
                file_info = {'original_filename': layout['filename'], 'filename': layout['filename'],
                             'page_idx': 0}
                feature['file_info'] = file_info

            features.append(feature)

        if shuffle:
            random.shuffle(features)

        if local_rank in [-1, 0] and cached_features_file is not None:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    if local_rank == 0:
        torch.distributed.barrier()

    return features


def load_and_cache_layoutlm_examples(
        example_path, tokenizer, local_rank, cached_features_file, max_src_length=1024,
        layout_flag=True, shuffle=True,
        src_shuffle_rate=0,
        file_info_flag=False
        ):
    # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()

    if cached_features_file is not None and os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset at %s", example_path)

        examples = []

        if os.path.isdir(example_path):
            text_files = glob.glob(f'{example_path}/*text*.json')
            layout_files = [re.sub('text|txt', 'layout', x, 1) for x in text_files]
        else:
            text_files = [example_path]
            layout_files = [re.sub('text|txt', 'layout', example_path, 1)]
        for text_file, layout_file in zip(text_files, layout_files):
            with open(text_file, mode='r', encoding='utf-8') as text_reader, \
                    open(layout_file, mode='r', encoding='utf-8') as layout_reader:
                logger.info(f'Start loading {text_file}')
                for i, (text_line, layout_line) in enumerate(zip(text_reader, layout_reader)):
                    if (i + 1) % 10000 == 0:
                        logger.info(f'{i + 1} lines ...')
                    examples.append((json.loads(text_line), json.loads(layout_line)))

        features = []

        def tokenize_text_and_layout_src(_text, _layout, _layout_flag):
            ret = []
            index_split = {}
            words = _text.split()
            # note: (OLD) the index should start from 1: 0-the cls token in src
            # note: (NEW) we need to remove the src embedding's CLS SEP token so we can still start from 0
            # note: (NEWER) we need to at least one blank pos for ignore index in loss function (we use sep's index)
            # NOTE: (NEWER-ER) 1 for all padding tgt index
            new_token_index = 1  # first ordinary index
            for i, (word, box) in enumerate(zip(words, _layout)):

                if (not box[2] >= box[0]) or (not box[3] >= box[1]):
                    continue

                tokens = tokenizer.tokenize(word)
                tokens = tokenizer.convert_tokens_to_ids(tokens)
                new_token_ids = []
                for token in tokens:
                    if _layout_flag:
                        ret.append([token] + box)
                    else:
                        ret.append(token)
                    new_token_ids.append(new_token_index)
                    new_token_index += 1
                index_split[i] = new_token_ids

            return ret, index_split

        def tokenize_text_and_layout_tgt(_text, _layout, _index, _index_split, _layout_flag):
            ret = []
            ret_index = []
            words = _text.split()
            for word, box, i in zip(words, _layout, _index):

                if (not box[2] >= box[0]) or (not box[3] >= box[1]):
                    continue

                tokens = tokenizer.tokenize(word)
                tokens = tokenizer.convert_tokens_to_ids(tokens)
                for token, ii in zip(tokens, _index_split[i]):
                    if _layout_flag:
                        ret.append([token] + box)
                    else:
                        ret.append(token)
                    ii = min(ii, max_src_length - 1)
                    ret_index.append(ii)
            return ret, ret_index

        for text, layout in tqdm.tqdm(examples):
            if 'bleu' in text:
                bleu = text['bleu']
            else:
                bleu = 0

            if random.random() < src_shuffle_rate:
                # print('Random!!!')
                # DONE: the random src! here has bug! index also need shuffle
                src_text = text['src']
                src_layout = layout['src']
                tgt_index = text['tgt_index']

                src_text = src_text.split()
                source_length = len(src_text)
                shuffle_index = list(range(source_length))
                random.shuffle(shuffle_index)

                shuffle_text = ['' for _ in range(source_length)]
                shuffle_layout = ['' for _ in range(source_length)]
                for i, j in enumerate(shuffle_index):
                    # NOTE: map i-th token to j-th token
                    shuffle_text[j] = src_text[i]
                    shuffle_layout[j] = src_layout[i]

                shuffle_target_index = [shuffle_index[i] for i in tgt_index]

                text['src'] = ' '.join(shuffle_text)
                text['tgt_index'] = shuffle_target_index
                layout['src'] = shuffle_layout

            src_ids, src_index_split = tokenize_text_and_layout_src(text['src'], layout['src'],
                                                                    _layout_flag=layout_flag)
            tgt_ids, tgt_index = tokenize_text_and_layout_tgt(text['tgt'], layout['tgt'], text['tgt_index'],
                                                              src_index_split, _layout_flag=layout_flag)

            feature = {
                "source_ids": src_ids,
                "target_ids": tgt_ids,
                "target_index": tgt_index,
                'bleu': bleu
            }

            if file_info_flag:
                file_info = {'original_filename': text['original_filename'], 'filename': text['filename'], 'page_idx': text['page_idx']}
                feature['file_info'] = file_info

            features.append(feature)

        if shuffle:
            random.shuffle(features)

        if local_rank in [-1, 0] and cached_features_file is not None:
            if not os.path.exists(os.path.dirname(cached_features_file)):
                os.makedirs(os.path.dirname(cached_features_file))
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    if local_rank == 0:
        torch.distributed.barrier()

    return features


def convert_src_layout_inputs_to_tokens(inputs, converter, max_src_length, layout_flag=True):
    ret = []
    if not layout_flag:
        for line in inputs:
            ret.append(converter(line["source_ids"])[: max_src_length])
    else:
        for line in inputs:
            raw_text_ids = [x[0] for x in line['source_ids']]
            raw_text = converter(raw_text_ids)
            new_line = [[t] + x[1:] for t, x in zip(raw_text, line['source_ids'])][: max_src_length]
            ret.append(new_line)
    return ret


def convert_tgt_layout_inputs_to_tokens(inputs, converter, max_tgt_length, layout_flag=True):
    ret = []
    if not layout_flag:
        for line in inputs:
            ret.append(converter(line["target_ids"])[: max_tgt_length])
    else:
        for line in inputs:
            raw_text_ids = [x[0] for x in line['target_ids']]
            ret.append(converter(raw_text_ids)[: max_tgt_length])
    return ret


def get_tokens_from_src_and_index(src, index, modifier=None):
    result = []
    for i in index:
        i = modifier(i)
        i = min(i, len(src) - 1)
        if isinstance(src[i], list):
            result.append(src[i][0])
        else:
            result.append(src[i])
    return result


def get_layout_from_src_and_index(src, index, modifier=None):
    result = []
    s = set()
    for i in index:
        i = modifier(i)
        i = min(i, len(src) - 1)
        layout = src[i][1:]
        if repr(layout) not in s:
            result.append(layout)
            s.add(repr(layout))
    return result


def get_everything_from_src_and_index(src, index, modifier=None):
    result = []
    for i in index:
        i = modifier(i)
        i = min(i, len(src) - 1)
        result.append(src[i])
    return result

