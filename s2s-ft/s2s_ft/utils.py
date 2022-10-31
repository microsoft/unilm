from __future__ import absolute_import, division, print_function

import logging
import os
import json
import random
import glob
import torch
import tqdm
import array
import collections
import torch.utils.data
from transformers.file_utils import WEIGHTS_NAME
try:
    import lmdb
except:
    pass

OPTIM_NAME = "optimizer.bin"


logger = logging.getLogger(__name__)


class TrainingExample(object):
    def __init__(self, source_ids, target_ids, example_id):
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.example_id = example_id


class Seq2seqDatasetForBert(torch.utils.data.Dataset):
    def __init__(
            self, features, max_source_len, max_target_len,
            vocab_size, cls_id, sep_id, pad_id, mask_id,
            random_prob, keep_prob, offset, num_training_instances, 
            mask_way='v1', target_mask_prob=-1.0, num_max_mask_token=0, 
            source_mask_prob=-1.0, 
            ):
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
        self.target_mask_prob = target_mask_prob
        if mask_way == 'v0':
            num_max_mask_token = self.max_target_len
            logger.info("Mask way v0: set num_max_mask_token = %d" % num_max_mask_token)
        self.num_max_mask_token = num_max_mask_token
        self.mask_way = mask_way
        assert mask_way in ('v0', 'v1', 'v2')
        self.source_mask_prob = source_mask_prob

    def __len__(self):
        return self.num_training_instances

    def __trunk(self, ids, max_len, append_sep=True):
        if append_sep:
            max_len -= 1
        if len(ids) > max_len:
            ids = ids[:max_len]
        if append_sep:
            ids = ids + [self.sep_id]
        return ids

    def __pad(self, ids, max_len):
        if len(ids) < max_len:
            return ids + [self.pad_id] * (max_len - len(ids))
        else:
            assert len(ids) == max_len
            return ids

    def get_masked_token(self, tk_id):
        p = random.random()
        if p < self.keep_prob:
            return tk_id
        elif p < self.keep_prob + self.random_prob:
            return random.randint(0, self.vocab_size - 1)
        else:
            return self.mask_id

    def __getitem__(self, _idx):
        idx = (self.offset + _idx) % len(self.features)
        # print("%d get %d" % (_idx, idx))
        feature = self.features[idx]
        source_ids = self.__trunk([self.cls_id] + feature.source_ids, self.max_source_len, append_sep=self.mask_way != 'v0')
        target_ids = feature.target_ids
        if self.mask_way == 'v0':
            target_ids = [self.sep_id] + target_ids
        target_ids = self.__trunk(target_ids, self.max_target_len, append_sep=self.mask_way != 'v0')

        num_source_tokens = len(source_ids)
        num_target_tokens = len(target_ids)

        if self.source_mask_prob > 0:
            for i in range(num_source_tokens):
                tk_id = source_ids[i]
                if tk_id != self.cls_id and tk_id != self.sep_id:
                    r = random.random()
                    if r < self.source_mask_prob:
                        source_ids[i] = self.get_masked_token(tk_id)

        source_ids = self.__pad(source_ids, self.max_source_len)
        target_ids = self.__pad(target_ids, self.max_target_len)

        if self.mask_way == 'v0':
            masked_pos = []
            masked_ids = []
            masked_weights = []
            for pos in range(num_target_tokens):
                if pos + 1 != num_target_tokens:
                    masked_ids.append(target_ids[pos + 1])
                else:
                    masked_ids.append(self.sep_id)
                masked_pos.append(pos)
                masked_weights.append(1)

                r = random.random()
                if r < self.target_mask_prob and pos > 0:
                    target_ids[pos] = self.get_masked_token(target_ids[pos])
            
            masked_ids = self.__pad(masked_ids, self.num_max_mask_token)
            masked_pos = self.__pad(masked_pos, self.num_max_mask_token)
            masked_weights = self.__pad(masked_weights, self.num_max_mask_token)

            return source_ids, target_ids, masked_ids, masked_pos, masked_weights, num_source_tokens, num_target_tokens
        elif self.mask_way == 'v1':
            masked_pos = list(range(num_target_tokens))
            random.shuffle(masked_pos)

            num_masked_token = \
                min(self.num_max_mask_token, int(self.target_mask_prob * num_target_tokens))
            if num_masked_token <= 0:
                num_masked_token = 1

            masked_pos = masked_pos[:num_masked_token]

            masked_ids = []
            masked_weights = []
            for pos in masked_pos:
                masked_ids.append(target_ids[pos])
                target_ids[pos] = self.get_masked_token(target_ids[pos])
                masked_weights.append(1)
            
            masked_ids = self.__pad(masked_ids, self.num_max_mask_token)
            masked_pos = self.__pad(masked_pos, self.num_max_mask_token)
            masked_weights = self.__pad(masked_weights, self.num_max_mask_token)

            return source_ids, target_ids, masked_ids, masked_pos, masked_weights, num_source_tokens, num_target_tokens
        elif self.mask_way == 'v2':
            pseudo_ids = []
            label_ids = []
            for pos in range(num_target_tokens):
                tk_id = target_ids[pos]
                masked_tk_id = self.get_masked_token(tk_id)
                pseudo_ids.append(masked_tk_id)
                label_ids.append(tk_id)
                r = random.random()
                if r < self.target_mask_prob:
                    target_ids[pos] = masked_tk_id
            label_ids = self.__pad(label_ids, self.max_target_len)
            pseudo_ids = self.__pad(pseudo_ids, self.max_target_len)

            return source_ids, target_ids, label_ids, pseudo_ids, num_source_tokens, num_target_tokens


def batch_list_to_batch_tensors(batch):
    batch_tensors = []
    for x in zip(*batch):
        if isinstance(x[0], torch.Tensor):
            batch_tensors.append(torch.stack(x))
        else:
            batch_tensors.append(torch.tensor(x, dtype=torch.long))
    return batch_tensors


def get_max_epoch_model(output_dir):
    fn_model_list = glob.glob(os.path.join(output_dir, "ckpt-*/%s" % WEIGHTS_NAME))
    fn_optim_list = glob.glob(os.path.join(output_dir, "ckpt-*/%s" % OPTIM_NAME))
    if (not fn_model_list) or (not fn_optim_list):
        return None
    both_set = set([int(os.path.dirname(fn).split('-')[-1]) for fn in fn_model_list]
                   ) & set([int(os.path.dirname(fn).split('-')[-1]) for fn in fn_optim_list])
    if both_set:
        return max(both_set)
    else:
        return None


def get_checkpoint_state_dict(output_dir, ckpt):
    model_recover_checkpoint = os.path.join(output_dir, "ckpt-%d" % ckpt, WEIGHTS_NAME)
    logger.info(" ** Recover model checkpoint in %s ** ", model_recover_checkpoint)
    model_state_dict = torch.load(model_recover_checkpoint, map_location='cpu')
    optimizer_recover_checkpoint = os.path.join(output_dir, "ckpt-%d" % ckpt, OPTIM_NAME)
    checkpoint_state_dict = torch.load(optimizer_recover_checkpoint, map_location='cpu')
    checkpoint_state_dict['model'] = model_state_dict
    return checkpoint_state_dict


def report_length(length_counter, total_count):
    max_len = max(length_counter.keys())
    a = 0
    tc = 0
    while a < max_len:
        cc = 0
        for i in range(16):
            cc += length_counter[a + i]

        tc += cc
        if cc > 0:
            logger.info("%d ~ %d = %d, %.2f%%" % (a, a + 16, cc, (tc * 100.0) / total_count))
        a += 16


def serialize_str(x):
    return u"{}".format(x).encode('ascii')


def serialize_array(x, dtype):
    data = array.array(dtype)
    data.fromlist(x)
    return data.tobytes()

def write_to_lmdb(db, key, value):
    success = False
    while not success:
        txn = db.begin(write=True)
        try:
            txn.put(key, value)
            txn.commit()
            success = True
        except lmdb.MapFullError:
            txn.abort()
            # double the map_size
            curr_limit = db.info()['map_size']
            new_limit = curr_limit*2
            print('>>> Doubling LMDB map size to %sMB ...' %
                  (new_limit >> 20,))
            db.set_mapsize(new_limit)  # double it


def deserialize_str(x):
    return x.decode('ascii')


class DocDB(object):
    def __init__(self, db_path):
        self.db_path = db_path
        self.env = lmdb.open(db_path, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.start_key_index = int(deserialize_str(txn.get(b'__start__')))
            self.size = int(deserialize_str(txn.get(b'__size__')))
            self.dtype = deserialize_str(txn.get(b'__dtype__'))

    def _deserialize_array(self, x):
        data = array.array(self.dtype)
        data.frombytes(x)
        return data.tolist()

    def __getitem__(self, doc_id):
        with self.env.begin(write=False) as txn:
            # example = {
            #     "source_ids": self._deserialize_array(txn.get(b"src_ids_%d" % doc_id)), 
            #     "target_ids": self._deserialize_array(txn.get(b"tgt_ids_%d" % doc_id)), 
            # }
            example = TrainingExample(
                source_ids=self._deserialize_array(txn.get(b"src_ids_%d" % doc_id)), 
                target_ids=self._deserialize_array(txn.get(b"tgt_ids_%d" % doc_id)),
                example_id=None, 
            )
        return example

    def __len__(self):
        return self.size


def load_and_cache_examples(
        example_file, tokenizer, local_rank, cached_features_file, shuffle=True, 
        lmdb_cache=None, lmdb_dtype='h', eval_mode=False):
    # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()

    if cached_features_file is not None and os.path.isfile(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    elif cached_features_file is not None and os.path.isdir(cached_features_file) \
        and os.path.exists(os.path.join(cached_features_file, 'lock.mdb')):
        logger.info("Loading features from cached LMDB %s", cached_features_file)
        features = DocDB(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", example_file)

        examples = []
        with open(example_file, mode="r", encoding="utf-8") as reader:
            for line in reader:
                examples.append(json.loads(line))
        features = []

        slc = collections.defaultdict(int)
        tlc = collections.defaultdict(int)

        for example in tqdm.tqdm(examples):
            if isinstance(example["src"], list):
                source_tokens = example["src"]
                target_tokens = [] if eval_mode else example["tgt"]
            else:
                source_tokens = tokenizer.tokenize(example["src"])
                target_tokens = [] if eval_mode else tokenizer.tokenize(example["tgt"])
            source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
            target_ids = tokenizer.convert_tokens_to_ids(target_tokens)

            slc[len(source_ids)] += 1
            tlc[len(target_ids)] += 1

            # features.append({
            #         "source_ids": source_ids,
            #         "target_ids": target_ids,
            #     })
            features.append(
                TrainingExample(
                    source_ids=source_ids,
                    target_ids=target_ids,
                    example_id=len(features), 
                )
            )

        if shuffle:
            random.shuffle(features)
            logger.info("Shuffle the features !")

        logger.info("Source length:")
        report_length(slc, total_count=len(examples))
        logger.info("Target length:")
        report_length(tlc, total_count=len(examples))

        if local_rank in [-1, 0] and cached_features_file is not None:
            if lmdb_cache:
                db = lmdb.open(cached_features_file, readonly=False, map_async=True)
                for idx, feature in enumerate(features):
                    write_to_lmdb(
                        db, b"src_ids_%d" % idx, 
                        serialize_array(feature.source_ids, dtype=lmdb_dtype))
                    write_to_lmdb(
                        db, b"tgt_ids_%d" % idx,
                        serialize_array(feature.target_ids, dtype=lmdb_dtype))
                write_to_lmdb(db, b"__start__", serialize_str(0))
                write_to_lmdb(db, b"__size__", serialize_str(len(features)))
                write_to_lmdb(db, b"__dtype__", serialize_str(lmdb_dtype))
                db.sync()
                db.close()
                logger.info("db_key_idx = %d" % len(features))
                del features
                features = cached_features_file
                logger.info("Saving features into cached lmdb dir %s", cached_features_file)
            else:
                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save(features, cached_features_file)

    # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    if local_rank == 0:
        torch.distributed.barrier()

    return features
