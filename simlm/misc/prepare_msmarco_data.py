import os
import io
import gzip
import json
import random
import argparse
import ir_datasets
import numpy as np
import sys
sys.path.insert(0, 'src/')

from tqdm import tqdm
from typing import Dict, List
from datasets import Dataset

from logger_config import logger
from utils import save_json_to_file
from data_utils import load_msmarco_predictions, load_queries, load_qrels, load_corpus, \
    ScoredDoc, save_to_readable_format

parser = argparse.ArgumentParser(description='data preprocessing')
parser.add_argument('--out-dir', default='./data/msmarco/', type=str, metavar='N',
                    help='output directory')
parser.add_argument('--train-pred-path', default='./preds/official/train.msmarco.txt',
                    type=str, metavar='N', help='path to train predictions to construct negatives')
parser.add_argument('--dev-pred-path', default='./preds/official/dev.msmarco.txt',
                    type=str, metavar='N', help='path to dev predictions to construct negatives')
parser.add_argument('--num-negatives', default=210, type=int, metavar='N',
                    help='number of negative passages')
parser.add_argument('--num-random-neg', default=10, type=int, metavar='N',
                    help='number of random negatives to use')
parser.add_argument('--depth', default=200, type=int, metavar='N',
                    help='depth to choose negative passages from')
parser.add_argument('--title-path', default='./data/msmarco/para.title.txt',
                    type=str, metavar='N', help='path to titles data')
parser.add_argument('--create-train-dev-only', action='store_true', help='path to titles data')
parser.add_argument('--filter-noisy-positives', action='store_true', help='filter noisy positives or not')

args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)
logger.info('Args: {}'.format(json.dumps(args.__dict__, ensure_ascii=False, indent=4)))


def _write_corpus_to_disk():
    dataset = ir_datasets.load('msmarco-passage/train')
    titles = []
    if os.path.exists(args.title_path):
        titles = [line.strip().split('\t')[1] for line in tqdm(open(args.title_path).readlines(), desc='load title')]
        logger.info('Load {} titles from {}'.format(len(titles), args.title_path))
    else:
        logger.warning('No title data found: {}'.format(args.title_path))

    title_idx = 0
    out_path = os.path.join(args.out_dir, 'passages.jsonl.gz')
    with gzip.open(out_path, 'wb') as output:
        with io.TextIOWrapper(output, encoding='utf-8') as writer:
            for doc in tqdm(dataset.docs_iter()):
                ex = {'id': doc.doc_id, 'contents': doc.text}
                if titles:
                    ex['title'] = titles[title_idx]
                    title_idx += 1
                writer.write(json.dumps(ex, ensure_ascii=False, separators=(',', ':')))
                writer.write('\n')

    if titles:
        assert title_idx == len(titles), '{} != {}'.format(title_idx, len(titles))


def _write_queries_to_disk(split: str, out_path: str):
    dataset = ir_datasets.load("msmarco-passage/{}".format(split))
    with open(out_path, 'w', encoding='utf-8') as writer:
        for query in dataset.queries_iter():
            writer.write('{}\t{}\n'.format(query.query_id, query.text))

    logger.info('Write {} queries to {}'.format(split, out_path))


def _write_qrels_to_disk(split: str, out_path: str):
    dataset = ir_datasets.load("msmarco-passage/{}".format(split))
    with open(out_path, 'w', encoding='utf-8') as writer:
        for qrel in dataset.qrels_iter():
            # query_id, iteration, doc_id, relevance
            writer.write('{}\t{}\t{}\t{}\n'
                         .format(qrel.query_id, qrel.iteration, qrel.doc_id, qrel.relevance))

    logger.info('Write {} qrels to {}'.format(split, out_path))


def _write_prepared_data_to_disk(out_path: str,
                                 corpus: Dataset,
                                 queries: Dict[str, str],
                                 qrels: Dict[str, Dict[str, int]],
                                 preds: Dict[str, List[ScoredDoc]],
                                 is_train: bool = False):
    cnt_noisy_positive = 0
    cnt_output = 0

    with open(out_path, 'w', encoding='utf-8') as writer:
        for query_id in tqdm(qrels, mininterval=2):
            positive_doc_ids: Dict = qrels.get(query_id)
            if not positive_doc_ids:
                logger.warning('No positive found for query_id={}'.format(query_id))
                continue
            if is_train and args.filter_noisy_positives \
                    and all(sd.pid not in positive_doc_ids for sd in preds.get(query_id, [])):
                cnt_noisy_positive += 1
                continue
            # For official triples, only use those with negative doc ids
            if not preds.get(query_id, []):
                continue

            doc_id_to_score = {scored_doc.pid: scored_doc.score for scored_doc in preds.get(query_id, [])}

            negative_scored_docs = [scored_doc for scored_doc in preds.get(query_id, [])
                                    if scored_doc.pid not in positive_doc_ids][:args.depth]

            np.random.shuffle(negative_scored_docs)
            negative_scored_docs = negative_scored_docs[:(args.num_negatives - args.num_random_neg)]
            if len(negative_scored_docs) < args.num_negatives:
                if not negative_scored_docs:
                    logger.warning('No negatives found for query_id={} ({}), will use random negatives'
                                   .format(len(negative_scored_docs), queries[query_id], query_id))
                while len(negative_scored_docs) < args.num_negatives:
                    sd = ScoredDoc(qid=query_id, pid=str(random.randint(0, len(corpus) - 1)), rank=args.depth)
                    if sd.pid not in positive_doc_ids and sd.pid not in doc_id_to_score:
                        negative_scored_docs.append(sd)
                np.random.shuffle(negative_scored_docs)

            example = {'query_id': query_id,
                       'query': queries[query_id],
                       'positives': {'doc_id': list(positive_doc_ids),
                                     'score': [doc_id_to_score.get(doc_id, -1.) for doc_id in positive_doc_ids]
                                     },
                       'negatives': {'doc_id': [scored_doc.pid for scored_doc in negative_scored_docs],
                                     'score': [scored_doc.score for scored_doc in negative_scored_docs]
                                     },
                       }
            writer.write(json.dumps(example, ensure_ascii=False, separators=(',', ':')))
            writer.write('\n')
            cnt_output += 1

    if is_train and args.filter_noisy_positives:
        logger.info('Filter {} noisy positives'.format(cnt_noisy_positive))
    logger.info('Write {} examples to {}'.format(cnt_output, out_path))


if __name__ == '__main__':
    if not args.create_train_dev_only:
        _write_queries_to_disk(split='dev/small', out_path=os.path.join(args.out_dir, 'dev_queries.tsv'))
        _write_queries_to_disk(split='eval/small', out_path=os.path.join(args.out_dir, 'test_queries.tsv'))
        _write_queries_to_disk(split='trec-dl-2019/judged',
                               out_path=os.path.join(args.out_dir, 'trec_dl2019_queries.tsv'))
        _write_queries_to_disk(split='trec-dl-2020/judged',
                               out_path=os.path.join(args.out_dir, 'trec_dl2020_queries.tsv'))
        _write_queries_to_disk(split='train/judged', out_path=os.path.join(args.out_dir, 'train_queries.tsv'))

        _write_qrels_to_disk(split='dev/small', out_path=os.path.join(args.out_dir, 'dev_qrels.txt'))
        _write_qrels_to_disk(split='trec-dl-2019/judged',
                             out_path=os.path.join(args.out_dir, 'trec_dl2019_qrels.txt'))
        _write_qrels_to_disk(split='trec-dl-2020/judged',
                             out_path=os.path.join(args.out_dir, 'trec_dl2020_qrels.txt'))
        _write_qrels_to_disk(split='train/judged', out_path=os.path.join(args.out_dir, 'train_qrels.txt'))

        _write_corpus_to_disk()

    corpus = load_corpus(path=os.path.join(args.out_dir, 'passages.jsonl.gz'))

    _write_prepared_data_to_disk(out_path=os.path.join(args.out_dir, 'dev.jsonl'),
                                 corpus=corpus,
                                 queries=load_queries(path=os.path.join(args.out_dir, 'dev_queries.tsv')),
                                 qrels=load_qrels(path=os.path.join(args.out_dir, 'dev_qrels.txt')),
                                 preds=load_msmarco_predictions(path=args.dev_pred_path))

    _write_prepared_data_to_disk(out_path=os.path.join(args.out_dir, 'train.jsonl'),
                                 corpus=corpus,
                                 queries=load_queries(path=os.path.join(args.out_dir, 'train_queries.tsv')),
                                 qrels=load_qrels(path=os.path.join(args.out_dir, 'train_qrels.txt')),
                                 preds=load_msmarco_predictions(path=args.train_pred_path),
                                 is_train=True)

    save_to_readable_format(in_path=os.path.join(args.out_dir, 'dev.jsonl'), corpus=corpus)
    save_to_readable_format(in_path=os.path.join(args.out_dir, 'train.jsonl'), corpus=corpus)

    save_json_to_file(args.__dict__, path=os.path.join(args.out_dir, 'train_dev_create_args.json'))

    src_path = args.dev_pred_path
    dst_path = '{}/{}'.format(args.out_dir, os.path.basename(args.dev_pred_path))
    logger.info('copy {} to {}'.format(src_path, dst_path))
    os.system('cp {} {}'.format(src_path, dst_path))

    for trec_split in ['trec_dl2019', 'trec_dl2020', 'test']:
        trec_pred_path = '{}/{}.msmarco.txt'.format(os.path.dirname(args.dev_pred_path), trec_split)
        dst_path = '{}/{}'.format(args.out_dir, os.path.basename(trec_pred_path))
        if not os.path.exists(trec_pred_path):
            logger.warning('{} does not exist'.format(trec_pred_path))
            continue
        logger.info('copy {} to {}'.format(trec_pred_path, dst_path))
        os.system('cp {} {}'.format(trec_pred_path, dst_path))
