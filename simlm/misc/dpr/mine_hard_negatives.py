import os
import json
import argparse
import sys
import numpy as np

sys.path.insert(0, 'src/')

from tqdm import tqdm
from typing import Dict, Any

from logger_config import logger
from data_utils import load_query_answers, load_corpus, save_to_readable_format

parser = argparse.ArgumentParser(description='data preprocessing for NQ & TriviaQA in DPR paper')
parser.add_argument('--out-dir', default='./data/dpr/', type=str, metavar='N',
                    help='output directory')
parser.add_argument('--task', default='nq', type=str, metavar='N',
                    help='task name, nq or tq')
parser.add_argument('--train-pred-path', default='amlt/0621_cont_100k_psg16_ft/nq/nq_train.dpr.json',
                    type=str, metavar='N', help='path to train predictions to construct negatives')
parser.add_argument('--dev-pred-path', default='amlt/0621_cont_100k_psg16_ft/nq/nq_dev.dpr.json',
                    type=str, metavar='N', help='path to dev predictions to construct negatives')
parser.add_argument('--num-negatives', default=100, type=int, metavar='N',
                    help='number of negative passages')
parser.add_argument('--depth', default=100, type=int, metavar='N',
                    help='depth to choose negative passages from')

args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)
logger.info('Args: {}'.format(json.dumps(args.__dict__, ensure_ascii=False, indent=4)))
assert args.task in ['nq', 'tq']


def _load_qid_to_positive(path: str) -> Dict[str, str]:
    if args.task != 'nq':
        logger.warning('Only NQ has manually labeled positives')
        return {}

    examples = json.load(open(path, 'r', encoding='utf-8'))
    qid_to_pos_id = {}
    for ex in examples:
        positive_ctxs = ex['positive_ctxs']
        if len(positive_ctxs) > 0:
            qid_to_pos_id[ex['question']] = str(int(positive_ctxs[0]['passage_id']) - 1)

    logger.info('Get {} manually labeled positives from {}'.format(len(qid_to_pos_id), path))
    return qid_to_pos_id


def _write_prepared_data_to_disk(out_path: str,
                                 split: str,
                                 queries: Dict[str, Dict[str, Any]],
                                 preds_path: str):
    qid_to_pos_id = _load_qid_to_positive(path='{}/biencoder-nq-{}.json'.format(args.out_dir, split))
    cnt_filtered = 0

    preds = json.load(open(preds_path, 'r', encoding='utf-8'))

    with open(out_path, 'w', encoding='utf-8') as writer:
        for query_id in tqdm(queries, mininterval=1, desc='prepare {} data'.format(split)):
            cur_pred: dict = preds[query_id] if query_id in preds else preds[query_id.strip()]

            positive_ids, negative_ids = [], []
            manual_positive_id = qid_to_pos_id.get(query_id, None)
            if manual_positive_id:
                positive_ids.append(manual_positive_id)

            for ctx in cur_pred['contexts'][:args.depth]:
                doc_id = str(ctx['docid'])
                if doc_id == manual_positive_id:
                    continue
                elif ctx['has_answer']:
                    positive_ids.append(doc_id)
                else:
                    negative_ids.append(doc_id)

            if not positive_ids or not negative_ids:
                cnt_filtered += 1
                continue

            np.random.shuffle(negative_ids)
            negative_ids = negative_ids[:args.num_negatives]

            doc_id_to_score = {str(ctx['docid']): float(ctx['score']) for ctx in cur_pred['contexts']}
            doc_id_to_score[manual_positive_id] = 1000.
            example = {
                'query_id': query_id,
                'query': queries[query_id]['query'],
                'answers': queries[query_id]['answers'],
                'positives': {'doc_id': positive_ids,
                              'score': [doc_id_to_score.get(doc_id, -1.) for doc_id in positive_ids]
                              },
                'negatives': {'doc_id': negative_ids,
                              'score': [doc_id_to_score.get(doc_id, -1.) for doc_id in negative_ids]
                              },
            }

            writer.write(json.dumps(example, ensure_ascii=False, separators=(',', ':')))
            writer.write('\n')

    if cnt_filtered > 0:
        logger.info('{} questions are filtered out'.format(cnt_filtered))

    logger.info('Done write {} data to {}'.format(split, out_path))


if __name__ == '__main__':
    for split in ['dev', 'train']:
        _write_prepared_data_to_disk(
            out_path=os.path.join(args.out_dir, '{}_hard_{}.jsonl'.format(args.task, split)),
            split=split,
            queries=load_query_answers(path=os.path.join(args.out_dir, '{}_{}_queries.tsv'.format(args.task, split))),
            preds_path=(args.train_pred_path if split == 'train' else args.dev_pred_path)
        )

    corpus = load_corpus(path=os.path.join(args.out_dir, 'passages.jsonl.gz'))
    for split in ['dev', 'train']:
        save_to_readable_format(in_path=os.path.join(args.out_dir, '{}_hard_{}.jsonl'.format(args.task, split)),
                                corpus=corpus)

    logger.info('Done')
