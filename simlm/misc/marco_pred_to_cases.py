import json
import os
import sys
import tqdm
import argparse
sys.path.insert(0, './src')

from typing import List, Dict

from utils import save_json_to_file
from logger_config import logger
from data_utils import load_qrels, load_corpus, load_queries, load_msmarco_predictions, ScoredDoc
from metrics import get_rel_threshold

parser = argparse.ArgumentParser(description='convert ms-marco predictions to a human-readable format')
parser.add_argument('--in-path', default='', type=str, metavar='N',
                    help='path to predictions in msmarco output format')
parser.add_argument('--split', default='dev', type=str, metavar='N',
                    help='which split to use')
parser.add_argument('--data-dir', default='./data/msmarco/', type=str, metavar='N',
                    help='data dir')

args = parser.parse_args()
logger.info('Args={}'.format(json.dumps(args.__dict__, ensure_ascii=False, indent=4)))


def main(topk: int = 10):
    predictions: Dict[str, List[ScoredDoc]] = load_msmarco_predictions(path=args.in_path)
    path_qrels = '{}/{}_qrels.txt'.format(args.data_dir, args.split)
    qrels = load_qrels(path=path_qrels) if os.path.exists(path_qrels) else None
    queries = load_queries(path='{}/{}_queries.tsv'.format(args.data_dir, args.split))
    corpus = load_corpus(path='{}/passages.jsonl.gz'.format(args.data_dir))

    pred_infos = []
    out_path = '{}.details.json'.format(args.in_path)
    rel_threshold = get_rel_threshold(qrels) if qrels else -1
    for qid in tqdm.tqdm(queries):
        pred_docs = []
        for scored_doc in predictions[qid][:topk]:
            correct = qrels is not None and scored_doc.pid in qrels[qid] and qrels[qid][scored_doc.pid] >= rel_threshold
            pred_docs.append({'id': scored_doc.pid,
                              'contents': corpus[int(scored_doc.pid)]['contents'],
                              'title': corpus[int(scored_doc.pid)]['title'],
                              'score': scored_doc.score})
            if qrels is not None:
                pred_docs[-1]['correct'] = correct
            if correct: break

        gold_rank, gold_score = -1, -1
        for idx, scored_doc in enumerate(predictions[qid]):
            if qrels is None:
                break
            if scored_doc.pid in qrels[qid] and qrels[qid][scored_doc.pid] >= rel_threshold:
                gold_rank = idx + 1
                gold_score = scored_doc.score
                break

        pred_info = {'query_id': qid,
                     'query': queries[qid],
                     'pred_docs': pred_docs}
        if qrels is not None:
            pred_info.update({
                'gold_docs': [corpus[int(doc_id)] for doc_id in qrels[qid] if qrels[qid][doc_id] >= rel_threshold],
                'gold_score': gold_score,
                'gold_rank': gold_rank
            })

        pred_infos.append(pred_info)

    save_json_to_file(pred_infos, out_path)
    logger.info('Save prediction details to {}'.format(out_path))


if __name__ == '__main__':
    main()
