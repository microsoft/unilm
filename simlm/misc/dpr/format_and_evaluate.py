import os
import argparse
import json
import sys
sys.path.insert(0, 'src/')

from tqdm import tqdm
from typing import Dict, Any
from datasets import Dataset

from evaluate_dpr_retrieval import has_answers, SimpleTokenizer, evaluate_retrieval
from data_utils import load_query_answers, load_corpus
from utils import save_json_to_file
from logger_config import logger


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert an TREC run to DPR retrieval result json.')
    parser.add_argument('--data-dir', required=True, help='data dir')
    parser.add_argument('--topics', required=True, help='topic name')
    parser.add_argument('--topk', type=int, nargs='+', help="topk to evaluate")
    parser.add_argument('--input', required=True, help='Input TREC run file.')
    parser.add_argument('--store-raw', action='store_true', help='Store raw text of passage')
    parser.add_argument('--regex', action='store_true', default=False, help="regex match")
    parser.add_argument('--output', required=True, help='Output DPR Retrieval json file.')
    args = parser.parse_args()

    qas = load_query_answers(path=args.topics)
    corpus = load_corpus(path=os.path.join(args.data_dir, 'passages.jsonl.gz'))

    retrieval = {}
    tokenizer = SimpleTokenizer()

    predictions = []
    for line in tqdm(open(args.input), mininterval=1):
        question_id, doc_idx, _, score = line.strip().split('\t')[:4]

        predictions.append({'question_id': question_id,
                            'doc_idx': int(doc_idx),
                            'score': score})

    dataset = Dataset.from_dict({'question_id': [ex['question_id'] for ex in predictions],
                                 'doc_idx': [ex['doc_idx'] for ex in predictions],
                                 'score': [ex['score'] for ex in predictions]})
    logger.info('Get {} predictions in total'.format(len(dataset)))

    def _map_func(example: Dict[str, Any]) -> dict:
        question_id, doc_idx, score = example['question_id'], example['doc_idx'], example['score']

        question = qas[question_id]['query']
        answers = qas[question_id]['answers']
        title, text = corpus[doc_idx]['title'], corpus[doc_idx]['contents']
        ctx = '{}\n{}'.format(title, text)

        answer_exist = has_answers(text, answers, tokenizer, args.regex)

        example['question'] = question
        example['answers'] = answers
        example['docid'] = doc_idx
        example['has_answer'] = answer_exist
        if args.store_raw:
            example['text'] = ctx

        return example

    dataset = dataset.map(_map_func,
                          num_proc=min(os.cpu_count(), 16))

    retrieval = {}
    for ex in tqdm(dataset, mininterval=2, desc='convert to dpr format'):
        question_id, question, answers = ex['question_id'], ex['question'], ex['answers']
        if question_id not in retrieval:
            retrieval[question_id] = {'question': question, 'answers': answers, 'contexts': []}
        retrieval[question_id]['contexts'].append(
            {k: ex[k] for k in ['docid', 'score', 'text', 'has_answer'] if k in ex}
        )

    save_json_to_file(retrieval, path=args.output)
    logger.info('Convert {} to {} done'.format(args.input, args.output))

    metrics = evaluate_retrieval(retrieval_file=args.output,
                                 topk=args.topk,
                                 regex=args.regex)
    logger.info('{} recall metrics: {}'.format(
        os.path.basename(args.output),
        json.dumps(metrics, ensure_ascii=False, indent=4)))

    base_dir, base_name = os.path.dirname(args.output), os.path.basename(args.output)
    save_json_to_file(metrics, path='{}/metrics_{}'.format(base_dir, base_name))
