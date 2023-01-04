import json
import sys
import argparse
sys.path.insert(0, './src')

from logger_config import logger
from metrics import compute_mrr, trec_eval
from utils import save_json_to_file
from data_utils import load_qrels, load_msmarco_predictions

parser = argparse.ArgumentParser(description='compute metrics for ms-marco predictions')
parser.add_argument('--in-path', default='', type=str, metavar='N',
                    help='path to predictions in msmarco output format')
parser.add_argument('--qrels', default='./data/msmarco/dev_qrels.txt', type=str, metavar='N',
                    help='path to qrels')

args = parser.parse_args()
logger.info('Args={}'.format(json.dumps(args.__dict__, ensure_ascii=False, indent=4)))


def main():
    qrels = load_qrels(path=args.qrels)
    predictions = load_msmarco_predictions(args.in_path)
    all_metrics = trec_eval(qrels=qrels, predictions=predictions)
    all_metrics['mrr'] = compute_mrr(qrels=qrels, predictions=predictions)

    logger.info(json.dumps(all_metrics, ensure_ascii=False, indent=4))

    save_json_to_file(all_metrics, '{}.metrics.json'.format(args.in_path))


if __name__ == '__main__':
    main()
