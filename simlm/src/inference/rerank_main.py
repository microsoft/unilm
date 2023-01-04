import os
import tqdm
import torch

from contextlib import nullcontext
from torch.utils.data import DataLoader
from functools import partial
from datasets import Dataset
from typing import Dict, List
from transformers.file_utils import PaddingStrategy
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerFast,
    DataCollatorWithPadding,
    HfArgumentParser,
    BatchEncoding
)

from config import Arguments
from logger_config import logger
from utils import move_to_cuda
from models import RerankerForInference
from data_utils import load_msmarco_predictions, load_corpus, load_queries, \
    merge_rerank_predictions, get_rerank_shard_path

parser = HfArgumentParser((Arguments,))
args: Arguments = parser.parse_args_into_dataclasses()[0]


def _rerank_transform_func(tokenizer: PreTrainedTokenizerFast,
                           corpus: Dataset,
                           queries: Dict[str, str],
                           examples: Dict[str, List]) -> BatchEncoding:
    input_docs: List[str] = []

    # ATTENTION: this code should be consistent with RerankDataLoader
    for doc_id in examples['doc_id']:
        doc_id = int(doc_id)
        prefix = ''
        if corpus[doc_id].get('title', ''):
            prefix = corpus[doc_id]['title'] + ': '
        input_docs.append(prefix + corpus[doc_id]['contents'])

    input_queries = [queries[query_id] for query_id in examples['query_id']]
    batch_dict = tokenizer(input_queries,
                           text_pair=input_docs,
                           max_length=args.rerank_max_length,
                           padding=PaddingStrategy.DO_NOT_PAD,
                           truncation=True)

    return batch_dict


@torch.no_grad()
def _worker_compute_reranker_score(gpu_idx: int):
    preds = load_msmarco_predictions(args.rerank_in_path)
    query_ids = sorted(list(preds.keys()))
    qid_pid = []
    for query_id in tqdm.tqdm(query_ids, desc='load qid-pid', mininterval=2):
        qid_pid += [(scored_doc.qid, scored_doc.pid) for scored_doc in preds[query_id]
                    if scored_doc.rank <= args.rerank_depth]

    dataset = Dataset.from_dict({'query_id': [t[0] for t in qid_pid],
                                 'doc_id': [t[1] for t in qid_pid]})
    dataset = dataset.shard(num_shards=torch.cuda.device_count(),
                            index=gpu_idx,
                            contiguous=True)

    logger.info('GPU {} needs to process {} examples'.format(gpu_idx, len(dataset)))
    torch.cuda.set_device(gpu_idx)

    query_ids, doc_ids = dataset['query_id'], dataset['doc_id']
    assert len(dataset) == len(query_ids)

    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model: RerankerForInference = RerankerForInference.from_pretrained(args.model_name_or_path)
    model.eval()
    model.cuda()

    corpus: Dataset = load_corpus(path=os.path.join(args.data_dir, 'passages.jsonl.gz'))
    queries = load_queries(path='{}/{}_queries.tsv'.format(args.data_dir, args.rerank_split),
                           task_type=args.task_type)
    dataset.set_transform(partial(_rerank_transform_func, tokenizer, corpus, queries))

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if args.fp16 else None)
    data_loader = DataLoader(
        dataset,
        batch_size=args.rerank_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.dataloader_num_workers,
        collate_fn=data_collator,
        pin_memory=True)

    scores = []
    for batch_dict in tqdm.tqdm(data_loader, desc='passage rerank', mininterval=5):
        batch_dict = move_to_cuda(batch_dict)

        with torch.cuda.amp.autocast() if args.fp16 else nullcontext():
            outputs: SequenceClassifierOutput = model(batch_dict)
        scores.append(outputs.logits.squeeze(dim=-1).cpu())
        assert len(scores[-1].shape) == 1

    all_scores = torch.cat(scores, dim=-1)
    assert all_scores.shape[0] == len(query_ids), '{} != {}'.format(all_scores.shape[0], len(query_ids))
    all_scores = all_scores.tolist()

    with open(get_rerank_shard_path(args, gpu_idx), 'w', encoding='utf-8') as writer:
        for idx in range(len(query_ids)):
            # dummy rank, since a query may be split across different workers
            writer.write('{}\t{}\t{}\t{}\n'.format(query_ids[idx], doc_ids[idx], -1, round(all_scores[idx], 5)))

    logger.info('Done computing rerank score for worker {}'.format(gpu_idx))


def _batch_compute_reranker_score():
    logger.info('Args={}'.format(str(args)))
    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        logger.error('No gpu available')
        return

    logger.info('Use {} gpus'.format(gpu_count))
    torch.multiprocessing.spawn(_worker_compute_reranker_score, args=(), nprocs=gpu_count)
    logger.info('Done batch compute rerank score')

    merge_rerank_predictions(args, gpu_count)
    logger.info('Done merge results')


if __name__ == '__main__':
    _batch_compute_reranker_score()
