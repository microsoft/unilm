import torch
import pytrec_eval

from typing import List, Dict, Tuple

from data_utils import ScoredDoc
from logger_config import logger


def trec_eval(qrels: Dict[str, Dict[str, int]],
              predictions: Dict[str, List[ScoredDoc]],
              k_values: Tuple[int] = (10, 50, 100, 200, 1000)) -> Dict[str, float]:
    ndcg, _map, recall = {}, {}, {}

    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0

    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])

    results: Dict[str, Dict[str, float]] = {}
    for query_id, scored_docs in predictions.items():
        results.update({query_id: {sd.pid: sd.score for sd in scored_docs}})

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string})
    scores = evaluator.evaluate(results)

    for query_id in scores:
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
            recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]

    def _normalize(m: dict) -> dict:
        return {k: round(v / len(scores), 5) for k, v in m.items()}

    ndcg = _normalize(ndcg)
    _map = _normalize(_map)
    recall = _normalize(recall)

    all_metrics = {}
    for mt in [ndcg, _map, recall]:
        all_metrics.update(mt)

    return all_metrics


@torch.no_grad()
def accuracy(output: torch.tensor, target: torch.tensor, topk=(1,)) -> List[float]:
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res


@torch.no_grad()
def batch_mrr(output: torch.tensor, target: torch.tensor) -> float:
    assert len(output.shape) == 2
    assert len(target.shape) == 1
    sorted_score, sorted_indices = torch.sort(output, dim=-1, descending=True)
    _, rank = torch.nonzero(sorted_indices.eq(target.unsqueeze(-1)).long(), as_tuple=True)
    assert rank.shape[0] == output.shape[0]

    rank = rank + 1
    mrr = torch.sum(100 / rank.float()) / rank.shape[0]
    return mrr.item()


def get_rel_threshold(qrels: Dict[str, Dict[str, int]]) -> int:
    # For ms-marco passage ranking, score >= 1 is relevant
    # for trec dl 2019 & 2020, score >= 2 is relevant
    rel_labels = set()
    for q_id in qrels:
        for doc_id, label in qrels[q_id].items():
            rel_labels.add(label)

    logger.info('relevance labels: {}'.format(rel_labels))
    return 2 if max(rel_labels) >= 3 else 1


def compute_mrr(qrels: Dict[str, Dict[str, int]],
                predictions: Dict[str, List[ScoredDoc]],
                k: int = 10) -> float:
    threshold = get_rel_threshold(qrels)
    mrr = 0
    for qid in qrels:
        scored_docs = predictions.get(qid, [])
        for idx, scored_doc in enumerate(scored_docs[:k]):
            if scored_doc.pid in qrels[qid] and qrels[qid][scored_doc.pid] >= threshold:
                mrr += 1 / (idx + 1)
                break

    return round(mrr / len(qrels) * 100, 4)
