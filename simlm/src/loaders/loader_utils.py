from typing import List, Dict


def _slice_with_mod(elements: List, offset: int, cnt: int) -> List:
    return [elements[(offset + idx) % len(elements)] for idx in range(cnt)]


def group_doc_ids(examples: Dict[str, List],
                  negative_size: int,
                  offset: int,
                  use_first_positive: bool = False) -> List[int]:
    pos_doc_ids: List[int] = []
    positives: List[Dict[str, List]] = examples['positives']
    for idx, ex_pos in enumerate(positives):
        all_pos_doc_ids = ex_pos['doc_id']

        if use_first_positive:
            # keep positives that has higher score than all negatives
            all_pos_doc_ids = [doc_id for p_idx, doc_id in enumerate(all_pos_doc_ids)
                               if p_idx == 0 or ex_pos['score'][p_idx] >= ex_pos['score'][0]
                               or ex_pos['score'][p_idx] > max(examples['negatives'][idx]['score'])]

        cur_pos_doc_id = _slice_with_mod(all_pos_doc_ids, offset=offset, cnt=1)[0]
        pos_doc_ids.append(int(cur_pos_doc_id))

    neg_doc_ids: List[List[int]] = []
    negatives: List[Dict[str, List]] = examples['negatives']
    for ex_neg in negatives:
        cur_neg_doc_ids = _slice_with_mod(ex_neg['doc_id'],
                                          offset=offset * negative_size,
                                          cnt=negative_size)
        cur_neg_doc_ids = [int(doc_id) for doc_id in cur_neg_doc_ids]
        neg_doc_ids.append(cur_neg_doc_ids)

    assert len(pos_doc_ids) == len(neg_doc_ids), '{} != {}'.format(len(pos_doc_ids), len(neg_doc_ids))
    assert all(len(doc_ids) == negative_size for doc_ids in neg_doc_ids)

    input_doc_ids: List[int] = []
    for pos_doc_id, neg_ids in zip(pos_doc_ids, neg_doc_ids):
        input_doc_ids.append(pos_doc_id)
        input_doc_ids += neg_ids

    return input_doc_ids
