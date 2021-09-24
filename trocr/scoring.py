from fairseq.scoring import BaseScorer, register_scorer
from nltk.metrics.distance import edit_distance
from fairseq.dataclass import FairseqDataclass
import fastwer

@register_scorer("cer2", dataclass=FairseqDataclass)
class CER2Scorer(BaseScorer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.refs = []
        self.preds = []

    def add_string(self, ref, pred):
        self.refs.append(ref)
        self.preds.append(pred)
    
    def score(self):
        return fastwer.score(self.preds, self.refs, char_level=True)

    def result_string(self) -> str:
        return f"CER2: {self.score():.2f}"

# def levenshtein(u, v):
#     prev = None
#     curr = [0] + range(1, len(v) + 1)
#     # Operations: (SUB, DEL, INS)
#     prev_ops = None
#     curr_ops = [(0, 0, i) for i in range(len(v) + 1)]
#     for x in xrange(1, len(u) + 1):
#         prev, curr = curr, [x] + ([None] * len(v))
#         prev_ops, curr_ops = curr_ops, [(0, x, 0)] + ([None] * len(v))
#         for y in xrange(1, len(v) + 1):
#             delcost = prev[y] + 1
#             addcost = curr[y - 1] + 1
#             subcost = prev[y - 1] + int(u[x - 1] != v[y - 1])
#             curr[y] = min(subcost, delcost, addcost)
#             if curr[y] == subcost:
#                 (n_s, n_d, n_i) = prev_ops[y - 1]
#                 curr_ops[y] = (n_s + int(u[x - 1] != v[y - 1]), n_d, n_i)
#             elif curr[y] == delcost:
#                 (n_s, n_d, n_i) = prev_ops[y]
#                 curr_ops[y] = (n_s, n_d + 1, n_i)
#             else:
#                 (n_s, n_d, n_i) = curr_ops[y - 1]
#                 curr_ops[y] = (n_s, n_d, n_i + 1)
#     return curr[len(v)], curr_ops[len(v)]

# @register_scorer("cer", dataclass=FairseqDataclass)
# class CERScorer(BaseScorer):
#     def __init__(self, cfg):
#         super().__init__(cfg)
#         self.cer_s, self.cer_i, self.cer_d, self.cer_n = 0, 0, 0, 0
#
#     def add_string(self, ref, pred):
#         _, (s, i, d) = levenshtein(ref, pred)
#         self.cer_s += s
#         self.cer_i += i
#         self.cer_d += d
#         self.cer_n += len(ref)
#
#     def score(self):
#         return 100.0 * (self.cer_s + self.cer_i + self.cer_d) / self.cer_n
#
#     def result_string(self) -> str:
#         return f"CER: {self.score():.2f}"

@register_scorer("acc_ed", dataclass=FairseqDataclass)
class AccEDScorer(BaseScorer):
    def __init__(self, args):
        super(AccEDScorer, self).__init__(args)
        self.n_data = 0
        self.n_correct = 0
        self.ed = 0

    def add_string(self, ref, pred):
        self.n_data += 1
        if ref == pred:
            self.n_correct += 1
        self.ed += edit_distance(ref, pred)
        self.ref.append(ref)
        self.pred.append(pred)

    def score(self):
        return self.n_correct / float(self.n_data) * 100, self.ed / float(self.n_data)

    def result_string(self):
        acc, norm_ed = self.score()
        return f"Accuracy: {acc:.3f} Norm ED: {norm_ed:.2f}"

@register_scorer("sroie", dataclass=FairseqDataclass)
class SROIEScorer(BaseScorer):
    def __init__(self, args):
        super(SROIEScorer, self).__init__(args)
        self.n_detected_words = 0
        self.n_gt_words = 0        
        self.n_match_words = 0

    def add_string(self, ref, pred):        
        pred_words = list(pred.split())
        ref_words = list(ref.split())
        self.n_gt_words += len(ref_words)
        self.n_detected_words += len(pred_words)
        for pred_w in pred_words:
            if pred_w in ref_words:
                self.n_match_words += 1
                ref_words.remove(pred_w)
        self.ref.append(ref)
        self.pred.append(pred)

    def score(self):
        prec = self.n_match_words / float(self.n_detected_words) * 100
        recall = self.n_match_words / float(self.n_gt_words) * 100
        f1 = 2 * (prec * recall) / (prec + recall)
        return prec, recall, f1

    def result_string(self):
        prec, recall, f1 = self.score()
        return f"Precision: {prec:.3f} Recall: {recall:.3f} F1: {f1:.3f}"