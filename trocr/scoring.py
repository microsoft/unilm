from fairseq.scoring import BaseScorer, register_scorer
from nltk.metrics.distance import edit_distance
from fairseq.dataclass import FairseqDataclass
import fastwer
from Levenshtein import distance
import string

@register_scorer("cer", dataclass=FairseqDataclass)
class CERScorer(BaseScorer):
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
        return f"CER: {self.score():.2f}"


@register_scorer("wpa", dataclass=FairseqDataclass)
class WPAScorer(BaseScorer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.refs = []
        self.preds = []
        self.alphabet = string.digits + string.ascii_lowercase

    def filter(self, string):
        string = ''.join([i for i in string if i in self.alphabet])
        return string

    def add_string(self, ref, pred):
        # print(f'[Pred] gt: "{ref}" | pred: "{pred}"')
        self.refs.append(self.filter(ref.lower()))
        self.preds.append(self.filter(pred.lower()))
    
    def score(self):

        length = len(self.refs)
        correct = 0
        for i in range(length):
            if self.refs[i] == self.preds[i]:
                correct += 1
        return round(correct / length * 100, 2)

        # return 100 - fastwer.score(self.preds, self.refs, char_level=False)

    def result_string(self) -> str:
        return f"WPA: {self.score():.2f}"

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