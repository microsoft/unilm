import os
from data import SROIETask2
from tqdm import tqdm


class SROIEScorer:
    def __init__(self):
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

    def score(self):
        prec = self.n_match_words / float(self.n_detected_words) * 100
        recall = self.n_match_words / float(self.n_gt_words) * 100
        f1 = 2 * (prec * recall) / (prec + recall)
        return prec, recall, f1

    def result_string(self):
        prec, recall, f1 = self.score()
        return f"Precision: {prec:.3f} Recall: {recall:.3f} F1: {f1:.3f}"

class NewSROIEScorer:
    def __init__(self):
        self.n_detected_words = 0
        self.n_gt_words = 0        
        self.n_match_words = 0
    
    def add_string(self, ref, pred):        
        pred_words = list(pred.split())
        self.n_detected_words += len(pred_words)
        
        for pred_w in pred_words:
            if pred_w in ref:
                self.n_match_words += 1
                self.n_gt_words += 1
                ref = ref.replace(pred_w, '', 1)
        self.n_gt_words += len(ref.split())

    def score(self):
        prec = self.n_match_words / float(self.n_detected_words) * 100
        recall = self.n_match_words / float(self.n_gt_words) * 100
        f1 = 2 * (prec * recall) / (prec + recall)
        return prec, recall, f1

    def result_string(self):
        prec, recall, f1 = self.score()
        return f"Precision: {prec:.3f} Recall: {recall:.3f} F1: {f1:.3f}"
