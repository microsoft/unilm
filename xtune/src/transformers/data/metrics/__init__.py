# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

try:
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import matthews_corrcoef, f1_score, average_precision_score, ndcg_score, roc_auc_score
    import numpy as np
    _has_sklearn = True
except (AttributeError, ImportError):
    _has_sklearn = False


def is_sklearn_available():
    return _has_sklearn


if _has_sklearn:

    def simple_accuracy(preds, labels):
        return (preds == labels).mean()

    def simple_ndcg(preds, labels, guids):
        ndcgs = []
        query2content = {}
        for guid, pred, label in zip(guids, preds, labels):
            query = guid.split("_")[0]
            if not query in query2content:
                query2content[query] = [[int(pred)], [int(label)]]
            else:
                query2content[query][0].append(int(pred))     
                query2content[query][1].append(int(label))     
 
        for key in query2content.keys():
            if len(query2content[key][1]) < 2 or len(query2content[key][0]) < 2:
                continue 
            ndcgs.append(ndcg_score(np.asarray([query2content[key][1]]), np.asarray([query2content[key][0]])))
        return {"ndcg" : np.array(ndcgs).mean()}

    def acc_and_f1(preds, labels):
        acc = simple_accuracy(preds, labels)
        f1 = f1_score(y_true=labels, y_pred=preds)
        return {
            "acc": acc,
            "f1": f1,
            "acc_and_f1": (acc + f1) / 2,
        }

    def acc_and_auc(preds, labels):   # auc of pr curve is equal to average precision
        acc = simple_accuracy(preds, labels)
        auc = average_precision_score(labels, preds)
        return {
            "acc": acc,
            "auc": auc,
            "acc_and_auc": (acc + auc) / 2,
        }

    def acc_and_roc_auc(preds, labels):   # auc of pr curve is equal to average precision
        acc = simple_accuracy(preds, labels)
        roc_auc = roc_auc_score(labels, preds)
        return {
            "acc": acc,
            "roc_auc": roc_auc,
            "acc_and_roc_auc": (acc + roc_auc) / 2,
        }

    def pearson_and_spearman(preds, labels):
        pearson_corr = pearsonr(preds, labels)[0]
        spearman_corr = spearmanr(preds, labels)[0]
        return {
            "pearson": pearson_corr,
            "spearmanr": spearman_corr,
            "corr": (pearson_corr + spearman_corr) / 2,
        }

    def xglue_compute_metrics(task_name, preds, labels, guids):
        assert len(preds) == len(labels)
        if task_name == "xnli":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "pawsx":
            return acc_and_auc(preds, labels)
        elif task_name == "qam":
            return acc_and_auc(preds, labels)
        elif task_name == "ads":
            return acc_and_roc_auc(preds, labels)
        elif task_name == "rel":
            return simple_ndcg(preds, labels, guids)
        elif task_name == "news":
            return {"acc": simple_accuracy(preds, labels)}
        else:
            raise KeyError(task_name)

    def xtreme_compute_metrics(task_name, preds, labels, guids):
        assert len(preds) == len(labels)
        if task_name == "xnli":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "pawsx":
            return acc_and_auc(preds, labels)
        else:
            raise KeyError(task_name)


    def glue_compute_metrics(task_name, preds, labels):
        assert len(preds) == len(labels)
        if task_name == "cola":
            return {"mcc": matthews_corrcoef(labels, preds)}
        elif task_name == "sst-2":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "mrpc":
            return acc_and_f1(preds, labels)
        elif task_name == "sts-b":
            return pearson_and_spearman(preds, labels)
        elif task_name == "qqp":
            return acc_and_f1(preds, labels)
        elif task_name == "mnli":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "mnli-mm":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "qnli":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "rte":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "wnli":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "hans":
            return {"acc": simple_accuracy(preds, labels)}
        else:
            raise KeyError(task_name)

    def xnli_compute_metrics(task_name, preds, labels):
        assert len(preds) == len(labels)
        if task_name == "xnli":
            return {"acc": simple_accuracy(preds, labels)}
        else:
            raise KeyError(task_name)
