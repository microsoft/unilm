import os
import numpy as np
import torch
import inspect


from src.pequod.data.utils_squad import RawResult, write_predictions
from src.pequod.data.utils_squad_evaluate import EVAL_OPTS, main as evaluate_on_squad


def to_list(tensor):
  return tensor.detach().cpu().tolist()


def score_dict_to_string(score_dict):
  return " ".join([("%s:%.2f" % (k, v)) for k, v in score_dict.items()])


def score_dicts_to_latex(score_dicts):
  keys = [k for k in score_dicts[0]]
  return "\n".join([""] + [(
    " & ".join([key] + [("%.2f" % (sd[key])) for sd in score_dicts])
    ) for key in keys])


def eval_classification(model, batch_dict_iter):
  model.eval()
  preds, labels = None, None
  for batch_dict in batch_dict_iter:
    label_id = batch_dict["labels"].detach().cpu().numpy()
    batch_dict.pop("labels")
    with torch.no_grad(): logits = model(**batch_dict)[0]
    pred = logits.detach().cpu().numpy()
    if preds is None: preds, labels = pred, label_id
    else:
      preds = np.append(preds, pred, axis=0)
      labels = np.append(labels, label_id)
  preds = np.argmax(preds, axis=1)
  result = (preds == labels).mean()
  return {"acc": result*100.0}
  

def eval_qa(model, batch_dict_iter, prefix="", **kwargs):

  features = kwargs["all_features"]
  output_dir = kwargs["output_dir"]

  model.eval()
  all_results = []
  for batch_dict, example_indices in batch_dict_iter:
    with torch.no_grad(): outputs = model(**batch_dict)

    for i, example_index in enumerate(example_indices):
      eval_feature = features[example_index.item()]
      unique_id = int(eval_feature.unique_id)
      result = RawResult(unique_id    = unique_id,
                         start_logits = to_list(outputs[0][i]),
                         end_logits   = to_list(outputs[1][i]))
      all_results.append(result)
  
  output_prediction_file = os.path.join(
    output_dir, "predictions_{}.json".format(prefix))
  output_nbest_file = os.path.join(
    output_dir, "nbest_predictions_{}.json".format(prefix))
  if kwargs["version_2_with_negative"]:
    output_null_log_odds_file = os.path.join(
      output_dir, "null_odds_{}.json".format(prefix))
  else: output_null_log_odds_file = None
  
  wrt_pred_kwargs = {
    "all_results": all_results,
    "output_prediction_file": output_prediction_file,
    "output_nbest_file": output_nbest_file,
    "output_null_log_odds_file": output_null_log_odds_file}
  
  for key in inspect.getfullargspec(write_predictions).args:
    if key not in wrt_pred_kwargs:
      wrt_pred_kwargs[key] = kwargs[key]
  
  write_predictions(**wrt_pred_kwargs)

  # Evaluate with the official SQuAD script
  evaluate_options = EVAL_OPTS(
    data_file=kwargs["predict_file"],
    pred_file=output_prediction_file,
    na_prob_file=output_null_log_odds_file,
    out_file="/dev/null")
  results = evaluate_on_squad(evaluate_options)
  return results
