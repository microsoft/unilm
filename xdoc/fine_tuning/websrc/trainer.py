from genericpath import exists
import os
import torch.nn as nn
import torch
import logging
from tqdm import tqdm, trange
import timeit
import collections
import json
import math
from bs4 import BeautifulSoup
from copy import deepcopy
import string
import re


from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler)
from transformers import (
    BasicTokenizer,
)

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)


def reorganize_batch_web(args, batch_web):
    dic = {}
    dic['input_ids'] = batch_web[0].cuda()
    dic['attention_mask'] = batch_web[1].cuda()
    dic['token_type_ids'] = batch_web[2].cuda()
    dic['xpath_tags_seq'] = batch_web[3].cuda()
    dic['xpath_subs_seq'] = batch_web[4].cuda()
    dic['start_positions'] = batch_web[5].cuda()
    dic['end_positions'] = batch_web[6].cuda()
    if 'box' in args.embedding_mode:
        dic['bbox'] = batch_web[7].cuda() # new added
    dic['embedding_mode'] = args.embedding_mode
    return dic




def train(args, dataset_web, model, tokenizer):
    # torch.cuda.set_device(args.local_rank)

    # Log when executing on clusters
    try:
        from azureml.core.run import Run
        aml_run = Run.get_context()
    except:
        aml_run = None
        
    # Open tensorboard
    writer = SummaryWriter(f'{args.output_dir}/output/{args.exp_name}')

    # Count batch
    gpu_nums = torch.cuda.device_count()
    batch = args.batch_per_gpu * gpu_nums


    dataloader_web = DataLoader(
        dataset_web, batch_size=batch, num_workers=args.num_workers, pin_memory=False, shuffle=True,
        
    )

    # Get warmup steps
    total_step = args.epoch * len(dataloader_web)
    warmup_steps = int(args.warmup_ratio * total_step)

    # Prepare optimizers
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_step
    )

    # Transfer the parameters to cuda
    model = model.cuda()


    # Prepare fp16
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level
        )
        logging.info('Successfully load fp16 mode')

    # Parallel or Distribute
    if gpu_nums > 1:
        model = torch.nn.DataParallel(model)


    # Record some training info
    logging.info("***** Running training *****")
    # logging.info("  Num examples in dataset_doc = %d", len(dataset_doc))
    logging.info("  Num examples in dataset_web = %d", len(dataset_web))
    # logging.info("  Num steps for each epoch for doc = %d", len(dataloader_doc))
    logging.info("  Num steps for each epoch for web = %d", len(dataloader_web))
    logging.info("  Num Epochs = %d", args.epoch)
    logging.info(
        "  Instantaneous batch size per GPU = %d", args.batch_per_gpu
    )
    logging.info("  Total optimization steps = %d", total_step)

    # Start training
    model.zero_grad()
    train_iterator = trange(
        0,
        int(args.epoch),
        desc="Epoch",
    )

    global_step = 0
    for now_epoch, _ in enumerate(tqdm(train_iterator, desc="Iteration")): # tqdm for epoch

        # epoch_iterator_doc = iter(dataloader_doc)
        epoch_iterator_web = iter(dataloader_web)

        min_step = len(epoch_iterator_web)

        for now_step in tqdm(range(min_step), desc="Iteration"): # tqdm for step

            # batch_doc = epoch_iterator_doc.next()
            batch_web = epoch_iterator_web.next()
            batch_web = reorganize_batch_web(args, batch_web)

            model.train()

            # loss_doc = model(**batch_doc)[0]
            loss_web = model(**batch_web)[0]
            loss = loss_web

            if gpu_nums > 1:
                loss = loss.mean()
                # loss_doc = loss_doc.mean()
                loss_web = loss_web.mean()
            
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if args.fp16:
                torch.nn.utils.clip_grad_norm_(
                    amp.master_params(optimizer), args.max_grad_norm
                )
            else:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm
                )

            if global_step % args.accumulation == 0:
                optimizer.step()
                model.zero_grad()
            scheduler.step()

            global_step += 1
           
            
            if global_step % args.log_step == 0:
                logging.info(f'epoch: {now_epoch} | step: {now_step+1} | total_step: {global_step} | loss: {loss} | lr: {scheduler.get_lr()[0]}')
                writer.add_scalar('loss', loss, global_step//args.log_step)
                # writer.add_scalar('loss_doc', loss_doc, global_step//args.log_step)
                writer.add_scalar('loss_web', loss_web, global_step//args.log_step)
                writer.add_scalar('lr', scheduler.get_lr()[0], global_step//args.log_step)
                if aml_run is not None:
                    aml_run.log('loss', loss.item())
                    # aml_run.log('loss_doc', loss_doc.item())
                    aml_run.log('loss_web', loss_web.item())
                    aml_run.log('lr', scheduler.get_lr()[0])

            if global_step % args.save_step == 0:
                # Save model checkpoint
                output_dir = os.path.join(args.output_dir, 'output', args.exp_name, f'step-{global_step}')
                os.makedirs(output_dir, exist_ok=True)
                
                model_to_save = (
                    model.module if hasattr(model, "module") else model
                )  # Take care of distributed/parallel training
                model_to_save.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)

                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                logging.info("Saving model checkpoint to %s", output_dir)

                torch.save(
                    optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt")
                )
                torch.save(
                    scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt")
                )
                logging.info(
                    "Saving optimizer and scheduler states to %s", output_dir
                )

            if global_step % 1000 == 0:
                # eval
                print('Start eval!')
                from data.datasets.websrc import get_websrc_dataset
                dataset_web, examples, features = get_websrc_dataset(args, tokenizer, evaluate=True, output_examples=True)
                evaluate(args, dataset_web, examples, features, model, tokenizer, global_step)


         



RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return ns_text, ns_to_s_map

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        # if verbose_logging:
        #     logging.info(
        #         "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logging.info("Length not equal after stripping spaces: '%s' vs '%s'",
                        orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logging.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logging.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text



def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs



class EvalOpts:
    r"""
    The options which the matrix evaluation process needs.
    
    Arguments:
        data_file (str): the SQuAD-style json file of the dataset in evaluation.
        root_dir (str): the root directory of the raw WebSRC dataset, which contains the HTML files.
        pred_file (str): the prediction file which contain the best predicted answer text of each question from the
                         model.
        tag_pred_file (str): the prediction file which contain the best predicted answer tag id of each question from
                             the model.
        result_file (str): the file to write down the matrix evaluation results of each question.
        out_file (str): the file to write down the final matrix evaluation results of the whole dataset.
    """
    def __init__(self, data_file, root_dir, pred_file, tag_pred_file, result_file='', out_file=""):
        self.data_file = data_file
        self.root_dir = root_dir
        self.pred_file = pred_file
        self.tag_pred_file = tag_pred_file
        self.result_file = result_file
        self.out_file = out_file


def write_predictions(all_examples, all_features, all_results, n_best_size, max_answer_length, do_lower_case,
                      output_prediction_file, output_tag_prediction_file,
                      output_nbest_file, verbose_logging, tokenizer):
    r"""
    Compute and write down the final results, including the n best results.

    Arguments:
        all_examples (list[SRCExample]): all the SRC Example of the dataset; note that we only need it to provide the
                                         mapping from example index to the question-answers id.
        all_features (list[InputFeatures]): all the features for the input doc spans.
        all_results (list[RawResult]): all the results from the models.
        n_best_size (int): the number of the n best buffer and the final n best result saved.
        max_answer_length (int): constrain the model to predict the answer no longer than it.
        do_lower_case (bool): whether the model distinguish upper and lower case of the letters.
        output_prediction_file (str): the file which the best answer text predictions will be written to.
        output_tag_prediction_file (str): the file which the best answer tag predictions will be written to.
        output_nbest_file (str): the file which the n best answer predictions including text, tag, and probabilities
                                 will be written to.
        verbose_logging (bool): if true, all of the warnings related to data processing will be printed.
    """
    logging.info("Writing predictions to: %s" % output_prediction_file)
    logging.info("Writing nbest to: %s" % output_nbest_file)

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit", "tag_ids"])

    all_predictions = collections.OrderedDict()
    all_tag_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []

        for (feature_index, feature) in enumerate(features): 
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    tag_ids = set(feature.token_to_tag_index[start_index: end_index + 1])
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index],
                            tag_ids=list(tag_ids)))
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit", "tag_ids"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = _get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit,
                    tag_ids=pred.tag_ids))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0, tag_ids=[-1]))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            output["tag_ids"] = entry.tag_ids
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        best = nbest_json[0]["text"].split()
        best = ' '.join([w for w in best
                         if (w[0] != '<' or w[-1] != '>')
                         and w != "<end-of-node>"
                         and w != tokenizer.sep_token
                         and w != tokenizer.cls_token])
        all_predictions[example.qas_id] = best
        all_tag_predictions[example.qas_id] = nbest_json[0]["tag_ids"]
        all_nbest_json[example.qas_id] = nbest_json

    with open(output_prediction_file, "w+") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    with open(output_nbest_file, "w+") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    with open(output_tag_prediction_file, 'w+') as writer:
        writer.write(json.dumps(all_tag_predictions, indent=4) + '\n')
    return



def make_qid_to_has_ans(dataset):
    r"""
    Pick all the questions which has answer in the dataset and return the list.
    """
    qid_to_has_ans = {}
    for domain in dataset:
        for w in domain['websites']:
            for qa in w['qas']:
                qid_to_has_ans[qa['id']] = bool(qa['answers'])
    return qid_to_has_ans



def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact(a_gold, a_pred):
    r"""
    Calculate the exact match.
    """
    if normalize_answer(a_gold) == normalize_answer(a_pred):
        return 1
    return 0



def get_raw_scores(dataset, preds, tag_preds, root_dir):
    r"""
    Calculate all the three matrix (exact match, f1, POS) for each question.

    Arguments:
        dataset (dict): the dataset in use.
        preds (dict): the answer text prediction for each question in the dataset.
        tag_preds (dict): the answer tags prediction for each question in the dataset.
        root_dir (str): the base directory for the html files.

    Returns:
        tuple(dict, dict, dict): exact match, f1, pos scores for each question.
    """
    exact_scores = {}
    f1_scores = {}
    pos_scores = {}
    for websites in dataset:
        for w in websites['websites']:
            f = os.path.join(root_dir, websites['domain'], w['page_id'][0:2], 'processed_data',
                             w['page_id'] + '.html')
            for qa in w['qas']:
                qid = qa['id']
                gold_answers = [a['text'] for a in qa['answers']
                                if normalize_answer(a['text'])]
                gold_tag_answers = [a['element_id'] for a in qa['answers']]
                additional_tag_information = [a['answer_start'] for a in qa['answers']]
                if not gold_answers:
                    # For unanswerable questions, only correct answer is empty string
                    gold_answers = ['']
                if qid not in preds:
                    print('Missing prediction for %s' % qid)
                    continue
                a_pred, t_pred = preds[qid], tag_preds[qid]
                # Take max over all gold answers
                exact_scores[qid] = max(compute_exact(a, a_pred) for a in gold_answers)
                f1_scores[qid] = max(compute_f1(a, a_pred) for a in gold_answers)
                pos_scores[qid] = max(compute_pos(f, t, a, t_pred)
                                      for t, a in zip(gold_tag_answers, additional_tag_information))
    return exact_scores, f1_scores, pos_scores

def get_tokens(s):
    r"""
    Get the word list in the input.
    """
    if not s:
        return []
    return normalize_answer(s).split()



def compute_f1(a_gold, a_pred):
    r"""
    Calculate the f1 score.
    """
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def compute_pos(f, t_gold, addition, t_pred):
    r"""
    Calculate the POS score.

    Arguments:
        f (str): the html file on which the question is based.
        t_gold (int): the gold answer tag id provided by the dataset (the value correspond to the key element_id).
        addition (int): the addition information used for yes/no question provided by the dataset (the value
                        corresponding to the key answer_start).
        t_pred (list[int]): the tag ids of the tags corresponding the each word in the predicted answer.
    Returns:
        float: the POS score.
    """
    h = BeautifulSoup(open(f), "lxml")
    p_gold, e_gold = set(), h.find(tid=t_gold)
    if e_gold is None:
        if len(t_pred) != 1:
            return 0
        else:
            t = t_pred[0]
            e_pred, e_prev = h.find(tid=t), h.find(tid=t-1)
            if (e_pred is not None) or (addition == 1 and e_prev is not None) or\
                    (addition == 0 and e_prev is None):
                return 0
            else:
                return 1
    else:
        p_gold.add(e_gold['tid'])
        for e in e_gold.parents:
            if int(e['tid']) < 2:
                break
            p_gold.add(e['tid'])
        p = None
        for t in t_pred:
            p_pred, e_pred = set(), h.find(tid=t)
            if e_pred is not None:
                p_pred.add(e_pred['tid'])
                if e_pred.name != 'html':
                    for e in e_pred.parents:
                        if int(e['tid']) < 2:
                            break
                        p_pred.add(e['tid'])
            else:
                p_pred.add(str(t))
            if p is None:
                p = p_pred
            else:
                p = p & p_pred
        return len(p_gold & p) / len(p_gold | p)




def make_pages_list(dataset):
    r"""
    Record all the pages which appears in the dataset and return the list.
    """
    pages_list = []
    last_page = None
    for domain in dataset:
        for w in domain['websites']:
            for qa in w['qas']:
                if last_page != qa['id'][:4]:
                    last_page = qa['id'][:4]
                    pages_list.append(last_page)
    return pages_list



def make_eval_dict(exact_scores, f1_scores, pos_scores, qid_list=None):
    r"""
    Make the dictionary to show the evaluation results.
    """
    if qid_list is None:
        total = len(exact_scores)
        return collections.OrderedDict([
            ('exact', 100.0 * sum(exact_scores.values()) / total),
            ('f1', 100.0 * sum(f1_scores.values()) / total),
            ('pos', 100.0 * sum(pos_scores.values()) / total),
            ('total', total),
        ])
    else:
        total = len(qid_list)
        if total == 0:
            return collections.OrderedDict([
                ('exact', 0),
                ('f1', 0),
                ('pos', 0),
                ('total', 0),
            ])
        return collections.OrderedDict([
            ('exact', 100.0 * sum(exact_scores[k] for k in qid_list) / total),
            ('f1', 100.0 * sum(f1_scores[k] for k in qid_list) / total),
            ('pos', 100.0 * sum(pos_scores[k] for k in qid_list) / total),
            ('total', total),
        ])


def merge_eval(main_eval, new_eval, prefix):
    for k in new_eval:
        main_eval['%s_%s' % (prefix, k)] = new_eval[k]


def evaluate_on_squad(opts):
    with open(opts.data_file) as f:
        dataset_json = json.load(f)
        dataset = dataset_json['data']
    if isinstance(opts.pred_file, str):
        with open(opts.pred_file) as f:
            preds = json.load(f)
    else:
        preds = opts.pred_file
    if isinstance(opts.tag_pred_file, str):
        with open(opts.tag_pred_file) as f:
            tag_preds = json.load(f)
    else:
        tag_preds = opts.tag_pred_file
    qid_to_has_ans = make_qid_to_has_ans(dataset)
    has_ans_qids = [k for k, v in qid_to_has_ans.items() if v]
    no_ans_qids = [k for k, v in qid_to_has_ans.items() if not v]
    exact, f1, pos = get_raw_scores(dataset, preds, tag_preds, opts.root_dir)
    out_eval = make_eval_dict(exact, f1, pos)
    if has_ans_qids:
        has_ans_eval = make_eval_dict(exact, f1, pos, qid_list=has_ans_qids)
        merge_eval(out_eval, has_ans_eval, 'HasAns')
    if no_ans_qids:
        no_ans_eval = make_eval_dict(exact, f1, pos, qid_list=no_ans_qids)
        merge_eval(out_eval, no_ans_eval, 'NoAns')
    print(json.dumps(out_eval, indent=2))
    pages_list, write_eval = make_pages_list(dataset), deepcopy(out_eval)
    for p in pages_list:
        pages_ans_qids = [k for k, _ in qid_to_has_ans.items() if p in k]
        page_eval = make_eval_dict(exact, f1, pos, qid_list=pages_ans_qids)
        merge_eval(write_eval, page_eval, p)
    if opts.result_file:
        with open(opts.result_file, 'w') as f:
            w = {}
            for k, v in qid_to_has_ans.items():
                w[k] = {'exact': exact[k], 'f1': f1[k], 'pos': pos[k]}
            json.dump(w, f)
    if opts.out_file:
        with open(opts.out_file, 'w') as f:
            json.dump(write_eval, f)
    print('****** result ******')
    print(out_eval)
    return out_eval



def evaluate(args, dataset_web, examples, features, model, tokenizer, step=0):

    gpu_nums = torch.cuda.device_count()
    batch = args.batch_per_gpu * gpu_nums

    eval_sampler = SequentialSampler(dataset_web)
    eval_dataloader = DataLoader(dataset_web, sampler=eval_sampler, batch_size=batch, num_workers=8)

    # Eval!
    logging.info("***** Running evaluation *****")
    logging.info("  Num examples = %d", len(dataset_web))
    logging.info("  Batch size = %d", batch)

    model = model.cuda()

    all_results = []
    start_time = timeit.default_timer()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.cuda() for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'xpath_tags_seq': batch[4],
                      'xpath_subs_seq': batch[5],
                      }
            feature_indices = batch[3]
            outputs = model(**inputs)

        for i, feature_index in enumerate(feature_indices):
            eval_feature = features[feature_index.item()]
            unique_id = int(eval_feature.unique_id)
            result = RawResult(unique_id=unique_id,
                               start_logits=to_list(outputs[0][i]),
                               end_logits=to_list(outputs[1][i]))
            all_results.append(result)

    eval_time = timeit.default_timer() - start_time
    logging.info("  Evaluation done in total %f secs (%f sec per example)", eval_time, eval_time / len(dataset_web))

    # Compute predictions
    # output_dir = os.path.join(args.output_dir, 'output', args.exp_name, f'step-{global_step}')

    output_prediction_file = os.path.join(args.output_dir,"output", args.exp_name, f"predictions_{step}.json")
    output_tag_prediction_file = os.path.join(args.output_dir,"output", args.exp_name, f"tag_predictions_{step}.json")
    output_nbest_file = os.path.join(args.output_dir,"output", args.exp_name, f"nbest_predictions_{step}.json")
    output_result_file = os.path.join(args.output_dir,"output", args.exp_name, f"qas_eval_results_{step}.json")
    output_file = os.path.join(args.output_dir,"output", args.exp_name, f"eval_matrix_results_{step}")

    write_predictions(examples, features, all_results, args.n_best_size, args.max_answer_length, args.do_lower_case,
                      output_prediction_file, output_tag_prediction_file, output_nbest_file, args.verbose_logging,
                      tokenizer)

    # Evaluate
    evaluate_options = EvalOpts(data_file=args.web_eval_file,
                                root_dir=args.root_dir,
                                pred_file=output_prediction_file,
                                tag_pred_file=output_tag_prediction_file,
                                result_file=output_result_file,
                                out_file=output_file)
    results = evaluate_on_squad(evaluate_options)
    return results
