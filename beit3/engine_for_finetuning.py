# --------------------------------------------------------
# Image as a Foreign Language: BEiT Pretraining for Vision and Vision-Language Tasks (https://arxiv.org/abs/2208.10442)
# Github source: https://github.com/microsoft/unilm/tree/master/beit3
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------'

import math
import sys
import json
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.utils import ModelEma
from timm.utils import accuracy, ModelEma
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from datasets import get_sentencepiece_model_for_beit3

import utils


class TaskHandler(object):
    def __init__(self) -> None:
        self.metric_logger = None
        self.split = None

    def train_batch(self, model, **kwargs):
        raise NotImplementedError()

    def eval_batch(self, model, **kwargs):
        raise NotImplementedError()

    def before_eval(self, metric_logger, data_loader, **kwargs):
        self.metric_logger = metric_logger
        self.split = data_loader.dataset.split

    def after_eval(self, **kwargs):
        raise NotImplementedError()


class NLVR2Handler(TaskHandler):
    def __init__(self) -> None:
        super().__init__()
        self.criterion = torch.nn.CrossEntropyLoss()

    def train_batch(self, model, image, image2, language_tokens, padding_mask, label):
        logits = model(
            image_a=image, image_b=image2, 
            text_description=language_tokens, 
            padding_mask=padding_mask)
        acc = (logits.max(-1)[-1] == label).float().mean()
        return {
            "loss": self.criterion(input=logits, target=label), 
            "acc": acc, 
        }

    def eval_batch(self, model, image, image2, language_tokens, padding_mask, label):
        logits = model(
            image_a=image, image_b=image2, 
            text_description=language_tokens, 
            padding_mask=padding_mask)
        batch_size = language_tokens.shape[0]
        acc = (logits.max(-1)[-1] == label).float().sum(0) * 100.0 / batch_size
        self.metric_logger.meters['acc'].update(acc.item(), n=batch_size)
    
    def after_eval(self, **kwargs):
        print('* Acc {acc.global_avg:.3f}'.format(acc=self.metric_logger.acc))
        return {k: meter.global_avg for k, meter in self.metric_logger.meters.items()}, "acc"
    

class ImageNetHandler(TaskHandler):
    def __init__(self, args) -> None:
        super().__init__()
        mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
        if mixup_active:
            # smoothing is handled with mixup label transform
            self.criterion = SoftTargetCrossEntropy()
        elif args.label_smoothing > 0.:
            self.criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

    def train_batch(self, model, image, label):
        logits = model(image=image)
        return {
            "loss": self.criterion(logits, label), 
        }

    def eval_batch(self, model, image, label):
        logits = model(image=image)
        batch_size = image.shape[0]
        acc1, acc5 = accuracy(logits, label, topk=(1, 5))
        self.metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        self.metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    
    def after_eval(self, **kwargs):
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
            .format(top1=self.metric_logger.acc1, top5=self.metric_logger.acc5))
        return {k: meter.global_avg for k, meter in self.metric_logger.meters.items()}, "acc1"


class RetrievalHandler(TaskHandler):
    def __init__(self) -> None:
        super().__init__()
        self.image_feats = []
        self.text_feats = []
        self.image_ids = []
        self.metric_logger = None

    def train_batch(self, model, image, language_tokens, padding_mask, image_id):
        loss, vision_cls, language_cls = model(
            image=image, text_description=language_tokens, padding_mask=padding_mask)
        return {
            "loss": loss, 
        }

    def before_eval(self, metric_logger, **kwargs):
        self.image_feats.clear()
        self.text_feats.clear()
        self.image_ids.clear()
        self.metric_logger = metric_logger

    def eval_batch(self, model, image, language_tokens, padding_mask, image_id):
        vision_cls, _ = model(image=image, only_infer=True)
        _, language_cls = model(
            text_description=language_tokens, padding_mask=padding_mask, only_infer=True)

        self.image_feats.append(vision_cls.clone())
        self.text_feats.append(language_cls.clone())
        self.image_ids.append(image_id.clone())
    
    def after_eval(self, **kwargs):
        image_feats = {}
        for feats, ids in zip(self.image_feats, self.image_ids):
            for i, _idx in enumerate(ids):
                idx = _idx.item()
                if idx not in image_feats:
                    image_feats[idx] = feats[i]
        
        tiids = torch.cat(self.image_ids, dim=0)
        iids = []
        sorted_tensors = []
        for key in sorted(image_feats.keys()):
            sorted_tensors.append(image_feats[key].view(1, -1))
            iids.append(key)

        image_cls_feats = torch.cat(sorted_tensors, dim=0)
        text_cls_feats = torch.cat(self.text_feats, dim=0)

        scores = image_cls_feats @ text_cls_feats.t()
        iids = torch.LongTensor(iids).to(scores.device)

        print("scores: {}".format(scores.size()))
        print("iids: {}".format(iids.size()))
        print("tiids: {}".format(tiids.size()))

        topk10 = scores.topk(10, dim=1)
        topk5 = scores.topk(5, dim=1)
        topk1 = scores.topk(1, dim=1)
        
        topk10_iids = tiids[topk10.indices]
        topk5_iids = tiids[topk5.indices]
        topk1_iids = tiids[topk1.indices]

        tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
        tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
        tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

        topk10 = scores.topk(10, dim=0)
        topk5 = scores.topk(5, dim=0)
        topk1 = scores.topk(1, dim=0)
        topk10_iids = iids[topk10.indices]
        topk5_iids = iids[topk5.indices]
        topk1_iids = iids[topk1.indices]

        ir_r10 = (tiids.unsqueeze(0) == topk10_iids).float().max(dim=0)[0].mean()
        ir_r5 = (tiids.unsqueeze(0) == topk5_iids).float().max(dim=0)[0].mean()
        ir_r1 = (tiids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0].mean()

        eval_result = {
            "tr_r10": tr_r10.item() * 100.0, 
            "tr_r5": tr_r5.item() * 100.0, 
            "tr_r1": tr_r1.item() * 100.0, 
            "ir_r10": ir_r10.item() * 100.0, 
            "ir_r5": ir_r5.item() * 100.0, 
            "ir_r1": ir_r1.item() * 100.0, 
            "average_score": 100.0 * (tr_r1 + tr_r5 + tr_r10 + ir_r1 + ir_r5 + ir_r10).item() / 6.0, 
        }

        print('* Eval result = %s' % json.dumps(eval_result))
        return eval_result, "average_score"


class VQAHandler(TaskHandler):
    def __init__(self) -> None:
        super().__init__()
        self.predictions = []
        self.criterion = nn.BCEWithLogitsLoss(reduction='mean')
        self.label2ans = None

    def train_batch(self, model, image, language_tokens, padding_mask, labels):
        logits = model(
            image=image, question=language_tokens, 
            padding_mask=padding_mask)
        return {
            "loss": self.criterion(input=logits.float(), target=labels.float()) * labels.shape[1], 
        }

    def before_eval(self, metric_logger, data_loader, **kwargs):
        self.predictions.clear()
        self.metric_logger = metric_logger
        self.label2ans = data_loader.dataset.label2ans

    def eval_batch(self, model, image, language_tokens, padding_mask, labels=None, qid=None):
        logits = model(
            image=image, question=language_tokens, 
            padding_mask=padding_mask)
        batch_size = language_tokens.shape[0]
        if labels is not None:
            scores = utils.VQAScore()(logits, labels) * 100.0
            self.metric_logger.meters['score'].update(scores.item(), n=batch_size)
        else:
            _, preds = logits.max(-1)
            for image_id, pred in zip(qid, preds):
                self.predictions.append({
                    "question_id": image_id.item(), 
                    "answer": self.label2ans[pred.item()], 
                })

    def after_eval(self, **kwargs):
        if len(self.predictions) == 0:
            print('* Score {score.global_avg:.3f}'.format(score=self.metric_logger.score))
            return {k: meter.global_avg for k, meter in self.metric_logger.meters.items()}, "score"
        else:
            return self.predictions, "prediction"


class CaptioningHandler(TaskHandler):
    def __init__(self, args) -> None:
        super().__init__()
        self.predictions = []
        self.criterion = utils.BertCaptioningLoss(args.label_smoothing, args.drop_worst_ratio, args.drop_worst_after)
        self.tokenizer = get_sentencepiece_model_for_beit3(args)
        self.num_beams = args.num_beams
        self.max_len = args.num_max_bpe_tokens
        self.length_penalty = args.length_penalty
        self.vocab_size = args.vocab_size

    def train_batch(self, model, image, language_tokens, masked_tokens, language_masked_pos, padding_mask, image_id, global_step):
        logits, _ = model(
            image=image, text_ids=masked_tokens, padding_mask=padding_mask, language_masked_pos=language_masked_pos, image_id=image_id)
        masked_labels = language_tokens[language_masked_pos.bool()]
        score = torch.max(logits, -1)[1].data == masked_labels
        acc = torch.sum(score.float()) / torch.sum(language_masked_pos)
        return {
            "loss": self.criterion(logits, masked_labels, global_step),
            "acc": acc
        }

    def before_eval(self, metric_logger, data_loader, **kwargs):
        self.predictions.clear()
        self.metric_logger = metric_logger

    def eval_batch(self, model, image, image_id=None):
        cur_len = 2
        num_keep_best = 1
        TOPN_PER_BEAM = 3

        batch_size = image.size(0)
        mask_id = self.tokenizer.mask_token_id
        cls_id = self.tokenizer.cls_token_id
        pad_id = self.tokenizer.pad_token_id
        sep_id = self.tokenizer.sep_token_id
        eos_token_ids = [sep_id]

        cls_ids = torch.full(
            (batch_size, 1), cls_id, dtype=torch.long, device=image.device
        )
        mask_ids = torch.full(
            (batch_size, 1), mask_id, dtype=torch.long, device=image.device
        )
        cur_input_ids = torch.cat([cls_ids, mask_ids], dim=1)
        tmp_ids = torch.full(
            (batch_size, self.max_len-1), mask_id, dtype=torch.long, device=image.device
        )
        decoding_results = torch.cat([cls_ids, tmp_ids], dim=1)
        
        # Expand input to num beams
        cur_input_ids = cur_input_ids.unsqueeze(1).expand(batch_size, self.num_beams, cur_len)
        cur_input_ids = cur_input_ids.contiguous().view(batch_size * self.num_beams, cur_len)  # (batch_size * num_beams, cur_len)
        decoding_results = decoding_results.unsqueeze(1).expand(batch_size, self.num_beams, self.max_len)
        decoding_results = decoding_results.contiguous().view(batch_size * self.num_beams, self.max_len)  # (batch_size * num_beams, cur_len)
        image = image.unsqueeze(1).expand(batch_size, self.num_beams, image.size(-3), image.size(-2), image.size(-1))
        image = image.contiguous().view(batch_size * self.num_beams, image.size(-3), image.size(-2), image.size(-1))

        generated_hyps = [
            utils.BeamHypotheses(
                num_keep_best, self.max_len, length_penalty=self.length_penalty, early_stopping=False
            ) for _ in range(batch_size)
        ]
        # scores for each sentence in the beam
        beam_scores = torch.zeros((batch_size, self.num_beams), dtype=torch.float, device=cur_input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)  # shape (batch_size * num_beams,)

        # done sentences
        done = [False for _ in range(batch_size)]
        incremental_state = {}

        while cur_len <= self.max_len:
            next_token_idx = 1
            padding_masks = torch.full(
                cur_input_ids.shape, 0, dtype=torch.long, device=image.device
            )
            input_image = image
            if cur_len != 2:
                input_image = None

            outputs, incremental_state_next = model(
                image=input_image, text_ids=cur_input_ids, language_masked_pos=None,
                padding_mask=padding_masks, text_len=cur_len, incremental_state=incremental_state)
            incremental_state = incremental_state_next

            # assert outputs.shape[1] == token_len
            scores = outputs[:, next_token_idx, :] # (batch_size * num_beams, vocab_size)
            scores = F.log_softmax(scores, dim=-1)  # (batch_size * num_beams, vocab_size)
            assert scores.size() == (batch_size * self.num_beams, self.vocab_size)
            # Add the log prob of the new beams to the log prob of the beginning of the sequence (sum of logs == log of the product)
            _scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)
            # re-organize to group the beam together (we are keeping top hypothesis accross beams)
            _scores = _scores.view(batch_size, self.num_beams * self.vocab_size)  # (batch_size, num_beams * vocab_size)
            next_scores, next_words = torch.topk(_scores, TOPN_PER_BEAM * self.num_beams, dim=1, largest=True, sorted=True)
            assert next_scores.size() == next_words.size() == (batch_size, TOPN_PER_BEAM * self.num_beams)

            # next batch beam content
            # list of (batch_size * num_beams) tuple(next hypothesis score, next word, current position in the batch)
            next_batch_beam = []
            # for each sentence
            for batch_ex in range(batch_size):
                # if we are done with this sentence
                done[batch_ex] = done[batch_ex] or generated_hyps[batch_ex].is_done(next_scores[batch_ex].max().item())
                if done[batch_ex]:
                    next_batch_beam.extend([(0, pad_id, 0)] * self.num_beams)  # pad the batch
                    continue

                # next sentence beam content
                next_sent_beam = []
                for idx, score in zip(next_words[batch_ex], next_scores[batch_ex]):
                    # get beam and word IDs
                    beam_id = idx // self.vocab_size
                    word_id = idx % self.vocab_size
                    # end of sentence, or next word
                    # if word_id.item() in eos_token_ids or cur_len + 1 == max_len:
                    if (word_id.item() in eos_token_ids and cur_len + 1 <= self.max_len) or (cur_len + 1 == self.max_len):
                        generated_hyps[batch_ex].add(
                            decoding_results[batch_ex * self.num_beams + beam_id, :cur_len].clone(), score.item()
                        )
                    else:
                        next_sent_beam.append((score, word_id, batch_ex * self.num_beams + beam_id))
                    # the beam for next step is full
                    if len(next_sent_beam) == self.num_beams:
                        break

                # update next beam content
                if cur_len + 1 == self.max_len:
                    assert len(next_sent_beam) == 0
                else:
                    assert len(next_sent_beam) == self.num_beams

                if len(next_sent_beam) == 0:
                    next_sent_beam = [(0, pad_id, 0)] * self.num_beams  # pad the batch
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == self.num_beams * (batch_ex + 1)
            
            # sanity check / prepare next batch
            assert len(next_batch_beam) == batch_size * self.num_beams
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_words = cur_input_ids.new([x[1] for x in next_batch_beam])
            beam_idx = cur_input_ids.new([x[2] for x in next_batch_beam])

            # re-order batch
            cur_input_ids = cur_input_ids[beam_idx, :]
            decoding_results = decoding_results[beam_idx, :]
            for module in incremental_state:
                for key in incremental_state[module]:
                    result = incremental_state[module][key].index_select(0, beam_idx)
                    incremental_state[module][key] = result[:,:,:-1,:]
            
            next_ids = torch.full(
                (batch_size * self.num_beams, 1), mask_id, dtype=torch.long, device=image.device
            )
            cur_input_ids = torch.cat([beam_words.unsqueeze(1), next_ids], dim=1)
            decoding_results[:, cur_len-1] = beam_words
            # update current length
            cur_len = cur_len + 1
            # stop when we are done with each sentence
            if all(done):
                break
        
        # select the best hypotheses
        tgt_len = torch.ones(batch_size, num_keep_best, dtype=torch.long)
        logprobs = torch.zeros(batch_size, num_keep_best,
                    dtype=torch.float).fill_(-1e5).to(cur_input_ids.device)
        all_best = []

        for i, hypotheses in enumerate(generated_hyps):
                best = []
                hyp_scores = torch.tensor([x[0] for x in hypotheses.hyp])
                _, best_indices = torch.topk(hyp_scores,
                        min(num_keep_best, len(hyp_scores)), largest=True)
                for best_idx, hyp_idx in enumerate(best_indices):
                    conf, best_hyp = hypotheses.hyp[hyp_idx]
                    best.append(best_hyp)
                    logprobs[i, best_idx] = conf
                    tgt_len[i, best_idx] = len(best_hyp) + 1  # +1 for the <EOS> symbol
                all_best.append(best)
        
        # generate target batch, pad to the same length
        decoded = cur_input_ids.new(batch_size, num_keep_best, self.max_len).fill_(pad_id)
        for batch_idx, best in enumerate(all_best):
            for best_idx, hypo in enumerate(best):
                decoded[batch_idx, best_idx, : tgt_len[batch_idx, best_idx] - 1] = hypo
                decoded[batch_idx, best_idx, tgt_len[batch_idx, best_idx] - 1] = eos_token_ids[0]
        
        captions = self.tokenizer.batch_decode(decoded.squeeze(1), skip_special_tokens=True)
        for qid, pred in zip(image_id, captions):
            self.predictions.append({
                "image_id": qid.item(), 
                "caption": pred, 
            })

    def after_eval(self, **kwargs):
        return self.predictions, "prediction"


def get_handler(args):
    if args.task == "nlvr2":
        return NLVR2Handler()
    elif args.task == "vqav2":
        return VQAHandler()
    elif args.task in ("flickr30k", "coco_retrieval"):
        return RetrievalHandler()
    elif args.task in ("coco_captioning", "nocaps"):
        return CaptioningHandler(args)
    elif args.task in ("imagenet"):
        return ImageNetHandler(args)
    else:
        raise NotImplementedError("Sorry, %s is not support." % args.task)


def train_one_epoch(
        model: torch.nn.Module, data_loader: Iterable, 
        optimizer: torch.optim.Optimizer, device: torch.device, 
        handler: TaskHandler, epoch: int, start_steps: int, 
        lr_schedule_values: list, loss_scaler, max_norm: float = 0, 
        update_freq: int = 1, model_ema: Optional[ModelEma] = None, 
        log_writer: Optional[utils.TensorboardLogger] = None, 
        task = None, mixup_fn=None,
):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        global_step = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[global_step] * param_group["lr_scale"]
        # put input data into cuda
        for tensor_key in data.keys():
            data[tensor_key] = data[tensor_key].to(device, non_blocking=True)
            # print("input %s = %s" % (tensor_key, data[tensor_key]))
            if loss_scaler is None and tensor_key.startswith("image"):
                data[tensor_key] = data[tensor_key].half()

        # mixup for imagenet finetuning
        if mixup_fn is not None:
            data["image"], data["label"] = mixup_fn(data["image"], data["label"])
        
        if task in ["coco_captioning", "nocaps"]:
            data["global_step"] = global_step

        if loss_scaler is None:
            results = handler.train_batch(model, **data)
        else:
            with torch.cuda.amp.autocast():
                results = handler.train_batch(model, **data)

        loss = results.pop("loss")
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = utils.get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            kwargs = {
                "loss": loss_value, 
            }
            for key in results:
                kwargs[key] = results[key]
            log_writer.update(head="train", **kwargs)

            kwargs = {
                "loss_scale": loss_scale_value, 
                "lr": max_lr, 
                "min_lr": min_lr, 
                "weight_decay": weight_decay_value, 
                "grad_norm": grad_norm, 
            }
            log_writer.update(head="opt", **kwargs)
            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, handler):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    handler.before_eval(metric_logger=metric_logger, data_loader=data_loader)

    for data in metric_logger.log_every(data_loader, 10, header):
        for tensor_key in data.keys():
            data[tensor_key] = data[tensor_key].to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            handler.eval_batch(model=model, **data)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    return handler.after_eval()
