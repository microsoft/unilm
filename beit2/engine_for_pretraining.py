# --------------------------------------------------------
# BEiT v2: Masked Image Modeling with Vector-Quantized Visual Tokenizers (https://arxiv.org/abs/2208.06366)
# Github source: https://github.com/microsoft/unilm/tree/master/beitv2
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Zhiliang Peng
# Based on BEiT, timm, DeiT and DINO code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'

from cgitb import enable
import math
import sys
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

def train_one_epoch(model: torch.nn.Module, vqkd: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None, args=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    loss_fn = nn.CrossEntropyLoss()

    for step, (batch, extra_info) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples, images, bool_masked_pos = batch
        images = images.to(device, non_blocking=True)
        samples = samples.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True)

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                input_ids = vqkd.get_codebook_indices(images)
            bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)
            labels = input_ids[bool_masked_pos]

        with torch.cuda.amp.autocast(): # enabled=False
            outputs = model(samples, bool_masked_pos=bool_masked_pos)

            if isinstance(outputs, list):
                loss_1 = loss_fn(input=outputs[0], target=labels)
                loss_2 = loss_fn(input=outputs[1], target=labels)
                loss = loss_1 + loss_2 
            else:
                loss = loss_fn(input=outputs, target=labels)


        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training at rank {utils.get_rank()}", force=True)
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()
        
        if isinstance(outputs, list):
            mlm_acc_1 = (outputs[0].max(-1)[1] == labels).float().mean().item()
            mlm_acc_2 = (outputs[1].max(-1)[1] == labels).float().mean().item()
            metric_logger.update(mlm_acc_1=mlm_acc_1)
            metric_logger.update(mlm_acc_2=mlm_acc_2)
            metric_logger.update(loss_1=loss_1.item())
            metric_logger.update(loss_2=loss_2.item())

            if log_writer is not None:
                log_writer.update(mlm_acc_1=mlm_acc_1, head="loss")
                log_writer.update(mlm_acc_2=mlm_acc_2, head="loss")
                log_writer.update(loss_1=loss_1.item(), head="loss")
                log_writer.update(loss_2=loss_2.item(), head="loss")
        else:
            mlm_acc = (outputs.max(-1)[1] == labels).float().mean().item()
            metric_logger.update(mlm_acc=mlm_acc)
            if log_writer is not None:
                log_writer.update(mlm_acc=mlm_acc, head="loss")

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
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

