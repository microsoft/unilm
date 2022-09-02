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

import math
import sys
from typing import Iterable

import torch
import torch.nn as nn

import utils

def train_one_epoch(model: torch.nn.Module, 
                            data_loader: Iterable, 
                            optimizer: torch.optim.Optimizer,
                            device: torch.device, 
                            epoch: int, 
                            loss_scaler, 
                            clip_grad: float = 0,
                            log_writer=None, 
                            lr_scheduler=None, 
                            start_steps=None,
                            lr_schedule_values=None,
                            args=None,
                            ):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
        
    if hasattr(model.module, 'quantize'):
        try:
            model.module.quantize.reset_cluster_size(device)
            print("Reset the codebook statistic info in quantizer before each epoch")
        except:
            pass
        
    for step, (batch, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group.get("lr_scale", 1.0)
        images = batch.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=True):
            loss, log_loss = model(images)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value), force=True)
            utils.save_nan_model(args, model)
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=clip_grad,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]
        
        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        
        new_log_loss = {k.split('/')[-1]:v for k, v in log_loss.items() if k not in ['total_loss']}
        metric_logger.update(**new_log_loss)

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
            log_writer.update(**new_log_loss, head="train/loss")

            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.update(loss_scale=loss_scale_value, head="opt")

            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    # stat the codebook usage information
    if hasattr(model.module, 'quantize'):
        try:
            codebook_cluster_size = model.module.quantize._codebook.cluster_size
        except:
            codebook_cluster_size = model.module.quantize.cluster_size
        zero_cnt = (codebook_cluster_size == 0).sum().item()
        train_stat = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        train_stat['Unused_code'] = zero_cnt
        print(f"Unused code in codebook: {zero_cnt}")
        return train_stat
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, log_writer=None, epoch=None, args=None):

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Validation:'

    # switch to evaluation mode
    model.eval()

    if hasattr(model.module, 'quantize'):
        try:
            model.module.quantize.reset_cluster_size(device)
            print("Reset the codebook statistic info in quantizer before testing")
        except:
            pass

    for step, (batch, extra_info) in enumerate(metric_logger.log_every(data_loader, 10, header)):

        images = batch.to(device, non_blocking=True)
        loss, log_loss = model(images)

        metric_logger.update(loss=loss.item())

        new_log_loss = {k.split('/')[-1]:v for k, v in log_loss.items() if k not in ['total_loss']}
        metric_logger.update(**new_log_loss)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # stat the codebook usage information
    if hasattr(model, 'module') and hasattr(model.module, 'quantize'):
        try:
            codebook_cluster_size = model.module.quantize._codebook.cluster_size
        except:
            codebook_cluster_size = model.module.quantize.cluster_size
        zero_cnt = (codebook_cluster_size == 0).sum().item()
        test_stat = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        test_stat['unused_code'] = zero_cnt
        print(f"Unused code in codebook: {zero_cnt}")
        return test_stat

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def calculate_codebook_usage(data_loader, model, device, log_writer=None, epoch=None, args=None):

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Calculating codebook usage:'

    # switch to evaluation mode
    model.eval()
    
    codebook_num = args.codebook_n_emd
    codebook_cnt = torch.zeros(codebook_num, dtype=torch.float64).to(device)

    for step, (images, _) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        images = images.to(device, non_blocking=True)

        outputs = utils.get_model(model).get_tokens(images)['token'].view(-1)
        
        outputs_gather_list = [torch.zeros_like(outputs) for _ in range(utils.get_world_size())]
        torch.distributed.all_gather(outputs_gather_list, outputs)
        all_tokens = torch.cat(outputs_gather_list, dim=0).view(-1) # [B * N * Ngpu, ]
        
        codebook_cnt += torch.bincount(all_tokens, minlength=codebook_num)

    # statistic
    zero_cnt = (codebook_cnt == 0).sum() # 0
    print(f"STAT:  {zero_cnt} tokens ({(zero_cnt / codebook_num) * 100}%) never are used in this codebook.")