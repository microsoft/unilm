# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
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
"""PyTorch optimization for BERT model."""

import math
import torch
from torch.optim import Optimizer
from torch.optim.optimizer import required
from torch.nn.utils import clip_grad_norm_

from collections import defaultdict
from torch._six import container_abcs
from copy import deepcopy
from itertools import chain


def warmup_cosine(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 0.5 * (1.0 + torch.cos(math.pi * x))


def warmup_constant(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return max((x-1.)/(warmup-1.), 0)


SCHEDULES = {
    'warmup_cosine': warmup_cosine,
    'warmup_constant': warmup_constant,
    'warmup_linear': warmup_linear,
}


class BertAdam(Optimizer):
    """Implements BERT version of Adam algorithm with weight decay fix.
    Params:
        lr: learning rate
        warmup: portion of t_total for the warmup, -1  means no warmup. Default: -1
        t_total: total number of training steps for the learning
            rate schedule, -1  means constant learning rate. Default: -1
        schedule: schedule to use for the warmup (see above). Default: 'warmup_linear'
        b1: Adams b1. Default: 0.9
        b2: Adams b2. Default: 0.999
        e: Adams epsilon. Default: 1e-6
        weight_decay: Weight decay. Default: 0.01
        max_grad_norm: Maximum norm for the gradients (-1 means no clipping). Default: 1.0
    """

    def __init__(self, params, lr=required, warmup=-1, t_total=-1, schedule='warmup_linear', b1=0.9, b2=0.999, e=1e-6, weight_decay=0.01, max_grad_norm=1.0):
        if lr is not required and lr < 0.0:
            raise ValueError(
                "Invalid learning rate: {} - should be >= 0.0".format(lr))
        if schedule not in SCHEDULES:
            raise ValueError("Invalid schedule parameter: {}".format(schedule))
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError(
                "Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(warmup))
        if not 0.0 <= b1 < 1.0:
            raise ValueError(
                "Invalid b1 parameter: {} - should be in [0.0, 1.0[".format(b1))
        if not 0.0 <= b2 < 1.0:
            raise ValueError(
                "Invalid b2 parameter: {} - should be in [0.0, 1.0[".format(b2))
        if not e >= 0.0:
            raise ValueError(
                "Invalid epsilon value: {} - should be >= 0.0".format(e))
        defaults = dict(lr=lr, schedule=schedule, warmup=warmup, t_total=t_total,
                        b1=b1, b2=b2, e=e, weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm)
        super(BertAdam, self).__init__(params, defaults)

    def get_lr(self):
        lr = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    return [0]
                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(
                        state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']
                lr.append(lr_scheduled)
        return lr

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['next_m'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['next_v'] = torch.zeros_like(p.data)

                next_m, next_v = state['next_m'], state['next_v']
                beta1, beta2 = group['b1'], group['b2']

                # Add grad clipping
                if group['max_grad_norm'] > 0:
                    clip_grad_norm_(p, group['max_grad_norm'])

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                next_m.mul_(beta1).add_(1 - beta1, grad)
                next_v.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                update = next_m / (next_v.sqrt() + group['e'])

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                if group['weight_decay'] > 0.0:
                    update += group['weight_decay'] * p.data

                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(
                        state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']

                update_with_lr = lr_scheduled * update
                p.data.add_(-update_with_lr)

                state['step'] += 1

                # step_size = lr_scheduled * math.sqrt(bias_correction2) / bias_correction1
                # No bias correction
                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']

        return loss


class BertAdamFineTune(BertAdam):
    def __init__(self, params, lr=required, warmup=-1, t_total=-1, schedule='warmup_linear', b1=0.9, b2=0.999, e=1e-6, weight_decay=0.01, max_grad_norm=1.0):
        self.init_param_group = []
        super(BertAdamFineTune, self).__init__(params, lr, warmup,
                                               t_total, schedule, b1, b2, e, weight_decay, max_grad_norm)

    def save_init_param_group(self, param_groups, name_groups, missing_keys):
        self.init_param_group = []
        for group, name in zip(param_groups, name_groups):
            if group['weight_decay'] > 0.0:
                init_p_list = []
                for p, n in zip(group['params'], name):
                    init_p = p.data.clone().detach()
                    if any(mk in n for mk in missing_keys):
                        print("[no finetuning weight decay]", n)
                        # should use the original weight decay
                        init_p.zero_()
                    init_p_list.append(init_p)
                self.init_param_group.append(init_p_list)
            else:
                # placeholder
                self.init_param_group.append([])

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for i_group, group in enumerate(self.param_groups):
            for i_p, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['next_m'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['next_v'] = torch.zeros_like(p.data)

                next_m, next_v = state['next_m'], state['next_v']
                beta1, beta2 = group['b1'], group['b2']

                # Add grad clipping
                if group['max_grad_norm'] > 0:
                    clip_grad_norm_(p, group['max_grad_norm'])

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                next_m.mul_(beta1).add_(1 - beta1, grad)
                next_v.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                update = next_m / (next_v.sqrt() + group['e'])

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                if group['weight_decay'] > 0.0:
                    if self.init_param_group:
                        update += group['weight_decay'] * \
                            (2.0 * p.data -
                             self.init_param_group[i_group][i_p])
                    else:
                        update += group['weight_decay'] * p.data

                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(
                        state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']

                update_with_lr = lr_scheduled * update
                p.data.add_(-update_with_lr)

                state['step'] += 1

                # step_size = lr_scheduled * math.sqrt(bias_correction2) / bias_correction1
                # No bias correction
                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']

        return loss

    def load_state_dict_subset_finetune(self, state_dict, num_load_group):
        r"""Loads the optimizer state.

        Arguments:
            state_dict (dict): optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # deepcopy, to be consistent with module API
        state_dict = deepcopy(state_dict)
        # Validate the state_dict
        groups = self.param_groups
        saved_groups = state_dict['param_groups']

        if len(groups) < num_load_group or len(saved_groups) < num_load_group:
            raise ValueError("loaded state dict has a different number of "
                             "parameter groups")
        param_lens = (len(g['params']) for g in groups[:num_load_group])
        saved_lens = (len(g['params']) for g in saved_groups[:num_load_group])
        if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
            raise ValueError("loaded state dict contains a parameter group "
                             "that doesn't match the size of optimizer's group")

        # Update the state
        id_map = {old_id: p for old_id, p in
                  zip(chain(*(g['params'] for g in saved_groups[:num_load_group])),
                      chain(*(g['params'] for g in groups[:num_load_group])))}

        def cast(param, value):
            r"""Make a deep copy of value, casting all tensors to device of param."""
            if isinstance(value, torch.Tensor):
                # Floating-point types are a bit special here. They are the only ones
                # that are assumed to always match the type of params.
                if param.is_floating_point():
                    value = value.to(param.dtype)
                value = value.to(param.device)
                return value
            elif isinstance(value, dict):
                return {k: cast(param, v) for k, v in value.items()}
            elif isinstance(value, container_abcs.Iterable):
                return type(value)(cast(param, v) for v in value)
            else:
                return value

        # Copy state assigned to params (and cast tensors to appropriate types).
        # State that is not assigned to params is copied as is (needed for
        # backward compatibility).
        state = defaultdict(dict)
        for k, v in state_dict['state'].items():
            if k in id_map:
                param = id_map[k]
                state[param] = cast(param, v)
            else:
                state[k] = v
        # handle additional params
        for k, v in self.state:
            if k not in state:
                state[k] = v

        # do not change groups: {'weight_decay': 0.01, 'lr': 9.995e-06, 'schedule': 'warmup_linear', 'warmup': 0.1, 't_total': 400000, 'b1': 0.9, 'b2': 0.999, 'e': 1e-06, 'max_grad_norm': 1.0, 'params': [...]}
        # # Update parameter groups, setting their 'params' value
        # def update_group(group, new_group):
        #     new_group['params'] = group['params']
        #     return new_group
        # param_groups = [
        #     update_group(g, ng) for g, ng in zip(groups[:num_load_group], saved_groups[:num_load_group])]
        # # handle additional params
        # param_groups.extend(groups[num_load_group:])

        self.__setstate__({'state': state, 'param_groups': groups})


def find_state_dict_subset_finetune(org_state_dict, org_name_list, no_decay, param_optimizer):
    # only use the bert encoder and embeddings
    want_name_set = set()
    for n in org_name_list:
        if ('bert.encoder' in n) or ('bert.embeddings' in n):
            want_name_set.add(n)
    # original: name to pid, pid to name
    org_grouped_names = [[n for n in org_name_list if not any(nd in n for nd in no_decay)],
                         [n for n in org_name_list if any(nd in n for nd in no_decay)]]
    org_n2id, org_id2n = {}, {}
    for ng, pg in zip(org_grouped_names, org_state_dict['param_groups']):
        for n, pid in zip(ng, pg['params']):
            org_n2id[n] = pid
            org_id2n[pid] = n
    # group by: whether pretrained; whether weight decay
    g_np_list = [
        [(n, p) for n, p in param_optimizer if n in want_name_set and not any(
            nd in n for nd in no_decay)],
        [(n, p) for n, p in param_optimizer if n in want_name_set and any(
            nd in n for nd in no_decay)],
        [(n, p) for n, p in param_optimizer if n not in want_name_set and not any(
            nd in n for nd in no_decay)],
        [(n, p) for n, p in param_optimizer if n not in want_name_set and any(
            nd in n for nd in no_decay)],
    ]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in g_np_list[0]], 'weight_decay': 0.01},
        {'params': [p for n, p in g_np_list[1]], 'weight_decay': 0.0},
        {'params': [p for n, p in g_np_list[2]], 'weight_decay': 0.01},
        {'params': [p for n, p in g_np_list[3]], 'weight_decay': 0.0}
    ]
    new_state_dict = {}
    # regroup the original state_dict
    new_state_dict['state'] = {pid: v for pid, v in org_state_dict['state'].items(
    ) if pid not in org_id2n or org_id2n[pid] in want_name_set}
    # reset step count to 0
    for pid, st in new_state_dict['state'].items():
        st['step'] = 0

    def _filter_group(group, g_np_list, i, org_n2id):
        packed = {k: v for k, v in group.items() if k != 'params'}
        packed['params'] = [pid for pid in group['params']
                            if pid in org_id2n and org_id2n[pid] in want_name_set]
        assert len(g_np_list[i]) == len(packed['params'])
        # keep them the same order
        packed['params'] = [org_n2id[n] for n, p in g_np_list[i]]
        return packed
    new_state_dict['param_groups'] = [_filter_group(
        g, g_np_list, i, org_n2id) for i, g in enumerate(org_state_dict['param_groups'])]
    return new_state_dict, optimizer_grouped_parameters
