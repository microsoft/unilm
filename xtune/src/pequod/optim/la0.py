from collections import defaultdict

import torch
from torch.optim.optimizer import Optimizer


class Lookahead0Wrapper(Optimizer):
    r"""Implements a Lookahead wrapper around a given optimizer
    """

    def __init__(self, optimizer, la_steps, la_alpha=0.5):
        self.optimizer = optimizer
        self._la_step = 0 # counter for inner optimizer
        self.la_alpha = la_alpha
        self._total_la_steps = la_steps

        self.state = defaultdict(dict)

        # Cache the current optimizer parameters
        for group in optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['cached_params'] = torch.zeros_like(p.data)
                param_state['cached_params'].copy_(p.data)

    def __getstate__(self):
        return self.optimizer.__getstate__()

    def __setstate__(self, state):
        self.optimizer.__setstate__(state)
    
    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def step(self, closure=None):
        """Performs a single Lookahead optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = self.optimizer.step(closure)
        self._la_step += 1

        if self._la_step >= self._total_la_steps:
            self._la_step = 0
            # Lookahead and cache the current optimizer parameters
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    p.data.mul_(self.la_alpha).add_(1 - self.la_alpha, param_state['cached_params'])
                    # param_state['cached_params'].copy_(p.data)
        return loss
