import torch.nn as nn
from .fairseq_dropout import FairseqDropout

class Lora(nn.Module):
    def __init__(self, u_dim, r, v_dim, shared_u=None, shared_v=None, dropout=0.0):
        super().__init__()
        self.u_dim = u_dim
        self.d_dim = r
        self.v_dim = v_dim
        self.dropout_module = FairseqDropout(dropout, module_name=self.__class__.__name__)

        if shared_u is not None:
            assert shared_u.weight.size() == (r, u_dim)
            self.linear_u = shared_u
        else:
            self.linear_u = nn.Linear(u_dim, r)

        if shared_v is not None:
            assert shared_v.weight.size() == (v_dim, r)
            self.linear_v = shared_v
        else:
            self.linear_v = nn.Linear(r, v_dim)

    def forward(self, x):
        x = self.linear_u(x)
        x = self.dropout_module(x)
        x = self.linear_v(x)
        x = self.dropout_module(x)
        return x

    @classmethod
    def ratio_r(cls, layerid):   # only consider 6 layers: 2 1.5 1 0.5 0.5 0.5
        if layerid < 4:
            return 2 - 0.5 * layerid
        else:
            return 0.5
