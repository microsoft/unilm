# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on OpenAI DALL-E and lucidrains' DALLE-pytorch code bases
# https://github.com/openai/DALL-E
# https://github.com/lucidrains/DALLE-pytorch
# --------------------------------------------------------'
from math import sqrt
import os
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange


def top_k(logits, thres = 0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner


class BasicVAE(nn.Module):

    def get_codebook_indices(self, images):
        raise NotImplementedError()

    def decode(self, img_seq):
        raise NotImplementedError()

    def get_codebook_probs(self, img_seq):
        raise NotImplementedError()

    def get_image_tokens_size(self):
        pass

    def get_image_size(self):
        pass


class ResBlock(nn.Module):
    def __init__(self, chan_in, hidden_size, chan_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(chan_in, hidden_size, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_size, hidden_size, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_size, chan_out, 1)
        )

    def forward(self, x):
        return self.net(x) + x


class DiscreteVAE(BasicVAE):
    def __init__(
        self,
        image_size = 256,
        num_tokens = 512,
        codebook_dim = 512,
        num_layers = 3,
        hidden_dim = 64,
        channels = 3,
        smooth_l1_loss = False,
        temperature = 0.9,
        straight_through = False,
        kl_div_loss_weight = 0.
    ):
        super().__init__()
        # assert log2(image_size).is_integer(), 'image size must be a power of 2'
        assert num_layers >= 1, 'number of layers must be greater than or equal to 1'

        self.image_size = image_size
        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.temperature = temperature
        self.straight_through = straight_through
        self.codebook = nn.Embedding(num_tokens, codebook_dim)

        enc_layers = []
        dec_layers = []

        enc_in = channels
        dec_in = codebook_dim

        for layer_id in range(num_layers):
            enc_layers.append(nn.Sequential(nn.Conv2d(enc_in, hidden_dim, 4, stride=2, padding=1), nn.ReLU()))
            enc_layers.append(ResBlock(chan_in=hidden_dim, hidden_size=hidden_dim, chan_out=hidden_dim))
            enc_in = hidden_dim
            dec_layers.append(nn.Sequential(nn.ConvTranspose2d(dec_in, hidden_dim, 4, stride=2, padding=1), nn.ReLU()))
            dec_layers.append(ResBlock(chan_in=hidden_dim, hidden_size=hidden_dim, chan_out=hidden_dim))
            dec_in = hidden_dim

        enc_layers.append(nn.Conv2d(hidden_dim, num_tokens, 1))
        dec_layers.append(nn.Conv2d(hidden_dim, channels, 1))

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder = nn.Sequential(*dec_layers)

        self.loss_fn = F.smooth_l1_loss if smooth_l1_loss else F.mse_loss
        self.kl_div_loss_weight = kl_div_loss_weight

    def get_image_size(self):
        return self.image_size

    def get_image_tokens_size(self):
        return self.image_size // 8

    @torch.no_grad()
    @eval_decorator
    def get_codebook_indices(self, images):
        logits = self.forward(images, return_logits = True)
        codebook_indices = logits.argmax(dim = 1)
        return codebook_indices

    @torch.no_grad()
    @eval_decorator
    def get_codebook_probs(self, images):
        logits = self.forward(images, return_logits = True)
        return nn.Softmax(dim=1)(logits)

    def decode(
        self,
        img_seq
    ):
        image_embeds = self.codebook(img_seq)
        b, n, d = image_embeds.shape
        h = w = int(sqrt(n))

        image_embeds = rearrange(image_embeds, 'b (h w) d -> b d h w', h = h, w = w)
        images = self.decoder(image_embeds)
        return images

    def forward(
        self,
        img,
        return_loss = False,
        return_recons = False,
        return_logits = False,
        temp = None
    ):
        device, num_tokens, image_size, kl_div_loss_weight = img.device, self.num_tokens, self.image_size, self.kl_div_loss_weight
        assert img.shape[-1] == image_size and img.shape[-2] == image_size, f'input must have the correct image size {image_size}'

        logits = self.encoder(img)

        if return_logits:
            return logits # return logits for getting hard image indices for DALL-E training

        temp = default(temp, self.temperature)
        soft_one_hot = F.gumbel_softmax(logits, tau = temp, dim = 1, hard = self.straight_through)
        sampled = einsum('b n h w, n d -> b d h w', soft_one_hot, self.codebook.weight)
        out = self.decoder(sampled)

        if not return_loss:
            return out

        # reconstruction loss

        recon_loss = self.loss_fn(img, out)

        # kl divergence

        logits = rearrange(logits, 'b n h w -> b (h w) n')
        qy = F.softmax(logits, dim = -1)

        log_qy = torch.log(qy + 1e-10)
        log_uniform = torch.log(torch.tensor([1. / num_tokens], device = device))
        kl_div = F.kl_div(log_uniform, log_qy, None, None, 'batchmean', log_target = True)

        loss = recon_loss + (kl_div * kl_div_loss_weight)

        if not return_recons:
            return loss

        return loss, out


from dall_e import load_model


class Dalle_VAE(BasicVAE):
    def __init__(self, image_size):
        super().__init__()
        self.encoder = None
        self.decoder = None
        self.image_size = image_size

    def load_model(self, model_dir, device):
        self.encoder = load_model(os.path.join(model_dir, "encoder.pkl"), device)
        self.decoder = load_model(os.path.join(model_dir, "decoder.pkl"), device)

    def decode(self, img_seq):
        bsz = img_seq.size()[0]
        img_seq = img_seq.view(bsz, self.image_size // 8, self.image_size // 8)
        z = F.one_hot(img_seq, num_classes=self.encoder.vocab_size).permute(0, 3, 1, 2).float()
        return self.decoder(z).float()

    def get_codebook_indices(self, images):
        z_logits = self.encoder(images)
        return torch.argmax(z_logits, axis=1)

    def get_codebook_probs(self, images):
        z_logits = self.encoder(images)
        return nn.Softmax(dim=1)(z_logits)

    def forward(self, img_seq_prob, no_process=False):
        if no_process:
            return self.decoder(img_seq_prob.float()).float()
        else:
            bsz, seq_len, num_class = img_seq_prob.size()
            z = img_seq_prob.view(bsz, self.image_size // 8, self.image_size // 8, self.encoder.vocab_size)
            return self.decoder(z.permute(0, 3, 1, 2).float()).float()
