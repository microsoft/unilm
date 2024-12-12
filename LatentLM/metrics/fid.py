"""Adapted from https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py"""
import glob
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg
from torch.utils.data import DataLoader
from tqdm import tqdm

from .inception import InceptionV3

import pickle


class InceptionWrapper(InceptionV3):

    def forward(self, inp):
        pred = super().forward(inp)[0]
        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))
        pred = pred.reshape(pred.shape[0], -1)

        return pred

    def get_logits(self, inp):
        _, logits = super().forward(inp, return_logits=True)

        return logits


def get_inception_model(dims=2048):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionWrapper([block_idx])
    return model


def mean_covar_torch(xs):
    mu = torch.mean(xs, dim=0, keepdim=True)
    ys = xs - mu
    unnormalized_sigma = (ys.T @ ys)
    sigma = unnormalized_sigma / (xs.shape[0] - 1)
    return mu, sigma


def mean_covar_numpy(xs):
    if isinstance(xs, torch.Tensor):
        xs = xs.cpu().numpy()
    return np.mean(xs, axis=0), np.cov(xs, rowvar=False)


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        logging.warning(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


@torch.no_grad()
def compute_statistics_dataset(dataset,
                               batch_size=64,
                               inception_model=None,
                               stage1_model=None,
                               device=torch.device('cuda'),
                               skip_original=False,
                               ):

    if skip_original and stage1_model is None:
        return None, None, None, None

    if inception_model is None:
        inception_model = get_inception_model().to(device)

    loader = DataLoader(dataset, shuffle=False, pin_memory=True, batch_size=batch_size, num_workers=16)

    inception_model.eval()
    if stage1_model:
        stage1_model.eval()

    acts = []
    acts_recon = []

    sample_size_sum = 0.0
    sample_sum = torch.tensor(0.0, device=device)
    sample_sq_sum = torch.tensor(0.0, device=device)
    sample_max = torch.tensor(float('-inf'), device=device)
    sample_min = torch.tensor(float('inf'), device=device)

    for xs, _ in tqdm(loader, desc="compute acts"):
        xs = xs.to(device, non_blocking=True)

        # we are assuming that dataset returns value in -1 ~ 1 -> remap to 0 ~ 1
        xs = torch.clamp(xs*0.5 + 0.5, 0, 1)

        sample_sum += xs.sum()
        sample_sq_sum += xs.pow(2.0).sum()
        sample_size_sum += xs.numel()
        sample_max = max(xs.max(), sample_max)
        sample_min = min(xs.min(), sample_min)

        act = inception_model(xs).cpu() if not skip_original else None
        acts.append(act)

        if stage1_model:
            # here we assume that stage1 model input & output values are in -1 ~ 1 range
            # this may not cover DiscreteVAE
            imgs = 2. * xs - 1.
            xs_recon = torch.cat([
                stage1_model(imgs[i:i+1])[0] for i in range(imgs.shape[0])
            ], dim=0)
            xs_recon = torch.clamp(xs_recon * 0.5 + 0.5, 0, 1)
            act_recon = inception_model(xs_recon).cpu()
            acts_recon.append(act_recon)

    sample_mean = sample_sum.item() / sample_size_sum
    sample_std = ((sample_sq_sum.item() / sample_size_sum) - (sample_mean ** 2.0)) ** 0.5
    logging.info(f'val imgs. stats :: '
                 f'max: {sample_max:.4f}, min: {sample_min:.4f}, mean: {sample_mean:.4f}, std: {sample_std:.4f}')

    acts = torch.cat(acts, dim=0) if not skip_original else None

    if skip_original:
        mu_acts, sigma_acts = None, None
    else:
        mu_acts, sigma_acts = mean_covar_numpy(acts)

    if stage1_model:
        acts_recon = torch.cat(acts_recon, dim=0)
        mu_acts_recon, sigma_acts_recon = mean_covar_numpy(acts_recon)
    else:
        mu_acts_recon, sigma_acts_recon = None, None

    return mu_acts, sigma_acts, mu_acts_recon, sigma_acts_recon


def create_dataset_from_files(path, verbose=False):
    samples = []
    pkl_lists = glob.glob(os.path.join(path, 'samples*.pkl'))
    first_file_name = os.path.basename(pkl_lists[0])
    last_file_name = os.path.basename(pkl_lists[-1])
    logging.info(f'loading generated images from {path}: [{first_file_name}, ..., {last_file_name}]')

    for pkl in tqdm(pkl_lists, desc='loading pickles'):
        with open(pkl, 'rb') as f:
            # samples.append(pickle.load(f).cpu().numpy())
            s = pickle.load(f)
            if isinstance(s, np.ndarray):
                s = torch.from_numpy(s)
            samples.append(s)

    datasets = [torch.utils.data.TensorDataset(sample) for sample in samples]
    dataset = torch.utils.data.ConcatDataset(datasets)

    if verbose:
        total_size = sum([sample.size for sample in samples])
        sample_mean = sum([sample.sum() for sample in samples]) / total_size
        sample_std = (sum([((sample - sample_mean)**2).sum() for sample in samples]) / total_size) ** 0.5
        sample_max = max([sample.max() for sample in samples])
        sample_min = min([sample.min() for sample in samples])
        logging.info(f'gen. imgs. stats :: '
                     f'max: {sample_max:.4f}, min: {sample_min:.4f}, mean: {sample_mean:.4f}, std: {sample_std:.4f}')

    return dataset


@torch.no_grad()
def compute_activations_from_dataset(dataset,
                                     batch_size=64,
                                     inception_model=None,
                                     device=torch.device('cuda'),
                                     normalized=False,
                                     ):
    if inception_model is None:
        inception_model = get_inception_model().to(device)

    loader = DataLoader(dataset, shuffle=False, pin_memory=True, batch_size=batch_size, num_workers=16)

    acts = []
    inception_model.eval()

    for xs in tqdm(loader, desc="compute acts (gen. imgs)"):
        xs = xs[0].to(device, non_blocking=True)
        if normalized:
            xs = 0.5 * xs + 0.5
        act = inception_model(xs)
        acts.append(act.cpu())

    acts = torch.cat(acts, dim=0)
    return acts


def compute_statistics_from_files(path,
                                  batch_size=64,
                                  inception_model=None,
                                  device=torch.device('cuda'),
                                  return_acts=False,
                                  ):
    dataset = create_dataset_from_files(path)
    acts = compute_activations_from_dataset(dataset,
                                            batch_size=batch_size,
                                            inception_model=inception_model,
                                            device=device)
    mu_acts, sigma_acts = mean_covar_numpy(acts)
    if return_acts:
        return mu_acts, sigma_acts, acts
    else:
        return mu_acts, sigma_acts


def compute_statistics_from_tensor(tensor,
                                   batch_size=64,
                                   inception_model=None,
                                   device=torch.device('cuda'),
                                   return_acts=False,
                                   ):
    dataset = torch.utils.data.TensorDataset(tensor)
    acts = compute_activations_from_dataset(dataset,
                                            batch_size=batch_size,
                                            inception_model=inception_model,
                                            device=device)
    mu_acts, sigma_acts = mean_covar_numpy(acts)
    if return_acts:
        return mu_acts, sigma_acts, acts
    else:
        return mu_acts, sigma_acts


def compute_rfid(dataset,
                 stage1_model,
                 batch_size=64,
                 device=torch.device('cuda'),
                 ):
    mu_orig, sigma_orig, mu_recon, sigma_recon = \
        compute_statistics_dataset(dataset,
                                   stage1_model=stage1_model,
                                   batch_size=batch_size,
                                   device=device,
                                   skip_original=False,
                                   )
    rfid = frechet_distance(mu_orig, sigma_orig, mu_recon, sigma_recon)
    return rfid


def compute_fid(fake_path,
                ref_stat_path,
                batch_size=64,
                device=torch.device('cuda'),
                ):
    act_path = Path(fake_path) / 'acts.npz'
    if not act_path.exists():
        mu, sigma, acts = compute_statistics_from_files(fake_path,
                                                        batch_size=batch_size,
                                                        device=device,
                                                        return_acts=True,
                                                        )
        np.savez(act_path, acts=acts, mu=mu, sigma=sigma)
        logging.info(f'activations saved to {act_path.as_posix()}')
    else:
        logging.info(f'precomputed activations found: {act_path.as_posix()}')

    acts_fake = np.load(act_path)

    stats_ref = np.load(ref_stat_path)
    mu_ref, sigma_ref = stats_ref['mu'], stats_ref['sigma']
    logging.info(f'reference batch stats loaded from {ref_stat_path}')

    mu_fake, sigma_fake = acts_fake['mu'], acts_fake['sigma']

    logging.info('computing fid...')
    fid = frechet_distance(mu_ref, sigma_ref, mu_fake, sigma_fake)
    logging.info('FID: {fid:.4f}'.format(fid=fid))

    return fid


def compute_fid_without_store(tensor, ref_stat_path, batch_size=64, device=torch.device('cuda')):
    print('Compute mu and sigma for fake images...')
    mu_fake, sigma_fake = compute_statistics_from_tensor(tensor, batch_size=batch_size, device=device)

    stats_ref = np.load(ref_stat_path)
    mu_ref, sigma_ref = stats_ref['mu'], stats_ref['sigma']
    print(f'reference batch stats loaded from {ref_stat_path}')

    print('computing fid...')
    fid = frechet_distance(mu_ref, sigma_ref, mu_fake, sigma_fake)
    print('FID: {fid:.4f}'.format(fid=fid))

    return fid
