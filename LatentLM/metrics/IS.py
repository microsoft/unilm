"""Utils for Inception Score calculation.
Borrowed from:
    PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
    The MIT License (MIT)
    See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details
"""
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .fid import get_inception_model, create_dataset_from_files


def inception_softmax(inception_model, images):
    with torch.no_grad():
        logits = inception_model.get_logits(images)
        ps = torch.nn.functional.softmax(logits, dim=1)
    return ps


@torch.no_grad()
def calculate_kl_div(ps, splits: int):
    scores = []
    num_samples = ps.shape[0]
    for j in range(splits):
        part = ps[(j * num_samples // splits):((j + 1) * num_samples // splits), :]
        kl = part * (torch.log(part) - torch.log(torch.unsqueeze(torch.mean(part, 0), 0)))
        kl = torch.mean(torch.sum(kl, 1))
        kl = torch.exp(kl)
        scores.append(kl.unsqueeze(0))
    scores = torch.cat(scores, 0)
    m_scores = torch.mean(scores).detach().cpu().numpy()
    m_std = torch.std(scores).detach().cpu().numpy()
    return m_scores, m_std


@torch.no_grad()
def compute_inception_score_from_dataset(dataset,
                                         splits,
                                         batch_size,
                                         device=torch.device('cuda'),
                                         inception_model=None,
                                         disable_tqdm=False):
    """
    Args:
        - dataset: dataset returning **float (0~1)** images
    """
    if inception_model is None:
        inception_model = get_inception_model().to(device)

    data_loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=16)

    inception_model.eval()
    probs_list = []

    for imgs in tqdm(data_loader, disable=disable_tqdm):
        imgs = imgs[0].to(device)
        logits = inception_model.get_logits(imgs)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        probs_list.append(probs)

    probs_list = torch.cat(probs_list, 0)
    m_scores, m_std = calculate_kl_div(probs_list, splits=splits)

    return m_scores, m_std


def compute_inception_score_from_files(path,
                                       splits=10,
                                       batch_size=500,
                                       device=torch.device('cuda'),
                                       inception_model=None,
                                       disable_tqdm=False):

    dataset = create_dataset_from_files(path)
    return compute_inception_score_from_dataset(dataset,
                                                splits,
                                                batch_size,
                                                device,
                                                inception_model,
                                                disable_tqdm)


def compute_inception_score_from_tensor(tensor,
                                        splits=10,
                                        batch_size=500,
                                        device=torch.device('cuda'),
                                        inception_model=None,
                                        disable_tqdm=False):

    dataset = torch.utils.data.TensorDataset(tensor)
    return compute_inception_score_from_dataset(dataset,
                                                splits,
                                                batch_size,
                                                device,
                                                inception_model,
                                                disable_tqdm)
