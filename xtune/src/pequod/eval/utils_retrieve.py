# coding=utf-8
# This repository is modified based on the LASER repository.
# https://github.com/facebookresearch/LASER
# Copyright The LASER Team Authors, and The XTREME Benchmark Authors.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""Utility functions for retrieval tasks."""


import os
import sys
import faiss
import tempfile
import numpy as np


def knn(x, y, k, use_gpu, dist='cosine'):
  return knnGPU(x, y, k) if use_gpu else knnCPU(x, y, k, dist)


def knnGPU(x, y, k, mem=5*1024*1024*1024):
  dim = x.shape[1]
  batch_size = mem // (dim*4)
  sim = np.zeros((x.shape[0], k), dtype=np.float32)
  ind = np.zeros((x.shape[0], k), dtype=np.int64)
  for xfrom in range(0, x.shape[0], batch_size):
    xto = min(xfrom + batch_size, x.shape[0])
    bsims, binds = [], []
    for yfrom in range(0, y.shape[0], batch_size):
      yto = min(yfrom + batch_size, y.shape[0])
      print('{}-{}  ->  {}-{}'.format(xfrom, xto, yfrom, yto))
      idx = faiss.IndexFlatIP(dim)
      idx = faiss.index_cpu_to_all_gpus(idx)
      idx.add(y[yfrom:yto])
      bsim, bind = idx.search(x[xfrom:xto], min(k, yto-yfrom))
      bsims.append(bsim)
      binds.append(bind + yfrom)
      del idx
    bsims = np.concatenate(bsims, axis=1)
    binds = np.concatenate(binds, axis=1)
    aux = np.argsort(-bsims, axis=1)
    for i in range(xfrom, xto):
      for j in range(k):
        sim[i, j] = bsims[i-xfrom, aux[i-xfrom, j]]
        ind[i, j] = binds[i-xfrom, aux[i-xfrom, j]]
  return sim, ind


def knnCPU(x, y, k, dist='cosine'):
  # x: query, y: database
  dim = x.shape[1]
  if dist == 'cosine':
    idx = faiss.IndexFlatIP(dim)
  else:
    idx = faiss.IndexFlatL2(dim)
  idx.add(y)
  sim, ind = idx.search(x, k)

  if dist != 'cosine':
    sim = 1 / (1 + sim)
  return sim, ind


def score(x, y, fwd_mean, bwd_mean, margin, dist='cosine'):
  if dist == 'cosine':
    return margin(x.dot(y), (fwd_mean + bwd_mean) / 2)
  else:
    l2 = ((x - y) ** 2).sum()
    sim = 1 / (1 + l2)
    return margin(sim, (fwd_mean + bwd_mean) / 2)


def score_candidates(x, y, candidate_inds, fwd_mean, bwd_mean, margin, dist='cosine'):
  print(' - scoring {:d} candidates using {}'.format(x.shape[0], dist))
  scores = np.zeros(candidate_inds.shape)
  for i in range(scores.shape[0]):
    for j in range(scores.shape[1]):
      k = candidate_inds[i, j]
      scores[i, j] = score(x[i], y[k], fwd_mean[i], bwd_mean[k], margin, dist)
  return scores


def text_load_unify(fname, encoding, unify=True):
  print(' - loading texts {:s}: '.format(fname), end='')
  fin = open(fname, encoding=encoding, errors='surrogateescape')
  inds = []
  sents = []
  sent2ind = {}
  n = 0
  nu = 0
  for line in fin:
    new_ind = len(sent2ind)
    inds.append(sent2ind.setdefault(line, new_ind))
    if unify:
      if inds[-1] == new_ind:
        sents.append(line[:-1])
        nu += 1
    else:
      sents.append(line[:-1])
      nu += 1
    n += 1
  print('{:d} lines, {:d} unique'.format(n, nu))
  del sent2ind
  return inds, sents


def unique_embeddings(emb, ind):
  aux = {j: i for i, j in enumerate(ind)}
  print(' - unify embeddings: {:d} -> {:d}'.format(len(emb), len(aux)))
  return emb[[aux[i] for i in range(len(aux))]]


def shift_embeddings(x, y):
  print(' - shift embeddings')
  delta = x.mean(axis=0) - y.mean(axis=0)
  x2y = x - delta
  y2x = y + delta
  return x2y, y2x


def mine_bitext(x, y, src_text_file, trg_text_file, output_file, mode='mine',
                retrieval='max', margin='ratio', threshold=0,
                neighborhood=4, use_gpu=False, encoding='utf-8', dist='cosine', use_shift_embeds=False):
  src_inds, src_sents = text_load_unify(src_text_file, encoding, True)
  trg_inds, trg_sents = text_load_unify(trg_text_file, encoding, True)

  x = unique_embeddings(x, src_inds)
  y = unique_embeddings(y, trg_inds)
  if dist == 'cosine':
    faiss.normalize_L2(x)
    faiss.normalize_L2(y)

  if use_shift_embeds:
    x2y, y2x = shift_embeddings(x, y)

  # calculate knn in both directions
  if retrieval is not 'bwd':
    print(' - perform {:d}-nn source against target, dist={}'.format(neighborhood, dist))
    if use_shift_embeds:
      # project x to y space, and search k-nn ys for each x
      x2y_sim, x2y_ind = knn(x2y, y, min(y.shape[0], neighborhood), use_gpu, dist)
      x2y_mean = x2y_sim.mean(axis=1)
    else:
      x2y_sim, x2y_ind = knn(x, y, min(y.shape[0], neighborhood), use_gpu, dist)
      x2y_mean = x2y_sim.mean(axis=1)

  if retrieval is not 'fwd':
    print(' - perform {:d}-nn target against source, dist={}'.format(neighborhood, dist))
    if use_shift_embeds:
      y2x_sim, y2x_ind = knn(y2x, x, min(x.shape[0], neighborhood), use_gpu, dist)
      y2x_mean = y2x_sim.mean(axis=1)
    else:
      y2x_sim, y2x_ind = knn(y, x, min(x.shape[0], neighborhood), use_gpu, dist)
      y2x_mean = y2x_sim.mean(axis=1)

  # margin function
  if margin == 'absolute':
    margin = lambda a, b: a
  elif margin == 'distance':
    margin = lambda a, b: a - b
  else:  # margin == 'ratio':
    margin = lambda a, b: a / b

  fout = open(output_file, mode='w', encoding=encoding, errors='surrogateescape')

  if mode == 'search':
    print(' - Searching for closest sentences in target')
    print(' - writing alignments to {:s}'.format(output_file))
    scores = score_candidates(x, y, x2y_ind, x2y_mean, y2x_mean, margin)
    best = x2y_ind[np.arange(x.shape[0]), scores.argmax(axis=1)]

    nbex = x.shape[0]
    ref = np.linspace(0, nbex-1, nbex).astype(int)  # [0, nbex)
    err = nbex - np.equal(best.reshape(nbex), ref).astype(int).sum()
    print(' - errors: {:d}={:.2f}%'.format(err, 100*err/nbex))
    for i in src_inds:
      print(trg_sents[best[i]], file=fout)

  elif mode == 'score':
    for i, j in zip(src_inds, trg_inds):
      s = score(x[i], y[j], x2y_mean[i], y2x_mean[j], margin)
      print(s, src_sents[i], trg_sents[j], sep='\t', file=fout)

  elif mode == 'mine':
    print(' - mining for parallel data')
    if use_shift_embeds:
      fwd_scores = score_candidates(x2y, y, x2y_ind, x2y_mean, y2x_mean, margin)
      bwd_scores = score_candidates(y2x, x, y2x_ind, y2x_mean, x2y_mean, margin)
    else:
      fwd_scores = score_candidates(x, y, x2y_ind, x2y_mean, y2x_mean, margin)
      bwd_scores = score_candidates(y, x, y2x_ind, y2x_mean, x2y_mean, margin)
    fwd_best = x2y_ind[np.arange(x.shape[0]), fwd_scores.argmax(axis=1)]
    bwd_best = y2x_ind[np.arange(y.shape[0]), bwd_scores.argmax(axis=1)]
    print(' - writing alignments to {:s}'.format(output_file))
    if threshold > 0:
      print(' - with threshold of {:f}'.format(threshold))
    if retrieval == 'fwd':
      for i, j in enumerate(fwd_best):
        print(fwd_scores[i].max(), src_sents[i], trg_sents[j], sep='\t', file=fout)
    if retrieval == 'bwd':
      for j, i in enumerate(bwd_best):
        print(bwd_scores[j].max(), src_sents[i], trg_sents[j], sep='\t', file=fout)
    if retrieval == 'intersect':
      for i, j in enumerate(fwd_best):
        if bwd_best[j] == i:
          print(fwd_scores[i].max(), src_sents[i], trg_sents[j], sep='\t', file=fout)
    if retrieval == 'max':
      indices = np.stack((np.concatenate((np.arange(x.shape[0]), bwd_best)),
                          np.concatenate((fwd_best, np.arange(y.shape[0])))), axis=1)
      scores = np.concatenate((fwd_scores.max(axis=1), bwd_scores.max(axis=1)))
      seen_src, seen_trg = set(), set()
      for i in np.argsort(-scores):
        src_ind, trg_ind = indices[i]
        if not src_ind in seen_src and not trg_ind in seen_trg:
          seen_src.add(src_ind)
          seen_trg.add(trg_ind)
          if scores[i] > threshold:
            print(scores[i], src_sents[src_ind], trg_sents[trg_ind], sep='\t', file=fout)
  fout.close()


def bucc_optimize(candidate2score, gold):
  items = sorted(candidate2score.items(), key=lambda x: -x[1])
  ngold = len(gold)
  nextract = ncorrect = 0
  threshold = 0
  best_f1 = 0
  for i in range(len(items)):
    nextract += 1
    if '\t'.join(items[i][0]) in gold:
      ncorrect += 1
    if ncorrect > 0:
      precision = ncorrect / nextract
      recall = ncorrect / ngold
      f1 = 2 * precision * recall / (precision + recall)
      if f1 > best_f1:
        best_f1 = f1
        threshold = (items[i][1] + items[i + 1][1]) / 2
  return threshold


def bucc_extract(cand2score, th, fname):
  if fname:
    of = open(fname, 'w', encoding='utf-8')
  bitexts = []
  for (src, trg), score in cand2score.items():
    if score >= th:
      bitexts.append(src + '\t' + trg)
      if fname:
        of.write(src + '\t' + trg + '\n')
  if fname:
    of.close()
  return bitexts


def read_sent2id(text_file, id_file, encoding='utf-8'):
  repeated = set()
  sent2id = {}
  with open(id_file, encoding=encoding, errors='surrogateescape') as f:
    ids = [l.strip() for l in f]
  with open(text_file, encoding=encoding, errors='surrogateescape') as f:
    sentences = [l.strip() for l in f]
  for id, sent in zip(ids, sentences):
    if sent in sent2id:
      repeated.add(sent)
    else:
      sent2id[sent] = id
  for sent in repeated:
    del sent2id[sent]
  return sent2id


def read_candidate2score(candidates_file, src_text_file, trg_text_file, src_id_file, trg_id_file, encoding='utf-8'):
  print(' - reading sentences {}'.format(candidates_file))
  src_sent2id = read_sent2id(src_text_file, src_id_file, encoding)
  trg_sent2id = read_sent2id(trg_text_file, trg_id_file, encoding)

  print(' - reading candidates {}'.format(candidates_file))
  candidate2score = {}
  with open(candidates_file, encoding=encoding, errors='surrogateescape') as f:
    for line in f:
      score, src, trg = line.split('\t')
      score = float(score)
      src = src.strip()
      trg = trg.strip()
      if src in src_sent2id and trg in trg_sent2id:
        src_id = src_sent2id[src]
        trg_id = trg_sent2id[trg]
        score = max(score, candidate2score.get((src_id, trg_id), score))
        candidate2score[(src_id, trg_id)] = score
  return candidate2score


def bucc_eval(candidates_file, gold_file, src_file, trg_file, src_id_file, trg_id_file, predict_file, threshold=None, encoding='utf-8'):
  candidate2score = read_candidate2score(candidates_file, src_file, trg_file, src_id_file, trg_id_file, encoding)

  if threshold is not None and gold_file is None:
    print(' - using threshold {}'.format(threshold))
  else:
    print(' - optimizing threshold on gold alignments {}'.format(gold_file))
    gold = {line.strip() for line in open(gold_file)}
    threshold = bucc_optimize(candidate2score, gold)

  bitexts = bucc_extract(candidate2score, threshold, predict_file)
  if gold_file is not None:
    ncorrect = len(gold.intersection(bitexts))
    if ncorrect > 0:
      precision = ncorrect / len(bitexts)
      recall = ncorrect / len(gold)
      f1 = 2*precision*recall / (precision + recall)
    else:
      precision = recall = f1 = 0

    print(' - best threshold={:f}: precision={:.2f}, recall={:.2f}, F1={:.2f}'
          .format(threshold, 100*precision, 100*recall, 100*f1))
    return {'best-threshold': threshold, 'precision': 100*precision, 'recall': 100*recall, 'F1': 100*f1}
  else:
    return None


def similarity_search(x, y, dim, normalize=False):
  num = x.shape[0]
  idx = faiss.IndexFlatL2(dim)
  if normalize:
    faiss.normalize_L2(x)
    faiss.normalize_L2(y)
  idx.add(x)
  scores, prediction = idx.search(y, 1)
  return prediction