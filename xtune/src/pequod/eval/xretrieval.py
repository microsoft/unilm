import faiss
import json
import logging
import numpy as np
import os
import torch


from src.pequod.data.xretrieval import load_and_cache_examples
from src.pequod.eval.evaluator import Evaluator


logger = logging.getLogger(__name__)


def similarity_search(x, y, dim, normalize=False, dist='L2'):
  top_k = 10
  num = x.shape[0]
  if dist == 'cosine':
    idx = faiss.IndexFlatIP(dim)
  else:
    idx = faiss.IndexFlatL2(dim)
  if normalize:
    faiss.normalize_L2(x)
    faiss.normalize_L2(y)
  idx.add(x)
  scores, prediction = idx.search(y, top_k)
  return prediction, scores


class TatoebaEvaluator(Evaluator):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.model_langs = ["share_lang", "order"]
    self.proj_matrix_fast = kwargs.get("proj_matrix_fast", None)
    if self.proj_matrix_fast is not None:
      logger.info("proj_matrix_fast:" + str(self.proj_matrix_fast.size()))
      self.proj_matrix_fast = self.proj_matrix_fast[0].float().cuda()
    self.res = {}

  def get_mean_emb(self, layer_outputs, pool_mask):
    embs = (layer_outputs * pool_mask.unsqueeze(2).float()).sum(dim=1) / \
      pool_mask.sum(dim=1).view(-1, 1).float()
    return embs
  
  def get_cxlm_emb(self, layer_outputs):
    if self.proj_matrix_fast is None:
      raise ValueError
    ret = torch.mm(layer_outputs[:,0,:], self.proj_matrix_fast)
    # ret = layer_outputs[:,0,:]
    return ret
  
  def get_cls_emb(self, layer_outputs):
    return layer_outputs[:,0,:]
  
  def bt_norm(self, x):
    m = x.mean(0, keepdim=True)
    v = x.var(0, unbiased=True, keepdim=True)
    return (x-m) / torch.sqrt(v+1e-5)

  def get_embeddings(self, batch, outputs, emb_type=None, is_bt_norm=False):
    if emb_type is None:
      emb_type = self.args.emb_type
    last_layer_outputs, first_token_outputs, all_layer_outputs = outputs

    if emb_type == "mean":
      ret = self.get_mean_emb(all_layer_outputs[self.args.mean_layer_id], batch["attention_mask"])
    elif emb_type == "cls":
      ret = self.get_cls_emb(all_layer_outputs[-1])
    elif emb_type == "cxlm":
      ret = self.get_cxlm_emb(all_layer_outputs[self.args.mean_layer_id]) #TODO
    else: raise ValueError
    
    if is_bt_norm:
      ret = self.bt_norm(ret)
    ret = ret.cpu().numpy().astype(np.float32)
    # ret = None
    del last_layer_outputs, first_token_outputs, all_layer_outputs
    torch.cuda.empty_cache()
    return ret
  
  def run(self):
    args = self.args
    self.model.eval()

    if args.data_prefix == "tatoeba":
      langs = ["ara", "bul", "deu", "ell", "spa", "fra", "hin", "rus", "swh", "tha", "tur", "urd", "vie", "cmn"]
      langpairs = ["%s-eng" % lang for lang in langs]
    elif args.data_prefix == "cxlm":
      langpairs = "ar-en bg-en de-en el-en en-es en-fr en-hi en-ru en-sw en-th en-tr en-ur en-vi en-zh".split()
    elif args.data_prefix == "debug":
      langpairs = ["ar-en" ]
    elif args.data_prefix == "tat15plus":
      args.data_prefix = "tatoeba"
      l15 = set(["ara", "bul", "deu", "ell", "spa", "fra", "hin", "rus", "swh", "tha", "tur", "urd", "vie", "cmn"])
      ld = {'ara':'ar', 'heb':'he', 'vie':'vi', 'ind':'id',
        'jav':'jv', 'tgl':'tl', 'eus':'eu', 'mal':'ml',
        'tel':'te', 'afr':'af', 'nld':'nl', 'deu':'de',
        'ell':'el', 'ben':'bn', 'hin':'hi', 'mar':'mr', 'urd':'ur',
        'tam':'ta', 'fra':'fr', 'ita':'it', 'por':'pt', 'spa':'es',
        'bul':'bg', 'rus':'ru', 'jpn':'ja', 'kat':'ka', 'kor':'ko',
        'tha':'th', 'swh':'sw', 'cmn':'zh', 'kaz':'kk', 'tur':'tr',
        'est':'et', 'fin':'fi', 'hun':'hu', 'pes':'fa'}
      langs_str = 'ar he vi id jv tl eu ml ta te af nl de el bn hi mr ur fa fr it pt es bg ru ja ka ko th sw zh kk tr et fi hu'
      #langs_str = 'hi mr ur fa fr it pt es bg ru ja ka ko th sw zh kk tr et fi hu'
      #langs_str = 'ar he'
      #langs_str = 'ara heb'
      langs = langs_str.split(' ')
      #for l in ld:
      #  if l in l15: continue
      #  langs.append(l)
      # langs = ["afr", "jpn", "kor", "kaz", "est", "fin", "hun", "pes"]
      langpairs = ["%s-en" % lang for lang in langs]
    else: raise ValueError

    for langpair in langpairs:
      lang1, lang2 = langpair.split("-")
      logger.info("Eval langpair: %s" % langpair)
      dl1 = self.get_dataloader(langpair, lang1)
      dl2 = self.get_dataloader(langpair, lang2)

      all_emb1 = []
      all_emb2 = []
      for batch1, batch2 in zip(dl1, dl2):
        batch1 = self._parse_batch(batch1, has_label=False)
        batch2 = self._parse_batch(batch2, has_label=False)
        #forward
        with torch.no_grad():
          outputs1 = self.model(**batch1)
          all_emb1.append(self.get_embeddings(batch1, outputs1, is_bt_norm=args.bt_norm))
          outputs2 = self.model(**batch2)
          all_emb2.append(self.get_embeddings(batch2, outputs2, is_bt_norm=args.bt_norm))
      
      all_emb1 = np.concatenate(all_emb1)
      all_emb2 = np.concatenate(all_emb2)
      emb_sz = all_emb1.shape[-1]
      if args.reverse_eval:
        all_emb1, all_emb2 = all_emb2, all_emb1
      predictions, scores = similarity_search(
        all_emb1, all_emb2, emb_sz, normalize=self.args.normalize, dist=self.args.dist)
      correct = tot = 0
      
      
      # output retrieval results 
      with open(os.path.join(args.output_dir, 'test-{0}.tsv'.format(lang1)), 'w', encoding='utf-8') as writer:
        for i, pred in enumerate(predictions):
          writer.write(str(pred[0]) + '\n')

      with open(os.path.join(args.output_dir, 'test-{0}-scores.tsv'.format(lang1)), 'w', encoding='utf-8') as writer:
        for  pred, score in zip(predictions, scores):
          writer.write(' '.join([str(p) for p in pred]) + '\t' + ' '.join([str(s) for s in score]) + '\n')

      for i, pred in enumerate(predictions):
        if i == pred[0]: correct += 1
        tot += 1
      logger.info("langpair:%s acc:%.2f" % (langpair, 100*correct/tot))
      self.res[langpair] = 100*correct/tot

    #output_fn = os.path.join(args.exp_results_dir, args.exp_name)
    #if args.reverse_eval: output_fn += "-rev"
    #with open(output_fn, "w") as fp:
    #  json.dump(self.res, fp)
      

  def load_and_cache_examples(self, langpair, lang, **kwargs):
    args = self.args
    cache_key = "%s-%s" % (args.model_key, args.model_type)
    return load_and_cache_examples(
      args=args,
      langpair=langpair,
      lang=lang,
      tokenizer=self.tokenizer,
      key=cache_key,
      prefix=args.data_prefix,
    )
