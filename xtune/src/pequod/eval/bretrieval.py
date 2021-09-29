import faiss
import json
import logging
import numpy as np
import os
import torch


from src.pequod.data.xretrieval import load_and_cache_examples
from src.pequod.eval.evaluator import Evaluator
from src.pequod.eval.utils_retrieve import mine_bitext, bucc_eval


logger = logging.getLogger(__name__)


def load_embeddings(embed_file, num_sentences=None):
  logger.info(' loading from {}'.format(embed_file))
  embeds = np.load(embed_file)
  return embeds


class BuccEvaluator(Evaluator):

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

    best_threshold = None
    SL, TL = args.src_language, args.tgt_language
    for split in ['test']:
    # for split in ['dev', 'test']:
      prefix = f'{SL}-{TL}.{split}'
      if args.extract_embeds:
        for lang in [SL, TL]:
            file = os.path.join(args.output_dir,  f'{prefix}.{lang}.npy') 
            if os.path.exists(file):
              continue
            langpair =  f'{SL}-{TL}.{split}'
            dl1 = self.get_dataloader(langpair, lang)
            all_emb1 = []
            for batch1 in dl1:
                batch1 = self._parse_batch(batch1, has_label=False)
                #forward
                with torch.no_grad():
                    outputs1 = self.model(**batch1)
                    all_emb1.append(self.get_embeddings(batch1, outputs1, is_bt_norm=args.bt_norm))
            
            all_emb1 = np.concatenate(all_emb1)
            file = os.path.join(args.output_dir,  f'{prefix}.{lang}.npy') 
            logger.info('save embed {} to file {}'.format(all_emb1.shape, file))
            np.save(file, all_emb1)

      if args.mine_bitext:
        threshold = None

        cand2score_file = os.path.join(args.output_dir, 'candidates.tsv')
        
        x = load_embeddings(os.path.join(args.output_dir,  f'{prefix}.{SL}.npy'))
        y = load_embeddings(os.path.join(args.output_dir,  f'{prefix}.{TL}.npy'))

        x_text_file = os.path.join(args.data_dir,  f'{prefix}.{SL}.txt')
        y_text_file = os.path.join(args.data_dir,  f'{prefix}.{TL}.txt')
        x_id_file = os.path.join(args.data_dir,  f'{prefix}.{SL}.id')
        y_id_file = os.path.join(args.data_dir,  f'{prefix}.{TL}.id')

        mine_bitext(x, y, x_text_file, y_text_file, cand2score_file, dist=args.dist, use_shift_embeds=args.use_shift_embeds)
        gold_file = os.path.join(args.data_dir,   f'{prefix}.gold')
        if os.path.exists(gold_file):
            predict_file = os.path.join(args.output_dir, f'test-{SL}.tsv')
            results = bucc_eval(cand2score_file, gold_file, x_text_file, y_text_file, x_id_file, y_id_file, predict_file, threshold)
            
            with open(os.path.join(args.output_dir,  'final.txt'), 'w', encoding='utf-8') as f:
              f.write(json.dumps(results))

            best_threshold = results['best-threshold']
            logger.info('--Candidates: {}'.format(cand2score_file))
            logger.info(' '.join('{}={:.4f}'.format(k,v) for k,v in results.items()))
      # if args.layer_ensemble:
      #   threshold = None
      #   prefix = 'mean_l2'
      #   layers = args.ens_layers.split(',')
      #
      #   cand2score_file = os.path.join(args.output_dir, 'candidates.tsv')
      #
      #   x = load_embeddings(os.path.join(args.output_dir,  f'{prefix}.{SL}.npy'))
      #   y = load_embeddings(os.path.join(args.output_dir,  f'{prefix}.{TL}.npy'))
      #
      #   x_text_file = os.path.join(args.data_dir,  f'{prefix}.{SL}.txt')
      #   y_text_file = os.path.join(args.data_dir,  f'{prefix}.{TL}.txt')
      #   x_id_file = os.path.join(args.data_dir,  f'{prefix}.{SL}.id')
      #   y_id_file = os.path.join(args.data_dir,  f'{prefix}.{TL}.id')
      #
      #   mine_bitext(x, y, x_text_file, y_text_file, cand2score_file, dist=args.dist, use_shift_embeds=args.use_shift_embeds)
      #   gold_file = os.path.join(args.data_dir,   f'{prefix}.gold')
      #   if os.path.exists(gold_file):
      #       predict_file = os.path.join(args.output_dir, f'test-{SL}.tsv')
      #       results = bucc_eval(cand2score_file, gold_file, x_text_file, y_text_file, x_id_file, y_id_file, predict_file, threshold)
      #
      #       with open(os.path.join(args.output_dir,  'final.txt'), 'w', encoding='utf-8') as f:
      #         f.write(json.dumps(results))
      #
      #       best_threshold = results['best-threshold']
      #       logger.info('--Candidates: {}'.format(cand2score_file))
      #       logger.info(' '.join('{}={:.4f}'.format(k,v) for k,v in results.items()))
    # output retrieval results 
    #   with open(os.path.join(args.output_dir, 'test-{0}.tsv'.format(lang1)), 'w', encoding='utf-8') as writer:
    #     for i, pred in enumerate(predictions):
    #       writer.write(str(pred[0]) + '\n')
      

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
