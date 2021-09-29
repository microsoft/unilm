import re
import sys
import os
import random
import torch
import pickle
import logging
import numpy as np

# from transformers import (WEIGHTS_NAME,
#   BertConfig, BertForSequenceClassification, BertTokenizer,
#   RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
#   RobertaModel, BertModel, XLMModel,
#   XLMConfig, XLMForSequenceClassification, XLMTokenizer,
#   XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer,
#   DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer,
#   BertForQuestionAnswering)
#
# from src.pequod.model.roberta import RobertaForQuestionAnswering
from transformers import XLMRobertaConfig, XLMRobertaForRetrieval, XLMRobertaTokenizer

# ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) \
#   for conf in (BertConfig, XLNetConfig, XLMConfig,
#     RobertaConfig, DistilBertConfig)), ())

ALL_MODELS = []

# # Model classes for classification
# MODEL_CLASSES = {
#   'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
#   'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
#   'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
#   'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
#   'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
#   "xlmr": (RobertaConfig, RobertaForSequenceClassification, XLMRTokenizer)
# }
#
# QA_MODELS = {
#   "bert": BertForQuestionAnswering,
#   "roberta": RobertaForQuestionAnswering,
#   "xlmr": RobertaForQuestionAnswering,
# }

BERT_CLASSES = {
  "xlmr": (XLMRobertaConfig, XLMRobertaForRetrieval, XLMRobertaTokenizer),
}


def to_cuda(tup):
  return tuple(t.cuda() for t in tup)


def set_seed(args):
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  #TODO multi gpu support
  # if args.n_gpu > 0:
  #   torch.cuda.manual_seed_all(args.seed)


def init_exp(args):
  # dump parameters
  set_dump_path(args)
  pickle.dump(args, open(os.path.join(args.dump_path, 'params.pkl'), 'wb'))

  # get running command
  command = ["python", sys.argv[0]]
  for x in sys.argv[1:]:
    if x.startswith('--'):
      assert '"' not in x and "'" not in x
      command.append(x)
    else:
      assert "'" not in x
      if re.match('^[a-zA-Z0-9_]+$', x):
        command.append("%s" % x)
      else:
        command.append("'%s'" % x)
  command = ' '.join(command)
  args.command = command + ' --exp_id "%s"' % args.exp_id

  # check experiment name
  assert len(args.exp_name.strip()) > 0

  logging.basicConfig(
    format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt = '%m/%d/%Y %H:%M:%S',
    level = logging.INFO)
  logger = logging.getLogger(__name__)
  logger.info("\n".join(
    "%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
  logger.info("The experiment will be stored in %s\n" % args.dump_path)
  logger.info("Running command: %s" % command)
  logger.info("")


def set_dump_path(args, output_dir=None, exp_name=None):
  if output_dir is None: output_dir = args.output_dir
  if exp_name is None: exp_name = args.exp_name
  chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
  while True:
    exp_id = ''.join(random.choice(chars) for _ in range(10))
    if not os.path.isdir(os.path.join(output_dir, exp_name, exp_id)):
      break
  args.exp_id = exp_id
  dump_path = os.path.join(output_dir, exp_name, exp_id)
  os.makedirs(dump_path)
  args.dump_path = dump_path
