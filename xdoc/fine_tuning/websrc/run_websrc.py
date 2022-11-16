import os
import sys
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
import shutil
import logging
import torch.distributed as dist


from transformers import (
    BertTokenizer,
    RobertaTokenizer
)

from args import args
from model import (  
    Layoutlmv1ForQuestionAnswering,
    Layoutlmv1Config,
    Layoutlmv1Config_roberta,
    Layoutlmv1ForQuestionAnswering_roberta
)
from util import set_seed, set_exp_folder, check_screen
from trainer import train, evaluate # choose a specific train function
# from data.datasets.docvqa import DocvqaDataset
from websrc import get_websrc_dataset

def main(args):

    set_seed(args)
    set_exp_folder(args)

    # Set up logger
    logging.basicConfig(filename="{}/output/{}/log.txt".format(args.output_dir, args.exp_name), level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info('Args '+str(args))

    # Get config, model, and tokenizer

    if args.model_type == 'bert':
        config_class, model_class, tokenizer_class = Layoutlmv1Config, Layoutlmv1ForQuestionAnswering, BertTokenizer
    elif args.model_type == 'roberta':
        config_class, model_class, tokenizer_class = Layoutlmv1Config_roberta, Layoutlmv1ForQuestionAnswering_roberta, RobertaTokenizer

    config = config_class.from_pretrained(
                args.model_name_or_path, cache_dir=args.cache_dir
            )
    config.add_linear = args.add_linear

    tokenizer = tokenizer_class.from_pretrained(
                args.model_name_or_path, cache_dir=args.cache_dir
            )


    model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
        )

    parameters = sum(p.numel() for p in model.parameters())
    print("Total params: %.2fM" % (parameters/1e6))


    ## Start training
    if args.do_train:
   
        dataset_web = get_websrc_dataset(args, tokenizer)
      
        logging.info(f'Web dataset is successfully loaded. Length : {len(dataset_web)}')
        train(args, dataset_web, model, tokenizer)

    # ## Start evaluating
    # if args.do_eval:

    logging.info('Start evaluating')
    dataset_web, examples, features = get_websrc_dataset(args, tokenizer, evaluate=True, output_examples=True)
    logging.info(f'[Eval] Web dataset is successfully loaded. Length : {len(dataset_web)}')
    evaluate(args, dataset_web, examples, features, model, tokenizer)


    ## Start testing
    if args.do_test:
        pass


if __name__ == '__main__':
    main(args)

   
