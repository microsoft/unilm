from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import glob

import numpy as np
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    get_linear_schedule_with_warmup,
)

from markuplmft.models.markuplm import MarkupLMConfig, MarkupLMTokenizer, MarkupLMForTokenClassification

from utils import get_swde_features, SwdeDataset
from eval_utils import page_level_constraint
import constants
import torch

import copy

logger = logging.getLogger(__name__)


def set_seed(args):
    r"""
    Fix the random seed for reproduction.
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def train(args, train_dataset, model, tokenizer, sub_output_dir):
    r"""
    Train the model
    """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()
    else:
        tb_writer = None

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate,
                      eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(args.warmup_ratio * t_total),
                                                num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for epoch in train_iterator:
        if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
            train_dataloader.sampler.set_epoch(epoch)
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'xpath_tags_seq': batch[3],
                      'xpath_subs_seq': batch[4],
                      'labels': batch[5],
                      }

            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:
                        raise ValueError("Shouldn't `evaluate_during_training` when ft SWDE!!")
                        # results = evaluate(args, model, tokenizer, prefix=str(global_step))
                        # for key, value in results.items():
                        #    tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(sub_output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break
        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def eval_on_one_website(args, model, website, sub_output_dir, prefix=""):
    dataset, info = get_dataset_and_info_for_websites([website], evaluate=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    # In our setting, we should not apply DDP
    eval_sampler = SequentialSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    eval_dataloader = DataLoader(dataset,
                                 sampler=eval_sampler,
                                 batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_logits = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'xpath_tags_seq': batch[3],
                      'xpath_subs_seq': batch[4],
                      }
            outputs = model(**inputs)
            logits = outputs["logits"]  # which is (bs,seq_len,node_type)
            all_logits.append(logits.detach().cpu())

    all_probs = torch.softmax(torch.cat(all_logits, dim=0), dim=2)  # (all_samples, seq_len, node_type)

    assert len(all_probs) == len(info)

    all_res = {}

    for sub_prob, sub_info in zip(all_probs, info):
        html_path, involved_first_tokens_pos, \
        involved_first_tokens_xpaths, involved_first_tokens_types, \
                involved_first_tokens_text   = sub_info

        if html_path not in all_res:
            all_res[html_path] = {}

        for pos, xpath, type,text in zip(involved_first_tokens_pos, involved_first_tokens_xpaths,
                                    involved_first_tokens_types, involved_first_tokens_text):

            pred = sub_prob[pos]  # (node_type_size)
            if xpath not in all_res[html_path]:
                all_res[html_path][xpath] = {}
                all_res[html_path][xpath]["pred"] = pred
                all_res[html_path][xpath]["truth"] = type
                all_res[html_path][xpath]["text"] = text
            else:
                all_res[html_path][xpath]["pred"] += pred
                assert all_res[html_path][xpath]["truth"] == type
                assert all_res[html_path][xpath]["text"] == text

    # we have build all_res
    # then write predictions

    lines = []

    for html_path in all_res:
        for xpath in all_res[html_path]:
            final_probs = all_res[html_path][xpath]["pred"] / torch.sum(all_res[html_path][xpath]["pred"])
            pred_id = torch.argmax(final_probs).item()
            pred_type = constants.ATTRIBUTES_PLUS_NONE[args.vertical][pred_id]
            final_probs = final_probs.numpy().tolist()

            s = "\t".join([
                html_path,
                xpath,
                all_res[html_path][xpath]["text"],
                all_res[html_path][xpath]["truth"],
                pred_type,
                ",".join([str(score) for score in final_probs]),
            ])

            lines.append(s)

    res = page_level_constraint(args.vertical, website, lines, sub_output_dir)

    return res  # (precision, recall, f1)


def evaluate(args, model, test_websites, sub_output_dir, prefix=""):
    r"""
    Evaluate the model
    """

    all_eval_res = {}
    all_precision = []
    all_recall = []
    all_f1 = []

    for website in test_websites:
        res_on_one_website = eval_on_one_website(args, model, website, sub_output_dir, prefix)
        all_precision.append(res_on_one_website[0])
        all_recall.append(res_on_one_website[1])
        all_f1.append(res_on_one_website[2])

    return {"precision": sum(all_precision) / len(all_precision),
            "recall": sum(all_recall) / len(all_recall),
            "f1": sum(all_f1) / len(all_f1),
            }


def load_and_cache_one_website(args, tokenizer, website):
    cached_features_file = os.path.join(
        args.root_dir,
        "cached",
        args.vertical,
        website,
        f"cached_markuplm_{str(args.max_seq_length)}_pages{args.n_pages}_prevnodes{args.prev_nodes_into_account}"
    )

    if not os.path.exists(os.path.dirname(cached_features_file)):
        os.makedirs(os.path.dirname(cached_features_file))

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)

    else:
        logger.info(
            f"Creating features for {args.vertical}-{website}-pages{args.n_pages}_prevnodes{args.prev_nodes_into_account}")

        features = get_swde_features(root_dir=args.root_dir,
                                     vertical=args.vertical,
                                     website=website,
                                     tokenizer=tokenizer,
                                     doc_stride=args.doc_stride,
                                     max_length=args.max_seq_length,
                                     prev_nodes=args.prev_nodes_into_account,
                                     n_pages=args.n_pages)

        if args.local_rank in [-1, 0] and args.save_features:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    return features


def load_and_cache_examples(args, tokenizer, websites):
    r"""
    Load and process the raw data.
    """
    # if args.local_rank not in [-1, 0]:
    #    torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset,
    # and the others will use the cache

    feature_dicts = {}

    for website in websites:
        features_per_website = load_and_cache_one_website(args, tokenizer, website)
        feature_dicts[website] = features_per_website

    return feature_dicts


def get_dataset_and_info_for_websites(websites, evaluate=False):
    """

    Args:
        websites: a list of websites

    Returns:
        a dataset object
    """

    all_features = []

    for website in websites:
        features_per_website = global_feature_dicts[website]
        all_features += features_per_website

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in all_features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in all_features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in all_features], dtype=torch.long)
    all_xpath_tags_seq = torch.tensor([f.xpath_tags_seq for f in all_features], dtype=torch.long)
    all_xpath_subs_seq = torch.tensor([f.xpath_subs_seq for f in all_features], dtype=torch.long)

    if not evaluate:
        all_labels = torch.tensor([f.labels for f in all_features], dtype=torch.long)
        dataset = SwdeDataset(all_input_ids=all_input_ids,
                              all_attention_mask=all_attention_mask,
                              all_token_type_ids=all_token_type_ids,
                              all_xpath_tags_seq=all_xpath_tags_seq,
                              all_xpath_subs_seq=all_xpath_subs_seq,
                              all_labels=all_labels)
        info = None
    else:
        # in evaluation, we do not add labels
        dataset = SwdeDataset(all_input_ids=all_input_ids,
                              all_attention_mask=all_attention_mask,
                              all_token_type_ids=all_token_type_ids,
                              all_xpath_tags_seq=all_xpath_tags_seq,
                              all_xpath_subs_seq=all_xpath_subs_seq)
        info = [(f.html_path,
                 f.involved_first_tokens_pos,
                 f.involved_first_tokens_xpaths,
                 f.involved_first_tokens_types,
                 f.involved_first_tokens_text) for f in all_features]

    return dataset, info


def do_something(train_websites, test_websites, args, config, tokenizer):
    # before each run, we reset the seed
    set_seed(args)
    
    model = MarkupLMForTokenClassification.from_pretrained(args.model_name_or_path, config=config)
    model.resize_token_embeddings(len(tokenizer))

    sub_output_dir = os.path.join(args.output_dir,
                                  args.vertical,
                                  f"seed-{args.n_seed}_pages-{args.n_pages}",
                                  "-".join(train_websites))

    # if args.local_rank == 0:
    #     torch.distributed.barrier()
    # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is
    # set. Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running
    # `--fp16_opt_level="O2"` will remove the need for this code, but it is still valid.
    if args.fp16:
        try:
            import apex
            apex.amp.register_half_function(torch, 'einsum')
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    # Training
    if args.do_train:
        train_dataset, _ = get_dataset_and_info_for_websites(train_websites)
        tokenizer.save_pretrained(sub_output_dir)
        model.to(args.device)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer, sub_output_dir)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Save the trained model and the tokenizer
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(sub_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(sub_output_dir)

        logger.info("Saving model checkpoint to %s", sub_output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(sub_output_dir)
        tokenizer.save_pretrained(sub_output_dir)
        torch.save(args, os.path.join(sub_output_dir, 'training_args.bin'))

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [sub_output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(sub_output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs

        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        config = MarkupLMConfig.from_pretrained(sub_output_dir)
        tokenizer = MarkupLMTokenizer.from_pretrained(sub_output_dir)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            try:
                int(global_step)
            except ValueError:
                global_step = ""
            if global_step and int(global_step) < args.eval_from_checkpoint:
                continue
            if global_step and args.eval_to_checkpoint is not None and int(global_step) >= args.eval_to_checkpoint:
                continue
            model = MarkupLMForTokenClassification.from_pretrained(checkpoint, config=config)
            model.to(args.device)

            # Evaluate
            result = evaluate(args, model, test_websites, sub_output_dir, prefix=global_step)

            result = dict((k + ('_{}'.format(global_step) if global_step else ''), v) for k, v in result.items())
            results.update(result)

    logger.info("Results: {}".format(results))

    return results


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--root_dir", default=None, type=str, required=True,
                        help="the root directory of the pre-processed SWDE dataset, "
                             "in which we have `book-abebooks-2000.pickle` files like that")
    parser.add_argument("--vertical", default="book", type=str,
                        help="Which vertical to train and test"
                             "Now we haven't supported multi-verticals in one program")
    parser.add_argument("--n_seed", default=2, type=int,
                        help="number of seed pages")
    parser.add_argument("--n_pages", default=2000, type=int,
                        help="number of pages in each website, set a small number for debugging")
    parser.add_argument("--prev_nodes_into_account", default=4, type=int,
                        help="how many previous nodes before a variable nodes will we use"
                             "large value means more context")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")

    # Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default=None, type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending "
                             "with step number")
    parser.add_argument('--eval_from_checkpoint', type=int, default=0,
                        help="Only evaluate the checkpoints with prefix larger than or equal to it, beside the final "
                             "checkpoint with no prefix")
    parser.add_argument('--eval_to_checkpoint', type=int, default=None,
                        help="Only evaluate the checkpoints with prefix smaller than it, beside the final checkpoint "
                             "with no prefix")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_ratio", default=0.0, type=float,
                        help="Linear warmup ratio over all steps")

    parser.add_argument('--logging_steps', type=int, default=10,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=3000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--save_features', type=bool, default=True,
                        help="whether or not to save the processed features, default is True")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count() if not args.no_cuda else 0
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
        # Make sure only the first process in distributed training will download model & vocab

    config = MarkupLMConfig.from_pretrained(args.model_name_or_path)
    config_dict = config.to_dict()
    config_dict.update({"node_type_size": len(constants.ATTRIBUTES_PLUS_NONE[args.vertical])})
    config = MarkupLMConfig.from_dict(config_dict)

    tokenizer = MarkupLMTokenizer.from_pretrained(args.model_name_or_path)

    # first we load the features

    feature_dicts = load_and_cache_examples(args=args,
                                            tokenizer=tokenizer,
                                            websites=constants.VERTICAL_WEBSITES[args.vertical],
                                            )

    global global_feature_dicts
    global_feature_dicts = feature_dicts

    all_precision = []
    all_recall = []
    all_f1 = []

    for i in range(10):
        wid_start = i
        wid_end = i + args.n_seed

        train_websites = []
        test_websites = []

        for wid in range(wid_start, wid_end):
            wwid = wid % 10
            train_websites.append(constants.VERTICAL_WEBSITES[args.vertical][wwid])

        for website in constants.VERTICAL_WEBSITES[args.vertical]:
            if website not in train_websites:
                test_websites.append(website)

        ori_config = copy.deepcopy(config)
        ori_tokenizer = copy.deepcopy(tokenizer)

        eval_res = do_something(train_websites, test_websites, args, config, tokenizer)
        all_precision.append(eval_res["precision"])
        all_recall.append(eval_res["recall"])
        all_f1.append(eval_res["f1"])

        config = ori_config
        tokenizer = ori_tokenizer

    p = sum(all_precision) / len(all_precision)
    r = sum(all_recall) / len(all_recall)
    f = sum(all_f1) / len(all_f1)

    logger.info("=================FINAL RESULTS=================")
    logger.info(f"Precision : {p}")
    logger.info(f"Recall : {r}")
    logger.info(f"F1 : {f}")

    res_file = os.path.join(args.output_dir, f"{args.vertical}-all-10-runs-score.txt")

    with open(res_file, "w") as fio:
        fio.write(f"Precision : {p}\nRecall : {r}\nF1 : {f}\n")


if __name__ == "__main__":
    main()
