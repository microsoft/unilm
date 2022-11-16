import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", default='your_exp_name', type=str)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--output_dir", default='.', type=str) 
parser.add_argument("--overwrite_output_dir", default=True)
parser.add_argument("--model_name_or_path", type=str, default='/path/to/xdoc-pretrain-roberta-1M')
parser.add_argument("--cache_dir", type=str, default='./cache')
parser.add_argument("--scratch", action='store_true')
parser.add_argument("--model_type", default='roberta', choices=['bert', 'roberta'])

parser.add_argument("--do_train", default=False, type=bool)
parser.add_argument("--do_eval", default=True, type=bool)
parser.add_argument("--do_test", default=False, type=bool)

parser.add_argument("--overwrite_cache", default=False, type=bool)
parser.add_argument("--train_file", default='train.json', type=str)
parser.add_argument("--dev_file", default='val.json', type=str)
parser.add_argument("--test_file", default='test.json', type=str)
parser.add_argument("--doc_stride", default=128, type=int)
parser.add_argument("--threads", default=1, type=int)
parser.add_argument("--max_query_length", default=64, type=int)
parser.add_argument("--local_rank", default=-1, type=int)
parser.add_argument("--pad_img_input", default=False, type=bool)
parser.add_argument("--fix_visual", default=False, type=bool)
parser.add_argument("--num_workers", default=8, type=int)

# dataset (args to load websrc)
parser.add_argument("--web_train_file", default='/path/to/WebSRC/websrc1.0_train_.json', type=str)
parser.add_argument("--web_eval_file", default='/path/to/WebSRC/websrc1.0_dev_.json', type=str)
parser.add_argument("--web_root_dir", default='/path/to/WebSRC', type=str)
parser.add_argument("--root_dir", default='/path/to/WebSRC', type=str)

parser.add_argument("--n_best_size", default=20, type=int)
parser.add_argument("--max_answer_length", default=30, type=int)
parser.add_argument("--do_lower_case", default=True, type=bool)

parser.add_argument("--web_num_features", default=0, type=int)
parser.add_argument("--web_save_features", default=True, type=bool)
parser.add_argument("--verbose_logging", default=True, type=bool)
parser.add_argument("--embedding_mode", choices=['html','box','html+box'], default='html', type=str)
parser.add_argument("--dataloader_shuffle", default=True, type=bool)

# train
parser.add_argument("--batch_per_gpu", default=16, type=int)
parser.add_argument("--epoch", default=5, type=int)
parser.add_argument("--warmup_ratio", default=0.1, type=float)
parser.add_argument("--weight_decay", default=0, type=float)
parser.add_argument("--fp16", default=False)
parser.add_argument("--fp16_opt_level", default='O1', type=str)
parser.add_argument("--learning_rate", default=5e-5, type=float)
parser.add_argument("--adam_epsilon", default=1e-8, type=float) 
parser.add_argument("--max_grad_norm", default=1, type=float) 

parser.add_argument("--log_step", default=50, type=int) 
parser.add_argument("--save_step", default=10000, type=int)

parser.add_argument("--add_linear", default=True, type=bool) 
parser.add_argument("--accumulation", default=1, type=int)

# mlm
parser.add_argument("--mlm_probability", default=0.15, type=float)

args = parser.parse_args()