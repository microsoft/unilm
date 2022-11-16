# WebSRC
The dataset will be **manually downloaded**. After downloading, please modify the argument ```--web_train_file```, ```--web_eval_file```, ```web_root_dir```, and ```root_dir``` in args.py.

## Installation

```bash
pip install -r requirements.txt
```

## Train

```bash
CUDA_VISIBLE_DEVICES=0 python run_docvqa.py --do_train True --do_eval True  --model_name_or_path /path/to/xdoc-docvqa
```

## Test
```bash
CUDA_VISIBLE_DEVICES=0 python run_docvqa.py --do_train False --do_eval False --model_name_or_path /path/to/xdoc-docvqa
```
