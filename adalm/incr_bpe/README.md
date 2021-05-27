# Incremental BPE builder

<strong>Modified, simplified version of text_encoder_build_subword.py and its dependencies included in [tensor2tensor library](https://github.com/tensorflow/tensor2tensor), making its output fits to [google research's open-sourced BERT project](https://github.com/google-research/bert).</strong>

## Requirement
The environment I made this project in consists of :
- python 3.6
- tensorflow 1.11 

## Basic usage
```
python subword_builder.py \
--corpus_filepattern {corpus_for_vocab} \
--raw_vocab {bert_raw_vocab_file} \
--output_filename {name_of_vocab} \
--min_count {minimum_subtoken_counts} \
--do_lower_case
```
The parameter *min_count* controls the vocabulary size of incremental BPE. 

TODO: set a *max_vocab_size* for incremental BPE. 