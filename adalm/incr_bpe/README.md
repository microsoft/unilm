# Incremental BPE builder

<strong>Modified, simplified version of text_encoder_build_subword.py and its dependencies included in [tensor2tensor library](https://github.com/tensorflow/tensor2tensor), making its output fits to [google research's open-sourced BERT project](https://github.com/google-research/bert).</strong>

## Requirement
The environment I made this project in consists of :
- python 3.6
- tensorflow 1.11 

## Basic usage

#### Build domain-specific vocabulary automatically

If you want to build a proper size of vocabulary for specific domain using the incremental algorithm in our paper, you can do as following:

```bash
python vocab_extend.py \
	--corpus {file for the domain corpus} \
	--raw_vocab {bert_raw_vocab_file} \
	--output_file {he output file of the final vocabulary} \
	--interval {vocab size interval} \
	--threshold {threshold for P(D)}

# Example using sample data
python vocab_extend.py --corpus test_data/chem.txt \
	--raw_vocab test_data/vocab.txt \
	--output_file test_data/chem.vocab \
	--interval 1000 --threshold 1

```

If you simply want to get a specific size of vocab, you can run the following 

```
python subword_builder.py \
--corpus_filepattern {corpus_for_vocab} \
--raw_vocab {bert_raw_vocab_file} \
--output_filename {name_of_vocab} \
--vocab_size {final vocab size} \
--do_lower_case
```
