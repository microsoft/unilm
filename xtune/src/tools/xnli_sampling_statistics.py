import argparse
import json
import random
from transformers import XLMRobertaTokenizer

from transformers import xglue_processors as processors

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--nbest_size", type=int, default=-1, help="nbest size in sampling subword sequence"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.2, help="alpha"
    )
    parser.add_argument(
        "--train_language", default=None, type=str, help="Train language if is different of the evaluation language."
    )
    parser.add_argument(
        "--language",
        default=None,
        type=str,
        required=True,
        help="Evaluation language. Also train language if `train_language` is set to None.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model",
    )
    parser.add_argument(
        "--sample_rounds",
        default=1,
        type=int,
        required=True,
        help="Path to pre-trained model",
    )

    args = parser.parse_args()

    tokenizer = XLMRobertaTokenizer.from_pretrained(
        args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=None,
    )

    task = "xnli"
    processor = processors[task](language=args.train_language, train_language=args.train_language)
    examples = processor.get_train_examples(args.data_dir)

    train_word_cnt_origin = {}
    for example in examples:
        tokens_a = tokenizer.tokenize(example.text_a, add_special_tokens=True)
        tokens_b = tokenizer.tokenize(example.text_b, add_special_tokens=True)
        for token in tokens_a + tokens_b:
            if token not in train_word_cnt_origin:
                train_word_cnt_origin[token] = 0
            train_word_cnt_origin[token] += 1

    all_examples = []
    for i in range(args.sample_rounds):
        all_examples += examples
    examples = all_examples

    sent_len = []
    example_len = []

    train_word_cnt = {}

    for example in examples:
        tokens_a = tokenizer.tokenize(example.text_a, add_special_tokens=True, nbest_size=args.nbest_size,
                                      alpha=args.alpha)
        tokens_b = tokenizer.tokenize(example.text_b, add_special_tokens=True, nbest_size=args.nbest_size,
                                      alpha=args.alpha)
        for token in tokens_a + tokens_b:
            if token not in train_word_cnt:
                train_word_cnt[token] = 0
            train_word_cnt[token] += 1

        sent_len += [len(tokens_a), len(tokens_b)]
        example_len += [len(tokens_a) + len(tokens_b)]

    print(sum(sent_len) / len(sent_len))
    print(sum(example_len) / len(example_len))

    total = 0
    n_oov = 0
    for token in train_word_cnt:
        if token not in train_word_cnt_origin:
            n_oov += train_word_cnt[token]
            # n_oov += 1
        total += train_word_cnt[token]
        # total += 1

    print("{} oov rate: {}".format("extra", n_oov / total))

    total = 0
    n_oov = 0
    for token in train_word_cnt_origin:
        if token not in train_word_cnt:
            n_oov += train_word_cnt_origin[token]
            # n_oov += 1
        total += train_word_cnt_origin[token]
        # total += 1

    print("{} oov rate: {}".format("origin", n_oov / total))

    # exit(0)
    eval_datasets = []
    eval_langs = args.language.split(',')
    for split in ["valid"]:
        for lang in eval_langs:
            eval_datasets.append((split, lang))

    for split, lang in eval_datasets:
        processor = processors[task](language=lang, train_language=lang)
        examples = processor.get_valid_examples(args.data_dir)

        sent_len = []
        example_len = []
        valid_word_cnt = {}
        for example in examples:
            tokens_a = tokenizer.tokenize(example.text_a, add_special_tokens=True)
            tokens_b = tokenizer.tokenize(example.text_b, add_special_tokens=True)
            for token in tokens_a + tokens_b:
                if token not in valid_word_cnt:
                    valid_word_cnt[token] = 0
                valid_word_cnt[token] += 1

                sent_len += [len(tokens_a), len(tokens_b)]
                example_len += [len(tokens_a) + len(tokens_b)]

        print(sum(sent_len) / len(sent_len))
        print(sum(example_len) / len(example_len))

        total = 0
        n_oov = 0
        for token in valid_word_cnt:
            if token not in train_word_cnt:
                n_oov += valid_word_cnt[token]
                # n_oov += 1
            total += valid_word_cnt[token]
            # total += 1

        print("{} oov rate: {}".format(lang, n_oov / total))
