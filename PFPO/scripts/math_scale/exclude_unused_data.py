import json
import argparse
import os.path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig_data_file", type=str)
    parser.add_argument("--orig_id_filed", type=str, default="id")
    parser.add_argument("--train_data_file", type=str)
    parser.add_argument("--train_id_field", type=str, default="idx")
    args = parser.parse_args()

    pass
