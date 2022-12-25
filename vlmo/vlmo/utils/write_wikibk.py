import json
import pandas as pd
import pyarrow as pa
import gc
import random
import os

from tqdm import tqdm
from glob import glob


def path2rest(line):
    return [
        "None",
        [line],
        "wikibk",
        "train",
    ]


def make_arrow(root, dataset_root):
    for index in range(0, 50):
        file_path = f"{root}/wikibk.{index}.txt"

        all_sents = []
        with open(file_path, "r", encoding="utf-8") as fp:
            for line in fp:
                all_sents.append(line.strip())

        print(file_path)
        print("Number of sentences: {}".format(len(all_sents)))

        bs = [path2rest(line) for line in tqdm(all_sents)]
        dataframe = pd.DataFrame(bs, columns=["image", "caption", "source", "split"],)

        table = pa.Table.from_pandas(dataframe)

        os.makedirs(dataset_root, exist_ok=True)
        with pa.OSFile(f"{dataset_root}/wikibk_train_{index}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)
        del dataframe
        del table
        del bs
        gc.collect()