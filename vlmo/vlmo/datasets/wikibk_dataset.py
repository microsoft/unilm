from glob import glob
from .base_dataset import BaseDataset


class WikibkDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        if split == "test":
            split = "val"

        if split == "train":
            names = [f"wikibk_train_{i}" for i in range(50)]
        elif split == "val":
            names = ["wikibk_val_0"]

        super().__init__(*args, **kwargs, names=names, text_column_name="caption")

    def __getitem__(self, index):
        return self.get_text_suite(index)
