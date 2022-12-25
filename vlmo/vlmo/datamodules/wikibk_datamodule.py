from vlmo.datasets import WikibkDataset
from .datamodule_base import BaseDataModule


class WikibkDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return WikibkDataset

    @property
    def dataset_name(self):
        return "wikibk"
