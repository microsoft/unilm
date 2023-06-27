import gzip
import itertools
from random import Random
import os
import shutil
import tempfile
from typing import Iterator
import unittest
import gc

from infinibatch.datasets import chunked_dataset_iterator


class TestBase(unittest.TestCase):
    def setUp(self):
        self.test_data = [
            ["item number one", "item number two", "item number three", "item number four"],
            ["item number five"],
            [
                "item number six",
                "item number seven",
                "item number eight",
                "item number nine",
                "item number ten",
                "item number eleven",
            ],
            ["item number twelve", "item number thirteen", "item number fourteen",],
        ]

        self.flattened_test_data = []
        for chunk in self.test_data:
            for item in chunk:
                self.flattened_test_data.append(item)

        self.data_dir = tempfile.mkdtemp()
        self.chunk_file_paths = []
        for chunk_id, chunk in enumerate(self.test_data):
            file_name = os.path.join(self.data_dir, "chunk_" + str(chunk_id).zfill(10) + ".gz")
            self.chunk_file_paths.append(file_name)
            file_content = "\n".join(chunk)
            with gzip.open(file_name, "wt", encoding="utf-8") as f:
                f.write(file_content)

    @staticmethod
    def read_chunk(textfile_path: str) -> Iterator[str]:  # read_chunk_fn for chunked_dataset_iterator
        with gzip.open(textfile_path, "rt", encoding="utf-8") as f:
            return iter(f.read().splitlines())

    def tearDown(self):
        gc.collect()  # this will get the pre-fetch terminated in some tests, which otherwise may still want to read these files
        shutil.rmtree(self.data_dir)

    def assertMultisetEqual(self, a, b):
        self.assertEqual(len(a), len(b))
        self.assertSetEqual(set(a), set(b))


class Test_chunked_dataset_iterator(TestBase):
    def test_no_shuffle(self):
        items = list(
            itertools.islice(
                chunked_dataset_iterator(self.chunk_file_paths, self.read_chunk, shuffle=False, buffer_size=1000),
                len(self.flattened_test_data),
            )
        )
        self.assertListEqual(items, self.flattened_test_data)

    def test_other_files_present(self):
        with open(os.path.join(self.data_dir, "i_do_not_belong_here.txt"), "w") as f:
            f.write("really ...")
        items = list(
            itertools.islice(
                chunked_dataset_iterator(self.chunk_file_paths, self.read_chunk, shuffle=False, buffer_size=1000),
                len(self.flattened_test_data),
            )
        )
        self.assertListEqual(items, self.flattened_test_data)

    def test_transform(self):
        transform = lambda s: s + "!"
        modified_test_data = [transform(s) for s in self.flattened_test_data]
        items = list(
            itertools.islice(
                chunked_dataset_iterator(
                    self.chunk_file_paths, self.read_chunk, shuffle=False, buffer_size=1000, transform=transform
                ),
                len(self.flattened_test_data),
            )
        )
        self.assertListEqual(items, modified_test_data)

    def test_two_instances(self):
        dataset0 = chunked_dataset_iterator(
            self.chunk_file_paths, self.read_chunk, shuffle=False, buffer_size=1000, num_instances=2, instance_rank=0
        )
        dataset1 = chunked_dataset_iterator(
            self.chunk_file_paths, self.read_chunk, shuffle=False, buffer_size=1000, num_instances=2, instance_rank=1
        )
        items0 = list(itertools.islice(dataset0, len(self.test_data[0]) + len(self.test_data[2])))
        items1 = list(itertools.islice(dataset1, len(self.test_data[1]) + len(self.test_data[3])))
        self.assertMultisetEqual(set(items0 + items1), self.flattened_test_data)

    def test_checkpointing(self):
        random = Random(1)
        for use_windowed in (True, False):
            for i in range(2):
                first_length = random.randrange(11, 21)
                extra_length = random.randrange(11, 21)
                dataset = chunked_dataset_iterator(
                    self.chunk_file_paths,
                    self.read_chunk,
                    shuffle=(i % 2 == 0),
                    buffer_size=1000,
                    seed=i,
                    num_instances=2,
                    instance_rank=0,
                    use_windowed=use_windowed,
                )
                for _ in range(first_length):
                    next(dataset)
                checkpoint = dataset.getstate()
                items1 = list(itertools.islice(dataset, extra_length))
                dataset.setstate(checkpoint)
                items2 = list(itertools.islice(dataset, extra_length))
                self.assertListEqual(items1, items2)


if __name__ == "__main__":
    unittest.main()
