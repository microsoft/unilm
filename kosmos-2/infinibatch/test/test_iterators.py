import copy
import itertools
import multiprocessing
from random import Random
import unittest

import torch

from infinibatch.iterators import *

if __name__ == "__main__":
    unittest.main()


class TestBase(unittest.TestCase):
    def setUp(self):
        self.lengths = [1, 2, 3, 42, 57]
        self.world_sizes = [1, 2, 3, 4, 5, 11, 16, 64, 73]
        self.seed = 42

    def assertMultisetEqual(self, a, b):
        def list_to_dict(l):
            d = {}
            for item in l:
                d[item] = d.get(item, 0) + 1
            return d

        self.assertEqual(list_to_dict(a), list_to_dict(b))


class TestFiniteIteratorMixin:
    """
    Mixin to be used in combination with TestBase
    to test basic function of finite CheckpointableIterators
    """

    def test_basic(self):
        for case_name, expected_result, it in self.test_cases:
            with self.subTest(case_name):
                result = list(it)
                self.assertEqual(result, expected_result)


class TestFiniteIteratorCheckpointingMixin:
    """
    Mixin to be used in combination with TestBase
    to test checkpointing functionality of finite CheckpointableIterators
    """

    def test_checkpointing_reset(self):
        for case_name, _, it in self.test_cases:
            with self.subTest(case_name):
                expected_result = list(it)  # extract data
                it.setstate(None)  # reset to start
                result = list(it)
                self.assertEqual(result, expected_result)

    # TODO: Can this be rewritten in terms of _test_checkpointing_from_pos?
    def test_checkpointing_from_start(self):
        for case_name, _, it in self.test_cases:
            with self.subTest(case_name):
                checkpoint = it.getstate()
                expected_result = list(it)  # extract data
                it.setstate(checkpoint)  # reset to start
                result = list(it)
                self.assertEqual(result, expected_result)

    def _test_checkpointing_from_pos(self, it, pos):
        for _ in range(pos):  # go to pos
            next(it)
        checkpoint = it.getstate()  # take checkpoint
        expected_result = list(it)  # extract data
        it.setstate(checkpoint)  # reset to checkpoint
        result = list(it)
        self.assertEqual(result, expected_result)

    def test_checkpointing_from_one(self):
        for case_name, _, it in self.test_cases:
            with self.subTest(case_name):
                pos = 1
                self._test_checkpointing_from_pos(it, pos)

    def test_checkpointing_from_quarter(self):
        for case_name, _, it in self.test_cases:
            with self.subTest(case_name):
                expected_result = list(it)
                it.setstate(None)
                pos = len(expected_result) // 4
                self._test_checkpointing_from_pos(it, pos)

    def test_checkpointing_from_third(self):
        for case_name, _, it in self.test_cases:
            with self.subTest(case_name):
                expected_result = list(it)
                it.setstate(None)
                pos = len(expected_result) // 3
                self._test_checkpointing_from_pos(it, pos)

    def test_checkpointing_from_half(self):
        for case_name, _, it in self.test_cases:
            with self.subTest(case_name):
                expected_result = list(it)
                it.setstate(None)
                pos = len(expected_result) // 2
                self._test_checkpointing_from_pos(it, pos)

    def test_checkpointing_before_end(self):
        for case_name, _, it in self.test_cases:
            with self.subTest(case_name):
                expected_result = list(it)
                it.setstate(None)
                pos = len(expected_result) - 1
                self._test_checkpointing_from_pos(it, pos)

    def test_checkpointing_at_end(self):
        for case_name, _, it in self.test_cases:
            with self.subTest(case_name):
                list(it)  # exhaust iterator
                self.assertRaises(StopIteration, it.__next__)
                checkpoint = it.getstate()  # take checkpoint
                it.setstate(None)  # reset to beginning
                it.setstate(checkpoint)  # reset to checkpoint
                self.assertRaises(StopIteration, it.__next__)

    def test_checkpointing_complex(self):
        for case_name, _, it in self.test_cases:
            with self.subTest(case_name):
                expected_result = list(it)

                # get a bunch of checkpoints at different positions
                it.setstate(None)
                positions = [
                    0,
                    len(expected_result) // 7,
                    len(expected_result) // 6,
                    len(expected_result) // 5,
                    len(expected_result) // 4,
                    len(expected_result) // 3,
                    len(expected_result) // 2,
                ]
                checkpoints = []
                for i in range(len(positions)):
                    offset = positions[i] - positions[i - 1] if i > 0 else positions[0]
                    for _ in range(offset):
                        next(it)
                    checkpoints.append(it.getstate())

                # check that iterator returns correct result at all checkpoints
                for pos, checkpoint in zip(positions, checkpoints):
                    it.setstate(checkpoint)
                    self.assertEqual(list(it), expected_result[pos:])

                # check that iterator returns correct result at all checkpoints in reverse order
                tuples = list(zip(positions, checkpoints))
                tuples.reverse()
                for pos, checkpoint in tuples:
                    it.setstate(checkpoint)
                    self.assertEqual(list(it), expected_result[pos:])

                # check that iterator returns correct result at all checkpoints
                # while resetting between any two checkpoints
                for pos, checkpoint in zip(positions, checkpoints):
                    it.setstate(None)
                    it.setstate(checkpoint)
                    self.assertEqual(list(it), expected_result[pos:])

                # and as the grand finale: reset and check again
                it.setstate(None)
                result = list(it)
                self.assertEqual(result, expected_result)


class TestInfinitePermutationSourceIterator(TestBase):
    def setUp(self):
        super().setUp()
        self.repeats = [1, 2, 3]

    def test_no_shuffle(self):
        for n, k, num_instances in itertools.product(self.lengths, self.repeats, self.world_sizes):
            data = list(range(n))
            for instance_rank in range(num_instances):
                with self.subTest(f"n={n}, k={k}, num_instances={num_instances}, instance_rank={instance_rank}"):
                    it = InfinitePermutationSourceIterator(
                        copy.deepcopy(data), shuffle=False, num_instances=num_instances, instance_rank=instance_rank
                    )
                    repeated_data = []
                    while len(repeated_data) < k * n * num_instances:
                        repeated_data.extend(data)
                    expected_result = []
                    pos = instance_rank
                    while len(expected_result) < k * n:
                        expected_result.append(repeated_data[pos])
                        pos += num_instances
                    result = [next(it) for _ in range(k * n)]
                    self.assertEqual(result, expected_result)

    def test_shuffle(self):
        for n, k, num_instances in itertools.product(self.lengths, self.repeats, self.world_sizes):
            data = list(range(n))
            for instance_rank in range(num_instances):
                with self.subTest(f"n={n}, k={k}, num_instances={num_instances}, instance_rank={instance_rank}"):
                    it = InfinitePermutationSourceIterator(
                        copy.deepcopy(data),
                        seed=self.seed,
                        shuffle=True,
                        num_instances=num_instances,
                        instance_rank=instance_rank,
                    )
                    random = Random(self.seed)
                    repeated_data = []
                    while len(repeated_data) < k * n * num_instances:
                        shuffled_data = copy.deepcopy(data)
                        random.shuffle(shuffled_data)
                        repeated_data.extend(shuffled_data)
                    expected_result = []
                    pos = instance_rank
                    while len(expected_result) < k * n:
                        expected_result.append(repeated_data[pos])
                        pos += num_instances
                    result = [next(it) for _ in range(k * n)]
                    self.assertEqual(result, expected_result)

    def test_single_instance_no_shuffle(self):
        # this test is technically included in test_no_shuffle
        # but the calculation of the expected result is less error prone
        for n, k in itertools.product(self.lengths, self.repeats):
            with self.subTest(f"n={n}, k={k}"):
                data = list(range(n))
                expected_result = data * k
                it = InfinitePermutationSourceIterator(copy.deepcopy(data), shuffle=False)
                result = [next(it) for _ in range(k * n)]
                self.assertEqual(result, expected_result)

    def test_single_instance_shuffle(self):
        # this test is technically included in test_shuffle
        # but the calculation of the expected result is less error prone
        for n, k in itertools.product(self.lengths, self.repeats):
            with self.subTest(f"n={n}, k={k}"):
                data = list(range(n))
                expected_result = data * k
                it = InfinitePermutationSourceIterator(copy.deepcopy(data), seed=self.seed, shuffle=True)
                result = [next(it) for _ in range(k * n)]
                self.assertMultisetEqual(result, expected_result)

    def test_checkpointing_reset_no_shuffle(self):
        for n, k, num_instances in itertools.product(self.lengths, self.repeats, self.world_sizes):
            data = list(range(n))
            for instance_rank in range(num_instances):
                with self.subTest(f"n={n}, k={k}, num_instances={num_instances}, instance_rank={instance_rank}"):
                    it = InfinitePermutationSourceIterator(
                        copy.deepcopy(data), shuffle=False, num_instances=num_instances, instance_rank=instance_rank
                    )
                    expected_result = [next(it) for _ in range(k * n)]  # extract data
                    it.setstate(None)  # reset to start
                    result = [next(it) for _ in range(k * n)]
                    self.assertEqual(result, expected_result)

    def test_checkpointing_reset_shuffle(self):
        for n, k, num_instances in itertools.product(self.lengths, self.repeats, self.world_sizes):
            data = list(range(n))
            for instance_rank in range(num_instances):
                with self.subTest(f"n={n}, k={k}, num_instances={num_instances}, instance_rank={instance_rank}"):
                    it = InfinitePermutationSourceIterator(
                        copy.deepcopy(data),
                        seed=self.seed,
                        shuffle=True,
                        num_instances=num_instances,
                        instance_rank=instance_rank,
                    )
                    expected_result = [next(it) for _ in range(k * n)]  # extract data
                    it.setstate(None)  # reset to start
                    result = [next(it) for _ in range(k * n)]
                    self.assertEqual(result, expected_result)

    def test_checkpointing_from_start_no_shuffle(self):
        for n, k, num_instances in itertools.product(self.lengths, self.repeats, self.world_sizes):
            data = list(range(n))
            for instance_rank in range(num_instances):
                with self.subTest(f"n={n}, k={k}, num_instances={num_instances}, instance_rank={instance_rank}"):
                    it = InfinitePermutationSourceIterator(
                        copy.deepcopy(data), shuffle=False, num_instances=num_instances, instance_rank=instance_rank
                    )
                    checkpoint = it.getstate()
                    expected_result = [next(it) for _ in range(k * n)]  # extract data
                    it.setstate(checkpoint)  # reset to start
                    result = [next(it) for _ in range(k * n)]
                    self.assertEqual(result, expected_result)

    def test_checkpointing_from_start_shuffle(self):
        for n, k, num_instances in itertools.product(self.lengths, self.repeats, self.world_sizes):
            data = list(range(n))
            for instance_rank in range(num_instances):
                with self.subTest(f"n={n}, k={k}, num_instances={num_instances}, instance_rank={instance_rank}"):
                    it = InfinitePermutationSourceIterator(
                        copy.deepcopy(data),
                        seed=self.seed,
                        shuffle=True,
                        num_instances=num_instances,
                        instance_rank=instance_rank,
                    )
                    checkpoint = it.getstate()
                    expected_result = [next(it) for _ in range(k * n)]  # extract data
                    it.setstate(checkpoint)  # reset to start
                    result = [next(it) for _ in range(k * n)]
                    self.assertEqual(result, expected_result)

    def test_checkpointing_from_middle_no_shuffle(self):
        for n, k, num_instances in itertools.product(self.lengths, self.repeats, self.world_sizes):
            data = list(range(n))
            for instance_rank in range(num_instances):
                with self.subTest(f"n={n}, k={k}, num_instances={num_instances}, instance_rank={instance_rank}"):
                    it = InfinitePermutationSourceIterator(
                        copy.deepcopy(data), shuffle=False, num_instances=num_instances, instance_rank=instance_rank
                    )
                    checkpoint_pos = k * n // 3
                    for _ in range(checkpoint_pos):  # go to checkpoint_pos
                        next(it)
                    checkpoint = it.getstate()  # take checkpoint
                    expected_result = [next(it) for _ in range(k * n)]  # extract data
                    for _ in range(checkpoint_pos):  # move forward some more
                        next(it)
                    it.setstate(checkpoint)  # reset to checkpoint
                    result = [next(it) for _ in range(k * n)]  # get data again
                    self.assertEqual(result, expected_result)

    def test_checkpointing_from_middle_shuffle(self):
        for n, k, num_instances in itertools.product(self.lengths, self.repeats, self.world_sizes):
            data = list(range(n))
            for instance_rank in range(num_instances):
                with self.subTest(f"n={n}, k={k}, num_instances={num_instances}, instance_rank={instance_rank}"):
                    it = InfinitePermutationSourceIterator(
                        copy.deepcopy(data),
                        seed=self.seed,
                        shuffle=True,
                        num_instances=num_instances,
                        instance_rank=instance_rank,
                    )
                    checkpoint_pos = k * n // 3
                    for _ in range(checkpoint_pos):  # go to checkpoint_pos
                        next(it)
                    checkpoint = it.getstate()  # take checkpoint
                    expected_result = [next(it) for _ in range(k * n)]  # extract data
                    for _ in range(checkpoint_pos):  # move forward some more
                        next(it)
                    it.setstate(checkpoint)  # reset to checkpoint
                    result = [next(it) for _ in range(k * n)]  # get data again
                    self.assertEqual(result, expected_result)

    def test_checkpointing_at_boundary_no_shuffle(self):
        for n, k, num_instances in itertools.product(self.lengths, self.repeats, self.world_sizes):
            data = list(range(n))
            for instance_rank in range(num_instances):
                with self.subTest(f"n={n}, k={k}, num_instances={num_instances}, instance_rank={instance_rank}"):
                    it = InfinitePermutationSourceIterator(
                        copy.deepcopy(data), shuffle=False, num_instances=num_instances, instance_rank=instance_rank
                    )
                    checkpoint_pos = k * n
                    for _ in range(checkpoint_pos):  # go to checkpoint_pos
                        next(it)
                    checkpoint = it.getstate()  # take checkpoint
                    expected_result = [next(it) for _ in range(k * n)]  # extract data
                    for _ in range(checkpoint_pos):  # move forward some more
                        next(it)
                    it.setstate(checkpoint)  # reset to checkpoint
                    result = [next(it) for _ in range(k * n)]  # get data again
                    self.assertEqual(result, expected_result)

    def test_checkpointing_at_boundary_shuffle(self):
        for n, k, num_instances in itertools.product(self.lengths, self.repeats, self.world_sizes):
            data = list(range(n))
            for instance_rank in range(num_instances):
                with self.subTest(f"n={n}, k={k}, num_instances={num_instances}, instance_rank={instance_rank}"):
                    it = InfinitePermutationSourceIterator(
                        copy.deepcopy(data),
                        seed=self.seed,
                        shuffle=True,
                        num_instances=num_instances,
                        instance_rank=instance_rank,
                    )
                    checkpoint_pos = k * n
                    for _ in range(checkpoint_pos):  # go to checkpoint_pos
                        next(it)
                    checkpoint = it.getstate()  # take checkpoint
                    expected_result = [next(it) for _ in range(k * n)]  # extract data
                    for _ in range(checkpoint_pos):  # move forward some more
                        next(it)
                    it.setstate(checkpoint)  # reset to checkpoint
                    result = [next(it) for _ in range(k * n)]  # get data again
                    self.assertEqual(result, expected_result)

    def test_empty_source(self):
        f = lambda: InfinitePermutationSourceIterator([])
        self.assertRaises(ValueError, f)

    def test_rank_too_large(self):
        f = lambda: InfinitePermutationSourceIterator([1], num_instances=2, instance_rank=2)
        self.assertRaises(ValueError, f)


class TestChunkedSourceIterator(TestBase, TestFiniteIteratorMixin, TestFiniteIteratorCheckpointingMixin):
    def setUp(self):
        super().setUp()
        self.test_cases = []
        for n in self.lengths:
            data = list(range(n))
            it = ChunkedSourceIterator(copy.deepcopy(data))
            self.test_cases.append(("n={}".format(n), data, it))

    def test_multiple_instances(self):
        for n, num_instances in itertools.product(self.lengths, self.world_sizes):
            with self.subTest("n={}, num_instances={}".format(n, num_instances)):
                data = list(range(n))
                result = []
                sizes = []
                for instance_rank in range(num_instances):
                    it = ChunkedSourceIterator(
                        copy.deepcopy(data), num_instances=num_instances, instance_rank=instance_rank
                    )
                    output = list(it)
                    result.extend(output)
                    sizes.append(len(output))
                self.assertEqual(data, result)
                self.assertTrue(max(sizes) - min(sizes) <= 1)  # make sure data is split as evenly as possible

    def test_rank_too_large(self):
        def create_iterator():
            it = ChunkedSourceIterator([1], num_instances=2, instance_rank=2)

        self.assertRaises(ValueError, create_iterator)


class TestSamplingRandomMapIterator(TestBase, TestFiniteIteratorMixin, TestFiniteIteratorCheckpointingMixin):
    @staticmethod
    def transform(random, item):
        return item + random.random()

    def setUp(self):
        super().setUp()
        self.test_cases = []
        for n in self.lengths:
            data = list(range(n))
            random = Random()
            random.seed(self.seed)
            expected_result = [n + random.random() for n in data]
            it = SamplingRandomMapIterator(NativeCheckpointableIterator(data), transform=self.transform, seed=self.seed)
            self.test_cases.append(("n={}".format(n), expected_result, it))


class TestMapIterator(TestBase, TestFiniteIteratorMixin, TestFiniteIteratorCheckpointingMixin):
    @staticmethod
    def transform(item):
        return 2 * item

    def setUp(self):
        super().setUp()
        self.test_cases = []
        for n in self.lengths:
            data = list(range(n))
            expected_result = [self.transform(item) for item in data]
            it = MapIterator(NativeCheckpointableIterator(data), self.transform)
            self.test_cases.append(("n={}".format(n), expected_result, it))


class TestZipIterator(TestBase, TestFiniteIteratorMixin, TestFiniteIteratorCheckpointingMixin):
    def setUp(self):
        super().setUp()
        self.test_cases = []

        # pairs
        for n in self.lengths:
            data1 = list(range(n))
            data2 = [item * item for item in data1]
            expected_result = list(zip(data1, data2))
            it = ZipIterator(NativeCheckpointableIterator(data1), NativeCheckpointableIterator(data2))
            self.test_cases.append(("n={}, pairs".format(n), expected_result, it))

        # triples
        for n in self.lengths:
            data1 = list(range(n))
            data2 = [item * item for item in data1]
            data3 = [item * item for item in data2]
            expected_result = list(zip(data1, data2, data3))
            it = ZipIterator(
                NativeCheckpointableIterator(data1),
                NativeCheckpointableIterator(data2),
                NativeCheckpointableIterator(data3),
            )
            self.test_cases.append(("n={}, triples".format(n), expected_result, it))

        # different lengths
        for n in self.lengths:
            if n > 3:  # smaller n give us an empty iterator, which causes issues
                data1 = list(range(n))
                data2 = [item * item for item in data1]
                data2 = data2[:-3]
                expected_result = list(zip(data1, data2))
                it = ZipIterator(NativeCheckpointableIterator(data1), NativeCheckpointableIterator(data2))
                self.test_cases.append(("n={}, different lengths".format(n), expected_result, it))


class TestPrefetchIterator(TestBase, TestFiniteIteratorMixin, TestFiniteIteratorCheckpointingMixin):
    def setUp(self):
        super().setUp()
        self.test_cases = []
        for n in self.lengths:
            for buffer_size in self.lengths:
                data = list(range(n))
                it = PrefetchIterator(NativeCheckpointableIterator(data), buffer_size)
                self.test_cases.append(("n={}, buffer_size={}".format(n, buffer_size), data, it))

    def test_zero_buffer_size(self):
        f = lambda: PrefetchIterator(NativeCheckpointableIterator([0]), buffer_size=0)
        self.assertRaises(ValueError, f)

    def test_torch_tensors(self):
        for n in self.lengths:
            for buffer_size in self.lengths:
                with self.subTest("n={}, buffer_size={}".format(n, buffer_size)):
                    data = [torch.Tensor([float(i)]) for i in range(n)]
                    it = PrefetchIterator(NativeCheckpointableIterator(copy.deepcopy(data)), buffer_size)
                    result = list(it)
                    self.assertEqual(result, data)


class TestPrefetchIteratorExperimental(TestBase, TestFiniteIteratorMixin, TestFiniteIteratorCheckpointingMixin):
    def setUp(self):
        super().setUp()
        self.test_cases = []
        for n in self.lengths:
            for buffer_size in self.lengths:
                data = list(range(n))
                it = PrefetchIterator(NativeCheckpointableIterator(data), buffer_size, buffer_in_main_process=True)
                self.test_cases.append(("n={}, buffer_size={}".format(n, buffer_size), data, it))

    def test_zero_buffer_size(self):
        f = lambda: PrefetchIterator(NativeCheckpointableIterator([0]), buffer_size=0, buffer_in_main_process=True)
        self.assertRaises(ValueError, f)

    def test_closing(self):
        if multiprocessing.get_start_method() != "fork":
            return  # dummy iterator used, skip test
        it = PrefetchIterator(NativeCheckpointableIterator([0]), buffer_size=42, buffer_in_main_process=True)
        it.close()
        f = lambda: it.__next__()
        self.assertRaises(RuntimeError, f)
        f = lambda: it.setstate(None)
        self.assertRaises(RuntimeError, f)

    def test_nested(self):
        for n in self.lengths:
            for buffer_size in self.lengths:
                for depth in [2, 3, 4, 5]:
                    with self.subTest("n={}, buffer_size={}, depth={}".format(n, buffer_size, depth)):
                        data = [torch.Tensor([float(i)]) for i in range(n)]
                        it = NativeCheckpointableIterator(copy.deepcopy(data))
                        for _ in range(depth):
                            it = PrefetchIterator(it, buffer_size, buffer_in_main_process=True)
                        result = list(it)
                        self.assertEqual(result, data)
                        it.close()

    def test_torch_tensors(self):
        for n in self.lengths:
            for buffer_size in self.lengths:
                with self.subTest("n={}, buffer_size={}".format(n, buffer_size)):
                    data = [torch.Tensor([float(i)]) for i in range(n)]
                    it = PrefetchIterator(
                        NativeCheckpointableIterator(copy.deepcopy(data)), buffer_size, buffer_in_main_process=True
                    )
                    result = list(it)
                    self.assertEqual(result, data)
                    it.close()

    def tearDown(self):
        if hasattr(self, "test_cases"):
            for _, _, it in self.test_cases:
                it.close()


class TestMultiplexIterator(TestBase, TestFiniteIteratorMixin, TestFiniteIteratorCheckpointingMixin):
    # TODO: Add test cases for behavior when source iterators end but item is retrieved
    def setUp(self):
        super().setUp()
        random = Random()
        random.seed(42)
        self.test_cases = []

        # two source iterators
        for n in self.lengths:
            indices = [random.randrange(0, 2) for _ in range(n)]
            data = [[2 * i + 0 for i in range(n)], [2 * i + 1 for i in range(n)]]
            data_copy = copy.deepcopy(data)
            expected_result = [data_copy[i].pop(0) for i in indices]
            it = MultiplexIterator(
                NativeCheckpointableIterator(indices), [NativeCheckpointableIterator(d) for d in data]
            )
            self.test_cases.append(("n={}, two source iterators".format(n), expected_result, it))

        # three source iterators
        for n in self.lengths:
            indices = [random.randrange(0, 3) for _ in range(n)]
            data = [[3 * i + 0 for i in range(n)], [3 * i + 1 for i in range(n)], [3 * i + 2 for i in range(n)]]
            data_copy = copy.deepcopy(data)
            expected_result = [data_copy[i].pop(0) for i in indices]
            it = MultiplexIterator(
                NativeCheckpointableIterator(indices), [NativeCheckpointableIterator(d) for d in data]
            )
            self.test_cases.append(("n={}, three source iterators".format(n), expected_result, it))


class TestNativeCheckpointableIterator(TestBase, TestFiniteIteratorMixin, TestFiniteIteratorCheckpointingMixin):
    def setUp(self):
        super().setUp()
        self.test_cases = []
        for n in self.lengths:
            data = list(range(n))
            expected_result = copy.deepcopy(data)
            it = NativeCheckpointableIterator(data)
            self.test_cases.append(("n={}".format(n), expected_result, it))

    def test_empty(self):
        it = NativeCheckpointableIterator([])
        self.assertRaises(StopIteration, it.__next__)

    def test_iterator_exception(self):
        self.assertRaises(ValueError, NativeCheckpointableIterator, iter(range(10)))


class TestFixedBatchIterator(TestBase, TestFiniteIteratorMixin, TestFiniteIteratorCheckpointingMixin):
    def setUp(self):
        super().setUp()
        self.test_cases = []
        for n in self.lengths:
            for batch_size in self.lengths:
                data = list(range(n))
                data_copy = copy.deepcopy(data)
                expected_result = []
                while data_copy:
                    expected_result.append(data_copy[:batch_size])
                    data_copy = data_copy[batch_size:]
                it = FixedBatchIterator(NativeCheckpointableIterator(data), batch_size=batch_size)
                self.test_cases.append(("n={}, batch_size={}".format(n, batch_size), expected_result, it))

    def test_invalid_batch_size(self):
        f = lambda: FixedBatchIterator(NativeCheckpointableIterator([0]), batch_size=0)
        self.assertRaises(ValueError, f)


class TestRecurrentIterator(TestBase, TestFiniteIteratorMixin, TestFiniteIteratorCheckpointingMixin):
    @staticmethod
    def step_function(prev_state, item):
        output = prev_state + item
        return output, output

    def setUp(self):
        super().setUp()
        self.test_cases = []
        for n in self.lengths:
            data = list(range(n))
            expected_result = [data[0]]
            for i in data[1:]:
                expected_result.append(self.step_function(expected_result[-1], i)[1])
            it = RecurrentIterator(NativeCheckpointableIterator(data), self.step_function, initial_state=0)
            self.test_cases.append(("n={}".format(n), expected_result, it))


class TestSelectManyIterator(TestBase, TestFiniteIteratorMixin, TestFiniteIteratorCheckpointingMixin):
    @staticmethod
    def custom_selector(l):
        return [l[0]]

    def setUp(self):
        super().setUp()
        self.test_cases = []

        # default selector
        for n in self.lengths:
            for list_length in [1, 4, 9]:
                data = list(range(n))
                expected_result = copy.deepcopy(data)
                lists = []
                while data:
                    lists.append(data[:list_length])
                    data = data[list_length:]
                it = SelectManyIterator(NativeCheckpointableIterator(lists))
                self.test_cases.append(
                    ("n={}, list_length={}, default selector".format(n, list_length), expected_result, it)
                )

        # custom selector
        for n in self.lengths:
            for list_length in [4, 9]:
                data = list(range(n))
                expected_result = [item for i, item in enumerate(data) if (i % list_length) == 0]
                lists = []
                while data:
                    lists.append(data[:list_length])
                    data = data[list_length:]
                it = SelectManyIterator(NativeCheckpointableIterator(lists), collection_selector=self.custom_selector)
                self.test_cases.append(
                    ("n={}, list_length={}, custom selector".format(n, list_length), expected_result, it)
                )


class TestBlockwiseShuffleIterator(TestBase, TestFiniteIteratorCheckpointingMixin):
    def setUp(self):
        super().setUp()
        self.test_cases = []
        for n in self.lengths:
            for block_size in self.lengths:
                data = list(range(n))
                it = BlockwiseShuffleIterator(NativeCheckpointableIterator(copy.deepcopy(data)), block_size, self.seed)
                self.test_cases.append(("n={}, block_size={}".format(n, block_size), data, it))

    def test_basic(self):
        for case_name, expected_result, it in self.test_cases:
            with self.subTest(case_name):
                result = list(it)
                self.assertMultisetEqual(result, expected_result)


class TestWindowedIterator(TestBase, TestFiniteIteratorMixin, TestFiniteIteratorCheckpointingMixin):
    def setUp(self):
        super().setUp()
        self.test_cases = []
        for n in self.lengths:
            for window_size in self.lengths:
                if n < window_size:
                    continue
                data = list(range(n))
                it = WindowedIterator(NativeCheckpointableIterator(copy.deepcopy(data)), window_size)
                expected_result = []
                for i in range(len(data)):
                    if i + window_size > len(data):
                        break
                    expected_result.append(tuple(data[i : i + window_size]))
                self.test_cases.append(("n={}, window_size={}".format(n, window_size), expected_result, it))


class TestSourceIterator(TestBase):
    # TODO: Do we need more tests for this?
    def test_exception(self):
        self.assertRaises(ValueError, create_source_iterator, [1], train=False, shuffle=True)


class TestBucketedReadaheadBatchIterator(TestBase, TestFiniteIteratorCheckpointingMixin):
    dynamic_batch_size = 15

    @staticmethod
    def key_fn(item):
        return len(item)

    @staticmethod
    def batch_size_fn(item):
        return TestBucketedReadaheadBatchIterator.dynamic_batch_size // len(item)

    @staticmethod
    def boundary_key_fn(item):
        return len(item) < 5

    @staticmethod
    def setup_data(n):
        data = []
        for i in range(n):
            data.append(tuple(range(i % 10 + 1)))
        return data

    def setUp(self):
        super().setUp()
        self.batch_sizes = [1, 2, 3, 9]
        self.test_cases = []

        # fixed batch size, not shuffled, no boundary key
        for n, read_ahead in itertools.product(self.lengths, self.lengths):
            for batch_size in self.batch_sizes:
                data = self.setup_data(n)
                it = BucketedReadaheadBatchIterator(
                    NativeCheckpointableIterator(copy.deepcopy(data)),
                    read_ahead=read_ahead,
                    key=self.key_fn,
                    batch_size=batch_size,
                    shuffle=False,
                )
                self.test_cases.append(
                    (
                        "n={}, read_ahead={}, batch_size={}, boundary_key=None, shuffled=False".format(
                            n, read_ahead, batch_size
                        ),
                        data,
                        it,
                    )
                )

        # fixed batch size, shuffled, no boundary key
        for n, read_ahead in itertools.product(self.lengths, self.lengths):
            for batch_size in self.batch_sizes:
                data = self.setup_data(n)
                it = BucketedReadaheadBatchIterator(
                    NativeCheckpointableIterator(copy.deepcopy(data)),
                    read_ahead=read_ahead,
                    key=self.key_fn,
                    batch_size=batch_size,
                    shuffle=True,
                    seed=self.seed,
                )
                self.test_cases.append(
                    (
                        "n={}, read_ahead={}, batch_size={}, boundary_key=None, shuffled=True".format(
                            n, read_ahead, batch_size
                        ),
                        data,
                        it,
                    )
                )

        # dynamic batch size, not shuffled, no boundary key
        for n, read_ahead in itertools.product(self.lengths, self.lengths):
            data = self.setup_data(n)
            it = BucketedReadaheadBatchIterator(
                NativeCheckpointableIterator(copy.deepcopy(data)),
                read_ahead=read_ahead,
                key=self.key_fn,
                batch_size=self.batch_size_fn,
                shuffle=False,
            )
            self.test_cases.append(
                (
                    "n={}, read_ahead={}, batch_size=dynamic, boundary_key=None, shuffled=False".format(n, read_ahead),
                    data,
                    it,
                )
            )

        # dynamic batch size, shuffled, no boundary key
        for n, read_ahead in itertools.product(self.lengths, self.lengths):
            data = self.setup_data(n)
            it = BucketedReadaheadBatchIterator(
                NativeCheckpointableIterator(copy.deepcopy(data)),
                read_ahead=read_ahead,
                key=self.key_fn,
                batch_size=self.batch_size_fn,
                shuffle=True,
                seed=self.seed,
            )
            self.test_cases.append(
                (
                    "n={}, read_ahead={}, batch_size=dynamic, boundary_key=None, shuffled=True".format(n, read_ahead),
                    data,
                    it,
                )
            )

        # fixed batch size, not shuffled, boundary key
        for n, read_ahead in itertools.product(self.lengths, self.lengths):
            for batch_size in self.batch_sizes:
                data = self.setup_data(n)
                it = BucketedReadaheadBatchIterator(
                    NativeCheckpointableIterator(copy.deepcopy(data)),
                    read_ahead=read_ahead,
                    key=self.key_fn,
                    batch_size=batch_size,
                    boundary_key=self.boundary_key_fn,
                    shuffle=False,
                )
                self.test_cases.append(
                    (
                        "n={}, read_ahead={}, batch_size={}, boundary_key=len(item)<5, shuffled=False".format(
                            n, read_ahead, batch_size
                        ),
                        data,
                        it,
                    )
                )

        # fixed batch size, shuffled, boundary key
        for n, read_ahead in itertools.product(self.lengths, self.lengths):
            for batch_size in self.batch_sizes:
                data = self.setup_data(n)
                it = BucketedReadaheadBatchIterator(
                    NativeCheckpointableIterator(copy.deepcopy(data)),
                    read_ahead=read_ahead,
                    key=self.key_fn,
                    batch_size=batch_size,
                    boundary_key=self.boundary_key_fn,
                    shuffle=True,
                    seed=self.seed,
                )
                self.test_cases.append(
                    (
                        "n={}, read_ahead={}, batch_size={}, boundary_key=len(item)<5, shuffled=True".format(
                            n, read_ahead, batch_size
                        ),
                        data,
                        it,
                    )
                )

        # dynamic batch size, not shuffled, boundary key
        for n, read_ahead in itertools.product(self.lengths, self.lengths):
            data = self.setup_data(n)
            it = BucketedReadaheadBatchIterator(
                NativeCheckpointableIterator(copy.deepcopy(data)),
                read_ahead=read_ahead,
                key=self.key_fn,
                batch_size=self.batch_size_fn,
                boundary_key=self.boundary_key_fn,
                shuffle=False,
                seed=self.seed,
            )
            self.test_cases.append(
                (
                    "n={}, read_ahead={}, batch_size=dynamic, boundary_key=len(item)<5, shuffled=False".format(
                        n, read_ahead
                    ),
                    data,
                    it,
                )
            )

        # dynamic batch size, shuffled, boundary key
        for n, read_ahead in itertools.product(self.lengths, self.lengths):
            data = self.setup_data(n)
            it = BucketedReadaheadBatchIterator(
                NativeCheckpointableIterator(copy.deepcopy(data)),
                read_ahead=read_ahead,
                key=self.key_fn,
                batch_size=self.batch_size_fn,
                boundary_key=self.boundary_key_fn,
                shuffle=True,
                seed=self.seed,
            )
            self.test_cases.append(
                (
                    "n={}, read_ahead={}, batch_size=dynamic, boundary_key=len(item)<5, shuffled=True".format(
                        n, read_ahead
                    ),
                    data,
                    it,
                )
            )

    def test_basic(self):
        for case_name, expected_result, it in self.test_cases:
            with self.subTest(case_name):
                result = list(it)
                flattened_result = [item for batch in result for item in batch]
                self.assertMultisetEqual(flattened_result, expected_result)

    def test_max_len(self):
        for case_name, expected_result, it in self.test_cases:
            if "batch_size=dynamic" in case_name:
                with self.subTest(case_name):
                    result = list(it)
                    for batch in result:
                        length = sum((len(item) for item in batch))
                        self.assertTrue(length <= TestBucketedReadaheadBatchIterator.dynamic_batch_size)

    def test_boundary_key(self):
        for case_name, expected_result, it in self.test_cases:
            if "boundary_key=len(item)<5" in case_name:
                with self.subTest(case_name):
                    result = list(it)
                    for batch in result:
                        boundary_keys = [self.boundary_key_fn(item) for item in batch]
                        self.assertTrue(all(boundary_keys) or not any(boundary_keys))
