"""
Infinibatch is a library of checkpointable iterators for randomized data loading of massive data sets in deep neural network training.


## Features

  * support for corpora much larger than fit into RAM
  * hierarchical block+sentence-level randomization over the whole corpus, different randomization in each epoch
  * only load the data that is needed
  * very fast start-up time (does not need to read full corpus)
  * only requires the most basic of data preparation (e.g. no indexing)
  * for multi-GPU, only load what the respective GPU needs
  * 100% accurate check-pointing, restore from checkpoint should not read all data up to the checkpoint
  * support automatic bucketed batching with dynamic batch sizes
  * pre-fetching thread
  * composable, as to support for complex batching, e.g. negative samples from multiple documents


## Getting Started

Infinibatch requires Python 3.5 and has no dependencies.
There is presently no pip package.

To install it, see README.md

## Tutorial

This little tutorial walks you through the steps of preparing your data and consuming them from Python code as batches.

### Infinibatch Basics: Iterators and Checkpointing

Infinibatch provides [Python iterators](https://docs.python.org/3.5/glossary.html#term-iterator)
to read your data.
An iterator represents a stream of data that can be retrieved item by item, e.g. via a
`for` loop or repeatedly calling `next()` on it.

Infinibatch is agnostic to the data type of the items, which is determined by a user-supplied file-read function.
In NLP applications, items would typically be tuples of text. In other applications,
they can be images or an audio file with a textual annotation.

Infinibatch makes it easy to read your data in randomized order, and supports checkpointing, which allows you to restart training exactly where you left off.

Randomization is done _on the fly_, which means that it is not necessary to read the entire data set into memory
to be shuffled. Infinibatch implements a hierarchical shuffling algorithm
that only holds a subset of the data in RAM at any point in time.

Infinibatch iterators are _checkpointable_.
Checkpointing lets you retrieve the current position (the "checkpoint") in the data stream at any time, so that
later, you can "rewind" to that same position.
The sad reality is that long-running trainings occasionally crash.
To be able to continue a crashed training as if it had not crashed,
save your Infinibatch iterator's checkpoint to disk whenever you save an intermediate model during training.
To restart a crashed training, reset the iterator to the saved checkpoint.
The data reader will now yield the exact same data-item sequence it would have yielded without the crash.

### Data Preparation

Infinibatch has one requirement on your data organization:
To use your data with Infinibatch, it must be split into a large number of small chunks.
A chunk is the smallest unit of data that is loaded from disk into RAM. Infinibatch holds a random subset of chunks in memory
that it randomly draws samples from.

Below we want to show how such a split can be created. An easy way to split your data into chunks is with the Linux `split` command.

In this tutorial, our "corpus" consists of 6 lines of text, where each line is one data item.
To create that corpus, please run this command in a bash shell. It creates a 6-line text file named `corpus.txt`:
```bash
echo \\
'Lorem ipsum dolor sit amet,
consectetur adipiscing elit,
sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.
The quick brown fox jumps over the lazy dog.' \\
> corpus.txt
```
Now let us split it into 3 chunks of 2 lines each. Each chunk is stored as a zipped text file.
We will create them inside a new subdirectory called `corpus_chunks`:
```bash
mkdir corpus_chunks
split  --lines 2  --numeric-suffixes                 \\
       --filter 'gzip > corpus_chunks/$FILE.txt.gz'  \\
       corpus.txt  corpus.
```
This will have created three files: `corpus_chunks/corpus.00.txt.gz`, `corpus_chunks/corpus.01.txt.gz`, and `corpus_chunks/corpus.02.txt.gz`.
To verify whether the data has been split as expected, you can use this command:
```bash
zcat corpus_chunks/corpus.*.txt.gz
```

Hint: For large corpora, we recommend replacing `gzip` by `pigz` (`apt-get install pigz`), which runs notably faster via multi-threading.

### Reading Items in Random Order With Infinibatch

We will first show the easiest way to read data with Infinibatch, using the helper function `chunked_dataset_iterator``()`.
This function will create an Infinibatch iterator that yields the content of your data in random order.
Please the following program:
```python
import gzip, glob

from infinibatch import datasets as ds

ds = ds.chunked_dataset_iterator(
    chunk_refs = glob.glob('corpus_chunks/corpus.*.txt.gz'),
    read_chunk_fn = lambda path: iter(gzip.decompress(open(path, "rb")  \\
                                      .read()).decode(encoding='utf-8') \\
                                      .splitlines()),
    buffer_size = 6, seed = 1)

for i in range(10):
    print(next(ds))
```
You should get output that contains the 6 example lines in randomized order:
```text
Lorem ipsum dolor sit amet,
consectetur adipiscing elit,
Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.
The quick brown fox jumps over the lazy dog.
sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
consectetur adipiscing elit,
Lorem ipsum dolor sit amet,
The quick brown fox jumps over the lazy dog.
sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
```
Note: The `buffer_size` parameter determines how many sentences are read into memory at any given time,
to draw randomized items from. In real settings with corpora of hundreds of millions of text lines,
the `buffer_size` parameter should be set in the millions.
RAM usage and startup time will be proportional to the buffer size
(but much lower than having to load the entire corpus into RAM).

### Reading Items of Different Lengths in Batches

For deep learning, we want to group multiple items into batches.
For NLP tasks, items are often lines of text of varying length.
Infinibatch implements an algorithm that randomizes the input sequence and groups it into
batches of approximately the same length (aka _bucketing_).

Infinibatch's `BucketedReadaheadBatchIterator` performs this task.
It implements an algorithm modeled after the [Marian toolkit](https://github.com/marian-nmt/marian)
that preloads a large number of randomized items (typically millions; in this example: 6),
sorts them and groups them into batches of similar length, and then yields
them, in turn, in randomized order.

Here is an example. Note that the `BucketedReadaheadBatchIterator` accepts
the previous randomized sentence sequence iterator (`ds`) as the source of items to randomize over.
This is an example how one forms pipelines of iterators with Infinibatch
(a concept familiar from Python's own `itertools`).
Once an iterator is passed to another as its source, consider it owned by that other iterator,
it must no longer be accessed by the calling code.
```python
import gzip, glob

from infinibatch import datasets as ds
from infinibatch import iterators as it

ds = ds.chunked_dataset_iterator(
    chunk_refs = glob.glob('corpus_chunks/corpus.*.txt.gz'),
    read_chunk_fn = lambda path: iter(gzip.decompress(open(path, "rb")  \\
                                      .read()).decode(encoding='utf-8') \\
                                      .splitlines()),
    buffer_size = 6, seed = 1)

bs = it.BucketedReadaheadBatchIterator(
    source_iterator = ds,   # note: this is the iterator from above
    read_ahead = 6,
    key = lambda line: len(line),
    batch_size = 2,
    seed = 1)

for i in range(25):
    print(next(bs))
```
This code should output something like this:
```python
['sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.',
 'The quick brown fox jumps over the lazy dog.']
['consectetur adipiscing elit,', 'Lorem ipsum dolor sit amet,']
['Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.',
 'Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.']
```
followed by different permutations of the same tuples.
As you can see, the sentences are in random order and grouped in batches of 2 of approximately the same length.
You may notice that there is no variation in how the items get grouped into batches--that
is an artifact of this example, and generally not the case in real use when the data size is much larger
than the batch size.

In NLP, sentence length often varies considerably. As a result, using batches of a fixed number of lines,
as in the example above, will waste GPU RAM and cores.
This is because the number of lines is limited by the longest possible sequence; batches of shorter lines
would leave GPU cycles on the table.
Ideally, one would use batches that have as many lines as fit into GPU RAM,
given the number of tokens of the longest line in the batch.
To support variable batch sizes, Infinibatch allows to pass a function as the `batch_size` parameter.
That function will be given the longest item of a batch and should estimate how many items of at most this length can fit.

In our example, we assume that batches can hold at most 150 tokens.
Please change the above code as follows:
```python
    batch_size = lambda longest_line: 150 // len(longest_line),
```
The output looks like this:
```
['consectetur adipiscing elit,', 'Lorem ipsum dolor sit amet,']
['Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.']
['sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.',
 'The quick brown fox jumps over the lazy dog.']
['Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.']
```
That shorter sentences got grouped, while longer did not because they would exceed the total of 150 characters.

### Reading Batches Into Numpy Arrays

Lastly, we will need to feed batches into our favorite deep-learning tool.
We will show how to convert the batches of text lines into padded `numpy` arrays.

In a typical NLP application, text items would be tokenized, and then each token
would be represented by an index into a unit vocabulary.
For simplicity, in this example each character is its own token,
and each token's numeric unit index is just its ASCII code.
These sequences are then padded to equal length with -1, and converted into a `numpy` array.

Please rerun the previous example, but first insert the following code before the final `for` loop.
This example uses an Infinibatch `MapIterator`, which applies a user-supplied function or
lambda to each item:
```python
import numpy as np
def collate(lines_batch):
    # tokenize all lines in the batch and map to unit ids
    ids_batch = [[ord(c) for c in line] for line in lines_batch]
    # create a padded numpy array as wide as the longest line,
    # where shorter sequences are padded with -1
    width = max(len(ids) for ids in ids_batch)
    return np.array([ids + [-1] * (width-len(ids)) for ids in ids_batch])

bs = it.MapIterator(
    source_iterator = bs,
    transform = collate)
```
This will output batches like this. Note that in batches with multiple sentences,
some entries are padded with `-1`.
```python
[[ 99 111 110 115 101  99 116 101 116 117 114  32  97 100 105 112 105 115
   99 105 110 103  32 101 108 105 116  44]
 [ 76 111 114 101 109  32 105 112 115 117 109  32 100 111 108 111 114  32
  115 105 116  32  97 109 101 116  44  -1]]
[[ 85 116  32 101 110 105 109  32  97 100  32 109 105 110 105 109  32 118
  101 110 105  97 109  44  32 113 117 105 115  32 110 111 115 116 114 117
  100  32 101 120 101 114  99 105 116  97 116 105 111 110  32 117 108 108
   97 109  99 111  32 108  97  98 111 114 105 115  32 110 105 115 105  32
  117 116  32  97 108 105 113 117 105 112  32 101 120  32 101  97  32  99
  111 109 109 111 100 111  32  99 111 110 115 101 113 117  97 116  46]]
[[115 101 100  32 100 111  32 101 105 117 115 109 111 100  32 116 101 109
  112 111 114  32 105 110  99 105 100 105 100 117 110 116  32 117 116  32
  108  97  98 111 114 101  32 101 116  32 100 111 108 111 114 101  32 109
   97 103 110  97  32  97 108 105 113 117  97  46]
 [ 84 104 101  32 113 117 105  99 107  32  98 114 111 119 110  32 102 111
  120  32 106 117 109 112 115  32 111 118 101 114  32 116 104 101  32 108
   97 122 121  32 100 111 103  46  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1
   -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1]]
[[ 68 117 105 115  32  97 117 116 101  32 105 114 117 114 101  32 100 111
  108 111 114  32 105 110  32 114 101 112 114 101 104 101 110 100 101 114
  105 116  32 105 110  32 118 111 108 117 112 116  97 116 101  32 118 101
  108 105 116  32 101 115 115 101  32  99 105 108 108 117 109  32 100 111
  108 111 114 101  32 101 117  32 102 117 103 105  97 116  32 110 117 108
  108  97  32 112  97 114 105  97 116 117 114  46]]
```

## Where To Go From Here

The above tutorial showed you the use of the most common iterator type, as created by the
convenience function `chunked_dataset_iterator()`.

Not all real-life scenarios are covered by this function. For example, multi-task learning
scenarios require more complex combinations of data. To create those, you will need
to compose the necessary data reader from the underlying building blocks.
This is described at the documentation of the module `iterators`.
"""

from .iterators import *
