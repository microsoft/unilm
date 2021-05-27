# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A simple invertible tokenizer.

Converts from a unicode string to a list of tokens
(represented as Unicode strings).

This tokenizer has the following desirable properties:
 - It is invertible.
 - Alphanumeric characters are broken away from non-alphanumeric characters.
 - A single space between words does not produce an extra token.
 - The full Unicode punctuation and separator set is recognized.

The tokenization algorithm is as follows:

1.  Split the text into a list of tokens, splitting at every boundary of an
    alphanumeric character and a non-alphanumeric character.  This produces
    a list which alternates between "alphanumeric tokens"
    (strings of alphanumeric characters) and "non-alphanumeric tokens"
    (strings of non-alphanumeric characters).

2.  Remove every token consisting of a single space, unless it is
    the very first or very last token in the list.  These tokens are now
    implied by the fact that there are two adjacent alphanumeric tokens.

e.g.  u"Dude - that's so cool."
        -> [u"Dude", u" - ", u"that", u"'", u"s", u"so", u"cool", u"."]
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import sys
import unicodedata
import six
import logging
from six.moves import range  # pylint: disable=redefined-builtin
# from tensor2tensor.utils import mlperf_log
import time
import glob

# Conversion between Unicode and UTF-8, if required (on Python2)
_native_to_unicode = (lambda s: s.decode("utf-8")) if six.PY2 else (lambda s: s)

logger = logging.getLogger(__name__)
# This set contains all letter and number characters.
_ALPHANUMERIC_CHAR_SET = set(
    six.unichr(i) for i in range(sys.maxunicode)
    if (unicodedata.category(six.unichr(i)).startswith("L") or
        unicodedata.category(six.unichr(i)).startswith("N") or
        unicodedata.category(six.unichr(i)).startswith("P")))
        # unicodedata.category(six.unichr(i)).startswith("S")


def encode(text):
  """Encode a unicode string as a list of tokens.

  Args:
    text: a unicode string
  Returns:
    a list of tokens as Unicode strings
  """
  if not text:
    return []
  ret = []
  token_start = 0
  # Classify each character in the input string
  is_alnum = [c in _ALPHANUMERIC_CHAR_SET for c in text]
  add_remaining = False
  for pos in range(1, len(text)):
    add_remaining = False
    if is_alnum[pos] != is_alnum[pos - 1]:
      if not is_alnum[pos]:
        token = text[token_start:pos]
        if token != u" " or token_start == 0:
          add_remaining = False
          ret.append(token)
      else:
        add_remaining = True
        token_start = pos

  final_token = text[token_start:] if text[-1] in _ALPHANUMERIC_CHAR_SET else text[token_start:-1]

  if add_remaining:
    ret.append(final_token)

  # split on punctuation
  final_tokens = []
  for token in ret:
    splitted_token = _run_split_on_punc(token)
    final_tokens.extend(splitted_token)

  return final_tokens


def _run_split_on_punc(text, never_split=None):
  """Splits punctuation on a piece of text."""
  if never_split is not None and text in never_split:
    return [text]
  chars = list(text)
  i = 0
  start_new_word = True
  output = []
  while i < len(chars):
    char = chars[i]
    if _is_punctuation(char):
      output.append([char])
      start_new_word = True
    else:
      if start_new_word:
        output.append([])
      start_new_word = False
      output[-1].append(char)
    i += 1

  return ["".join(x) for x in output]


def _is_punctuation(char):
  """Checks whether `chars` is a punctuation character."""
  cp = ord(char)
  # We treat all non-letter/number ASCII as punctuation.
  # Characters such as "^", "$", and "`" are not in the Unicode
  # Punctuation class but we treat them as punctuation anyways, for
  # consistency.
  if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
    return True
  cat = unicodedata.category(char)
  if cat.startswith("P"):
    return True
  return False


def decode(tokens):
  """Decode a list of tokens to a unicode string.

  Args:
    tokens: a list of Unicode strings
  Returns:
    a unicode string
  """
  token_is_alnum = [t[0] in _ALPHANUMERIC_CHAR_SET for t in tokens]
  ret = []
  for i, token in enumerate(tokens):
    if i > 0 and token_is_alnum[i - 1] and token_is_alnum[i]:
      ret.append(u" ")
    ret.append(token)
  return "".join(ret)


def _read_filepattern(filepattern, max_lines=None, split_on_newlines=True, do_lower_case=False):
  """Reads files matching a wildcard pattern, yielding the contents.

  Args:
    filepattern: A wildcard pattern matching one or more files.
    max_lines: If set, stop reading after reading this many lines.
    split_on_newlines: A boolean. If true, then split files by lines and strip
        leading and trailing whitespace from each line. Otherwise, treat each
        file as a single string.

  Yields:
    The contents of the files as lines, if split_on_newlines is True, or
    the entire contents of each file if False.
  """
  filenames = sorted(glob.glob(filepattern))
  print(filenames, 'do lower case:', do_lower_case)
  lines_read = 0
  for filename in filenames:
    start = time.time()
    with open(filename) as f:
      if split_on_newlines:
        for line in f:
          if do_lower_case:
            line = line.lower()
          yield line.strip()
          lines_read += 1
          if max_lines and lines_read >= max_lines:
            return
          if lines_read % 100000 == 0:
            print("read", lines_read, "lines,", time.time() - start, "secs elapsed")

      else:
        if max_lines:
          doc = []
          for line in f:
            if do_lower_case:
              line = line.lower()
            doc.append(line)
            lines_read += 1
            if max_lines and lines_read >= max_lines:
              yield "".join(doc)
              return
          yield "".join(doc)

        else:
          yield f.read()

    print(time.time() - start, "for reading read file :", filename)


def corpus_token_counts(
    text_filepattern, corpus_max_lines, split_on_newlines=True, additional_chars="", do_lower_case=False):
  """Read the corpus and compute a dictionary of token counts.

  Args:
    text_filepattern: A pattern matching one or more files.
    corpus_max_lines: An integer; maximum total lines to read.
    split_on_newlines: A boolean. If true, then split files by lines and strip
        leading and trailing whitespace from each line. Otherwise, treat each
        file as a single string.
    additional_chars: A String. Each consisting characters will be treat as normal
        alphabets so that they will be included in each vocab.

  Returns:
    a dictionary mapping token to count.
  """
  if additional_chars:
    _ALPHANUMERIC_CHAR_SET.add(additional_chars)

  counts = collections.Counter()
  for doc in _read_filepattern(
      text_filepattern,
      max_lines=corpus_max_lines,
      split_on_newlines=split_on_newlines,
      do_lower_case=do_lower_case):
    counts.update(encode(_native_to_unicode(doc)))
  print("read all files")
  return counts


def vocab_token_counts(text_filepattern, max_lines, do_lower_case=False):
  """Read a vocab file and return a dictionary of token counts.

  Reads a two-column CSV file of tokens and their frequency in a dataset. The
  tokens are presumed to be generated by encode() or the equivalent.

  Args:
    text_filepattern: A pattern matching one or more files.
    max_lines: An integer; maximum total lines to read.

  Returns:
    a dictionary mapping token to count.
  """
  ret = {}
  for i, line in enumerate(
      _read_filepattern(text_filepattern, max_lines=max_lines)):
    if "," not in line:
      logger.warning("Malformed vocab line #%d '%s'", i, line)
      continue
    
    if do_lower_case:
      line = line.lower()

    token, count = line.rsplit(",", 1)
    ret[_native_to_unicode(token)] = int(count)

  return ret
