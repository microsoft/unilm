"""
#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

"""

mkdir data

# WMT16 EN-RO
cd data
mkdir wmt16.en-ro
cd wmt16.en-ro
gdown https://drive.google.com/uc?id=1YrAwCEuktG-iDVxtEW-FE72uFTLc5QMl
tar -zxvf wmt16.tar.gz
mv wmt16/en-ro/train/corpus.bpe.en train.en
mv wmt16/en-ro/train/corpus.bpe.ro train.ro
mv wmt16/en-ro/dev/dev.bpe.en valid.en
mv wmt16/en-ro/dev/dev.bpe.ro valid.ro
mv wmt16/en-ro/test/test.bpe.en test.en
mv wmt16/en-ro/test/test.bpe.ro test.ro
rm wmt16.tar.gz
rm -r wmt16