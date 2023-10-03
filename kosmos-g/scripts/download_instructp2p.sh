#!/bin/bash

# Make data folder relative to script location
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

mkdir -p $SCRIPT_DIR/../data

# Copy text datasets
wget -q --show-progress http://instruct-pix2pix.eecs.berkeley.edu/gpt-generated-prompts.jsonl -O $SCRIPT_DIR/../data/gpt-generated-prompts.jsonl
wget -q --show-progress http://instruct-pix2pix.eecs.berkeley.edu/human-written-prompts.jsonl -O $SCRIPT_DIR/../data/human-written-prompts.jsonl

# If dataset name isn't provided, exit.
if [ -z $1 ]
then
	exit 0
fi

# Copy dataset files
mkdir $SCRIPT_DIR/../data/$1
wget -A zip,json -R "index.html*" -q --show-progress -r --no-parent http://instruct-pix2pix.eecs.berkeley.edu/$1/ -nd -P $SCRIPT_DIR/../data/$1/

# Unzip to folders
unzip $SCRIPT_DIR/../data/$1/\*.zip -d $SCRIPT_DIR/../data/$1/

# Cleanup
rm -f $SCRIPT_DIR/../data/$1/*.zip
rm -f $SCRIPT_DIR/../data/$1/*.html