#!/usr/bin/env python
# -*- coding: utf-8 -*-

from nltk.tokenize import sent_tokenize
from nltk.tokenize.moses import MosesTokenizer
import argparse
import io
import logging
import sys
import tqdm

tokenizer = MosesTokenizer(lang='en')
with io.TextIOWrapper(sys.stdin.buffer, encoding='8859') as sin:
    for line in tqdm.tqdm(sin):
        if line.startswith('CURRENT URL'):
            continue
        for sent in sent_tokenize(line.strip()):
            print(tokenizer.tokenize(sent, return_str=True).lower())
