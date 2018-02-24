#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import lzma
import gzip
import re
from nltk.corpus import stopwords
import editdistance  # pip install editdistance

verbose = False
logger = None

languages = {'en': 'english', 'it': 'italian'}


def init_logger(name='logger'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    log_fmt = '%(asctime)s/%(name)s[%(levelname)s]: %(message)s'
    logging.basicConfig(format=log_fmt)
    return logger


def read_vocab(filename, lang='english', m=None, p=None):
    """Read ranked vocabulary.

    Input file format:
    <word>\t<frequency>
    """
    sw = stopwords.words(languages.get(lang, lang))
    sw = set(sw)
    r_number = re.compile('[\d\.,]+')
    if filename.endswith('.xz'):
        f = lzma.open(filename, 'rt')
    elif filename.endswith('.xz'):
        f = gzip.open(filename, 'rt')
    else:
        f = open(filename, 'rt')
    vocab = []
    for line in f:
        word, _ = line.strip().split('\t')
        if word in sw:
            continue
        if r_number.match(word):
            continue
        vocab.append(word)
        if m is None or p is None:
            continue
        if len(vocab) > m + p:
            break
    if verbose:
        logger.info('Read {} words from {}'.format(len(vocab), filename))
    f.close()
    return vocab


def extract_seed(vocab_src, vocab_trg, m, p, d):
    """Extract seed lexicon."""
    lexicon = []
    for i, word_src in enumerate(vocab_src):
        start, end = i - p, i + p  # constraint 1
        for word_trg in vocab_trg[start:end+1]:
            if word_src == word_trg:  # constraint 3
                continue
            ned = editdistance.eval(word_src, word_trg) \
                  / max(len(word_src), len(word_trg))
            if ned > d:  # constraint 2
                continue
            lexicon.append((word_src, word_trg))
            if verbose:
                logger.info('Append: {}'.format(lexicon[-1]))
    if verbose:
        logger.info('Extracted {} pairs'.format(len(lexicon)))
    return lexicon


def write_lexicon(filename, lexicon):
    """Write seed lexicon to file."""
    if verbose:
        logger.info('Write to ' + filename)
    with open(filename, 'w') as f:
        for word_src, word_trg in lexicon:
            f.write('{}\t{}\n'.format(word_src, word_trg))


def main(args):
    global verbose
    verbose = args.verbose

    lang, filename = args.path_src.split(':')
    vocab_src = read_vocab(filename, lang=lang, m=args.m, p=args.p)
    lang, filename = args.path_trg.split(':')
    vocab_trg = read_vocab(filename, lang=lang, m=args.m, p=args.p)

    if verbose:
        logger.info('Extract top {}'.format(args.m))
    vocab_src = vocab_src[:args.m + args.p]
    vocab_trg = vocab_trg[:args.m + args.p]

    lexicon = extract_seed(vocab_src, vocab_trg, m=args.m, p=args.p, d=args.d)

    write_lexicon(args.path_output, lexicon)

    return 0


if __name__ == '__main__':
    logger = init_logger('Seed')
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', dest='path_src', required=True,
                        help='path to a source ranked vocabulary file')
    parser.add_argument('--trg', dest='path_trg', required=True,
                        help='path to a source ranked vocabulary file')
    parser.add_argument('-m', type=int, default=10000,
                        help='number of candidates')
    parser.add_argument('-p', type=int, default=100,
                        help='max rank difference')
    parser.add_argument('-d', type=float, default=0.25,
                        help='max edit distance')
    parser.add_argument('-o', '--output', dest='path_output',
                        required=True, help='path to output file')
    parser.add_argument('-v', '--verbose',
                        action='store_true', default=False,
                        help='verbose output')
    args = parser.parse_args()
    main(args)
