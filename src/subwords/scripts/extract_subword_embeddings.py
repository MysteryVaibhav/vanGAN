#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tqdm import tqdm
import argparse
import fastText
import logging
import numpy as np

verbose = False
logger = None


def init_logger(name='logger'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    log_fmt = '%(asctime)s/%(name)s[%(levelname)s]: %(message)s'
    logging.basicConfig(format=log_fmt)
    return logger


def main(args):
    global verbose
    verbose = args.verbose

    path_model = args.path_data + '.bin'
    path_subwords = args.path_data + '.subwords'

    # Read subword IDs
    if verbose:
        logger.info('Read subwords from ' + path_subwords)
    subword_seqs = []
    vocab = set()
    with open(path_subwords) as f:
        for i, line in tqdm(enumerate(f, start=1)):
            subword_seqs.append([int(v) for v in line.strip().split(' ')[1:]])
            for v in subword_seqs[-1]:
                vocab.add(v)
            if i == args.topn:
                break
    idx2id = sorted(list(vocab))
    id2idx = {i: idx for idx, i in enumerate(idx2id)}  # reverse
    # Re-index
    for i, seq in enumerate(subword_seqs):
        subword_seqs[i] = [id2idx[v] for v in seq]
    if verbose:
        logger.info('# of subwords: {}'.format(len(idx2id) - 1))

    if verbose:
        logger.info('Load a model from ' + path_model)
    model = fastText.load_model(path_model)
    if verbose:
        logger.info('Done.')

    W = np.r_[[model.get_input_vector(i) for i in idx2id]]
    if verbose:
        logger.info('W: {}'.format(W.shape))

    path_output = args.path_data + '.subwords.topn{}.npz'.format(args.topn)
    if verbose:
        logger.info('Write to ' + path_output)
    np.savez(path_output, W=W, seqs=subword_seqs, idx2id=idx2id)

    return 0


if __name__ == '__main__':
    logger = init_logger('Extract')
    parser = argparse.ArgumentParser()
    parser.add_argument('path_data', help='path to a data file')
    parser.add_argument('-n', '--topn', type=int, default=10000,
                        help='number of words')
    parser.add_argument('-v', '--verbose',
                        action='store_true', default=False,
                        help='verbose output')
    args = parser.parse_args()
    main(args)
