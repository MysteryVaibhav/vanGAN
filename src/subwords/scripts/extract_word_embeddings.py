#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gensim.models.keyedvectors import KeyedVectors
from tqdm import tqdm
import argparse
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

    path_model = args.path_data + '.vec'
    if verbose:
        logger.info('Load a model from ' + path_model)
    with open(path_model) as f:
        shape = [int(v) for v in f.readline().strip().split()]
        if shape[0] > args.topn:
            shape[0] = args.topn
        W = np.empty(shape)
        idx2word = [_ for _ in range(shape[0])]
        for i, line in tqdm(enumerate(f), total=shape[0]):
            if i == shape[0]:
                break
            w, vec = line.split(' ', 1)
            idx2word[i] = w
            W[i] = [float(v) for v in vec.split()]
    if verbose:
        logger.info('Done.')

    W = np.array(W)
    if verbose:
        logger.info('W: {}'.format(W.shape))

    path_output = args.path_data + '.words.npz'
    if verbose:
        logger.info('Write to ' + path_output)
    np.savez(path_output, W=W, idx2word=idx2word)

    return 0


if __name__ == '__main__':
    logger = init_logger('Extract')
    parser = argparse.ArgumentParser()
    parser.add_argument('path_data', help='path to a data file')
    parser.add_argument('-n', '--topn', type=int, default=200000,
                        help='number of words')
    parser.add_argument('-v', '--verbose',
                        action='store_true', default=False,
                        help='verbose output')
    args = parser.parse_args()
    main(args)
