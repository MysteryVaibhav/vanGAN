#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
    vocab = set()
    with open(path_subwords) as f:
        for i, line in enumerate(f, start=1):
            for wid in line.strip().split(' ')[1:]:
                vocab.add(int(wid))
            if i == args.topn:
                break
    idx2id = sorted(list(vocab))  # index (starting from zero) to word ID
    id2idx = {i: idx for idx, i in enumerate(idx2id)}  # reverse
    if verbose:
        logger.info('# of subwords: {}'.format(len(idx2id)))

    if verbose:
        logger.info('Load a model from ' + path_model)
    model = fastText.load_model(path_model)
    if verbose:
        logger.info('Done.')

    W = np.r_[[model.get_input_vector(i) for i in idx2id]]

    path_output = args.path_data + '.subwords.npz'
    if verbose:
        logger.info('Write to ' + path_output)
    np.savez(path_output, W=W, idx2id=idx2id, id2idx=id2idx)

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
