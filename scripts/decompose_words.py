#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Decompose words into subwordIDs."""

from os import path
from tqdm import tqdm
import argparse
import fastText
import logging

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

    if verbose:
        logger.info('Loading ' + args.path_data + '.bin')
    model = fastText.load_model(args.path_data + '.bin')

    if verbose:
        logger.info('Outpu to ' + args.path_data + '.subwords')
    of = open(args.path_data + '.subwords', 'w')
    with open(args.path_data + '.vec') as f:
        next(f)
        for line in tqdm(f):
            word = line.split(' ', 1)[0]
            _, indices = model.get_subwords(word)
            of.write('{} {}\n'.format(
                word, ' ' .join(str(i) for i in indices)))
    of.close()

    return 0


if __name__ == '__main__':
    logger = init_logger('Decompose')
    parser = argparse.ArgumentParser()
    parser.add_argument('path_data', help='path to data file')
    parser.add_argument('-v', '--verbose',
                        action='store_true', default=False,
                        help='verbose output')
    args = parser.parse_args()
    main(args)
