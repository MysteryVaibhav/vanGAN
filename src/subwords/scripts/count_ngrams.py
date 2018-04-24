#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
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

    ngrams = set()
    with open(args.path_input) as f:
        for i, line in enumerate(f, start=1):
            if i == args.topn:
                break
            for idx in line.strip().split()[1:]:
                ngrams.add(idx)
    print(len(ngrams), args.topn)
    return 0


if __name__ == '__main__':
    logger = init_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument('path_input', help='path to input file')
    parser.add_argument('--topn', type=int, default=10000)
    parser.add_argument('-v', '--verbose',
                        action='store_true', default=False,
                        help='verbose output')
    args = parser.parse_args()
    main(args)
