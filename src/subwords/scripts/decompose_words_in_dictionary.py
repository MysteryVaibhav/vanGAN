#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Decompose words in the eval dictionary into subwordIDs."""
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

    path_dict = path.join(args.dir_data, '{}-{}.5000-6500.txt'.format(
        *args.lang))
    if verbose:
        logger.info('Read ' + path_dict)
    vocab = [set(), set()]
    with open(path_dict) as f:
        for line in f:
            for i, word in enumerate(line.rstrip('\n').split()):
                vocab[i].add(word)

    indices = [{}, {}]
    for i, lang in enumerate(args.lang):
        path_model = path.join(args.dir_data, 'wiki.{}.bin'.format(lang))
        if verbose:
            logger.info('Loading ' + path_model)
        model = fastText.load_model(path_model)
        for word in tqdm(vocab[i], total=len(vocab[i])):
            _, indices[i][word] = model.get_subwords(word)

    path_output = path.join(args.dir_data, '{}-{}.5000-6500.subwords'.format(
        *args.lang))
    if verbose:
        logger.info('Outpu to ' + path_output)
    with open(path_output, 'w') as of:
        with open(path_dict) as f:
            for line in f:
                buff = []
                # word -> indices (space-delimited)
                for i, word in enumerate(line.rstrip('\n').split()):
                    buff.append(' ' .join(str(i) for i in indices[i][word]))
                of.write('\t'.join(buff) + '\n')

    return 0


if __name__ == '__main__':
    logger = init_logger('Decompose')
    parser = argparse.ArgumentParser()
    parser.add_argument('dir_data', help='path to a data directory')
    parser.add_argument('--lang', nargs=2,
                        required=True, help='language pair')
    parser.add_argument('-v', '--verbose',
                        action='store_true', default=False,
                        help='verbose output')
    args = parser.parse_args()
    main(args)
