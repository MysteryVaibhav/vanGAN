#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
import argparse
import logging
from tqdm import tqdm

from util import pad
from embeddings import SubwordEmbedding

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

    # Read subword sequences and embeddings
    if verbose:
        logger.info('Load subwords from ' + args.path_subwords)
    data = dict(np.load(args.path_subwords))
    N = data['seqs'].shape[0]
    D = data['W'].shape[1]
    W = np.r_[np.zeros((1, D)), data['W']]
    seqs = np.array([[v + 1 for v in seq] for seq in data['seqs']])  # 0=PAD

    # Load a trained transformer
    if verbose:
        logger.info('Load a transformer from ' + args.path_transformer)
    transformer = SubwordEmbedding(D, n_layers=args.n_layers)
    transformer.load(args.path_transformer)

    # Vocabularies (words)
    vocab = []
    with open(args.path_original) as f:
        next(f) # skip a header
        for i, line in enumerate(f):
            if i == N:  # reached the vocabulary size
                break
            vocab.append(line.split(' ', 1)[0])

    batch_size = args.batch_size
    n_batches = N // batch_size + int(N % batch_size > 0)  # number of batches
    batches = [np.arange(n * batch_size, min(N, (n + 1) * batch_size)) for n in range(n_batches)]


    # Transform subword embeddings and aggregate them to word embeddings
    def embedding_layer(idx_seqs):  # Imitating embeddings.Embedding class
        return torch.FloatTensor([[W[idx] for idx in idx_seq] for idx_seq in idx_seqs])

    if verbose:
        logger.info('Converting...')
    vecs = []
    for batch in tqdm(batches, total=n_batches):
        batch_seqs = torch.LongTensor(pad(seqs[batch]))
        batch_seqs.requires_grad = False
        vecs += [vec.detach().numpy() for vec in transformer(batch_seqs, embedding_layer)]

    # Output to file in text word2vec format
    if verbose:
        logger.info('Output to ' + args.path_output)
    with open(args.path_output, 'w') as f:
        f.write('{N} {D}\n'.format(N=N, D=D))
        for word, vec in zip(vocab, vecs):
            f.write('{word} {vec}\n'.format(
                word=word, vec=' '.join(str(v) for v in vec)))

    return 0


if __name__ == '__main__':
    logger = init_logger('Output')
    parser = argparse.ArgumentParser()
    parser.add_argument('--subwords', dest='path_subwords',
                        required=True, help='path to a subwords file (.npz)')
    parser.add_argument('--transformer', dest='path_transformer',
                        required=True, help='path to a transformer file (.pth)')
    parser.add_argument('--original', dest='path_original',
                        required=True, help='path to an original vector file (.vec)')
    parser.add_argument('-o', '--output', dest='path_output',
                        required=True, help='path to output file')
    parser.add_argument('--n-layers', type=int, default=0,
                        help='number of hidden layers of a transformer')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='batch size')
    parser.add_argument('-v', '--verbose',
                        action='store_true', default=False,
                        help='verbose output')
    args = parser.parse_args()
    main(args)
