#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict
from main import TransMat
from main import read_embeddings
from torch.autograd import Variable
from tqdm import tqdm
import argparse
import logging
import numpy as np
import torch

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

    embs_src = read_embeddings(args.path_embeddings_src)
    embs_trg = read_embeddings(args.path_embeddings_trg)

    if verbose:
        logger.info('Load a model from ' + args.path_model)
    model = TransMat(embs_src.vector_size,
                     trg=embs_trg.vector_size,
                     verbose=args.verbose)
    model.load_state_dict(torch.load(args.path_model))

    if verbose:
        logger.info('k = {}'.format(args.topk))
        logger.info('Eval = ' + args.path_test)
    golds = defaultdict(set)
    with open(args.path_test) as f:
        for line in f:
            src, trg = line.strip().split()
            golds[src].add(trg)

    precisions = []
    oov = 0
    for src, trgs in tqdm(golds.items(), total=len(golds)):
        try:
            vec = Variable(torch.from_numpy(embs_src[src]),
                           requires_grad=False)
        except KeyError:
            oov += 1
            precisions.append(0.0)
        trg_preds = embs_trg.most_similar(
            positive=[model.src2trg(vec).data.numpy()], topn=args.topk)
        print(', '.join(trgs) + ': ' + ', '.join(w for w, _ in trg_preds))
        correct = 0
        for trg_pred, _ in trg_preds:
            if trg_pred in trgs:
                correct += 1
        # precisions.append(correct / float(args.topk))  # avg. precision@k (from the original definition)
        precisions.append(min(correct, 1))  # avg. precision@k in this field
    print(np.mean(precisions))
    print('OOV: {}'.format(oov))

    return 0


if __name__ == '__main__':
    logger = init_logger('Eval')
    parser = argparse.ArgumentParser()
    parser.add_argument('path_test', help='path to test file')
    parser.add_argument('--src', dest='path_embeddings_src',
                        required=True,
                        help='path to embeddings of source language')
    parser.add_argument('--trg', dest='path_embeddings_trg',
                        required=True,
                        help='path to embeddings of target language')
    parser.add_argument('-k', '--topk', type=int, default=1,
                        help='number of predictions to be considered')
    parser.add_argument('-m', '--model', dest='path_model',
                        help='path to a model file')
    parser.add_argument('-v', '--verbose',
                        action='store_true', default=False,
                        help='verbose output')
    args = parser.parse_args()
    main(args)
