#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gensim.models.keyedvectors import KeyedVectors
from nltk.corpus import stopwords
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
from tqdm import tqdm
import argparse
import gzip
import logging
import lzma
import numpy as np
import re
import string
import torch

verbose = False
logger = None


def init_logger(name='logger'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    log_fmt = '%(asctime)s/%(name)s[%(levelname)s]: %(message)s'
    logging.basicConfig(format=log_fmt)
    return logger


def read_embeddings(filename):
    """Read word embeddings."""
    if verbose:
        logger.info('Read word embeddings from ' + filename)
    binary = True if filename.endswith('.bin') else False
    return KeyedVectors.load_word2vec_format(filename, binary=binary,
                                             unicode_errors='ignore')

def read_seed_lexicon(filename):
    """Read seed lexicon."""
    lexicon = []
    with open(filename) as f:
        for line in f:
            lexicon.append(tuple(line.strip().split('\t')))
    if verbose:
        logger.info('Read {} pairs from {}'.format(len(lexicon), filename))
    return lexicon


def get_vocabulary(src, trg, embs_src, embs_trg, seed_lexicon=None, n=2000):
    def read_helper(filename, sw):
        if filename.endswith('.xz'):
            f = lzma.open(filename, 'rt')
        elif filename.endswith('.xz'):
            f = gzip.open(filename, 'rt')
        else:
            f = open(filename, 'rt')
        for line in f:
            word, _ = line.strip().split('\t')
            if word in sw:
                continue
            if word in puncts:
                continue
            if r_number.match(word):
                continue
            yield word
        f.close()

    # Words in the seed_lexicon
    if seed_lexicon is None:
        seed_src = set()
        seed_trg = set()
    else:
        seed_src, seed_trg = zip(*seed_lexicon)
        seed_src = set(seed_src)
        seed_trg = set(seed_trg)
    we_vocab_src = set(embs_src.vocab)
    we_vocab_trg = set(embs_trg.vocab)

    languages = {'en': 'english', 'it': 'italian'}
    r_number = re.compile('[\d\.,]+')
    puncts = set(list(string.punctuation))

    # Source
    lang_src, path_src = src.split(':')
    vocab_src = []
    for word in read_helper(
            path_src,
            set(stopwords.words(languages.get(lang_src, lang_src)))):
        if word in seed_src:
            continue
        if word not in we_vocab_src:
            continue
        vocab_src.append(word)
        if len(vocab_src) == n:
            break

    # Target
    lang_trg, path_trg = trg.split(':')
    vocab_trg = []
    for word in read_helper(
            path_trg,
            set(stopwords.words(languages.get(lang_trg, lang_trg)))):
        if word in seed_trg:
            continue
        if word not in we_vocab_trg:
            continue
        vocab_trg.append(word)
        if len(vocab_trg) == n:
            break
    return vocab_src, vocab_trg


def update_vocabulary(vocab, exclude):
    """Update vocabulary."""
    new_vocab = []
    for word in vocab:
        if word in exclude:
            continue
        new_vocab.append(word)
    return new_vocab


class TransMat(nn.Module):
    def __init__(self, dim_emb_src, dim_emb_trg=None, **kwargs):
        super(TransMat, self).__init__()
        self.verbose = kwargs.get('verbose', False)
        self.logger = init_logger('TransMat')

        self.dim_src = dim_emb_src
        self.dim_trg = dim_emb_src if dim_emb_trg is None else dim_emb_trg
        self.src2trg = nn.Linear(self.dim_src, self.dim_trg, bias=False)
        self.trg2src = nn.Linear(self.dim_trg, self.dim_src, bias=False)

    def forward(self, src):
        return self.src2trg(src)

    def forward_src2trg(self, src):
        return self.forward(src)

    def forward_trg2src(self, trg):
        return self.trg2src(trg)

    def eval_lexicon(self, lexicon, reduce=True):
        """Evaluate translation pairs in a given lexicon."""
        loss = 0
        for emb_src, emb_trg in lexicon:
            diff_1 = emb_trg - self.forward_src2trg(emb_src)
            loss += diff_1.pow(2).sum()
            diff_2 = emb_src - self.forward_trg2src(emb_trg)
            loss += diff_2.pow(2).sum()
        if reduce:
            loss /= len(lexicon)
        return loss

    def bootstrap(self, vocab_src, vocab_trg, embs_src, embs_trg, k=25,
                  batch_size=50):
        def make_batch(words, embs, bs):
            batch = []
            for word in words:
                batch.append((word, embs[word].tolist()))
                if len(batch) >= bs:
                    w, e = zip(*batch)
                    batch = []
                    yield w, Variable(torch.FloatTensor(e))
            if len(batch) > 0:
                w, e = zip(*batch)
                yield w, Variable(torch.FloatTensor(e))

        scores = []
        for wsrc, esrc in tqdm(make_batch(vocab_src, embs_src, bs=1),
                               total=len(vocab_src)):
            scores_memo = []
            for wtrg, etrg in make_batch(vocab_trg, embs_trg, bs=batch_size):
                vals = F.cosine_similarity(
                    self.forward_src2trg(esrc[0]).view((1, -1)), etrg)
                vals += F.cosine_similarity(
                    self.forward_src2trg(etrg[0]).view((1, -1)), esrc)
                scores_memo += [
                    (wsrc[0], esrc[0], w, e, v)
                    for w, e, v in sorted(zip(wtrg, etrg, vals.data.numpy()),
                                              key=lambda t: -t[-1])]
            scores += sorted(scores_memo, key=lambda t: -t[-1])[:k]
            # 10it/sec
        return sorted(scores, key=lambda t: -t[-1])[:k]


def main(args):
    global verbose
    verbose = args.verbose

    embs_src = read_embeddings(args.path_embeddings_src)
    embs_trg = read_embeddings(args.path_embeddings_trg)
    seed_lexicon = read_seed_lexicon(args.path_seed_lexicon)

    lexicon = []
    for i, (word_src, word_trg) in enumerate(seed_lexicon):
        try:
            emb_src = embs_src[word_src]
            emb_trg = embs_trg[word_trg]
        except KeyError:
            continue
        emb_src = Variable(torch.FloatTensor(emb_src), requires_grad=False)
        emb_trg = Variable(torch.FloatTensor(emb_trg), requires_grad=False)
        lexicon.append((emb_src, emb_trg))
    if verbose:
        logger.info('Seed: {}'.format(len(lexicon)))

    vocab_src, vocab_trg = get_vocabulary(
        args.path_vocab_src, args.path_vocab_trg,
        embs_src=embs_src, embs_trg=embs_trg, seed_lexicon=seed_lexicon)

    model = TransMat(embs_src.vector_size,
                     trg=embs_trg.vector_size,
                     verbose=args.verbose)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    TOL = 1e-2
    for it in range(args.n_iters):
        # Training
        loss_prev, loss_cur = None, None
        epoch = 1
        while True:
            model.zero_grad()
            loss = model.eval_lexicon(lexicon)
            loss_prev, loss_cur = loss_cur, float(loss.data.numpy())
            try:
                # If loss_prev is None, exception will be thrown.
                diff = abs((loss_prev - loss_cur) / loss_prev)
                if diff <= TOL:
                    break
            except TypeError:
                pass
            if verbose and epoch % 10 == 0:
                logger.info('[{}] loss: {:.5f}'.format(epoch, loss_cur))
            loss.backward()
            optimizer.step()
            epoch += 1
        if verbose:
            logger.info('[{}] loss: {:.5f}'.format(epoch, loss_cur))
        # Bootstrapping
        data = model.bootstrap(vocab_src, vocab_trg, embs_src, embs_trg)
        wsrc, esrc, wtrg, etrg, scores = zip(*data)
        if verbose:
            for w1, w2, s in zip(wsrc, wtrg, scores):
                if verbose:
                    logger.info('{}-{} {:.3f}'.format(w1, w2, s))
        lexicon += [(e1, e2) for e1, e2 in zip(esrc, etrg)]
        vocab_src = update_vocabulary(vocab_src, set(wsrc))
        vocab_trg = update_vocabulary(vocab_trg, set(wtrg))

    if verbose:
        logger.info('Save a model to ' + args.path_output)
    torch.save(model.state_dict(), args.path_output)

    return 0


if __name__ == '__main__':
    logger = init_logger('Trans')
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', dest='path_embeddings_src',
                        required=True,
                        help='path to embeddings of source language')
    parser.add_argument('--trg', dest='path_embeddings_trg',
                        required=True,
                        help='path to embeddings of target language')
    parser.add_argument('--vocab-src', dest='path_vocab_src',
                        default='en:data/ukWaC/vocab.txt',
                        help='path to source vocabulary')
    parser.add_argument('--vocab-trg', dest='path_vocab_trg',
                        default='it:data/itWaC/vocab.txt',
                        help='path to target vocabulary')
    parser.add_argument('--lex', dest='path_seed_lexicon',
                        help='path to seed lexicon')
    parser.add_argument('--iter', dest='n_iters',
                        type=int, default=30,
                        help='number of bootstrapping iterations')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='number of bootstrapping iterations')
    parser.add_argument('-o', '--output', dest='path_output',
                        help='path to output file')
    parser.add_argument('-v', '--verbose',
                        action='store_true', default=False,
                        help='verbose output')
    args = parser.parse_args()
    main(args)
