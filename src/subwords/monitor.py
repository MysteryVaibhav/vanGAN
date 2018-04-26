#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch.autograd import Variable
from torch.nn import functional as F
from os import path
import torch

from util import read_validation_file
from util import drop_oov_from_validation_set
from util import pad

class Monitor():
    """Monitor a training progress by cosine similarities of correct translations. (debug purpose)"""
    def __init__(self, params, src_data, tgt_data):
        "docstring"
        self.data_dir = params.data_dir

        self.src_emb = src_data['E']
        self.src_indexer = src_data['id2idx']
        self.tgt_emb = tgt_data['E']
        self.tgt_indexer = tgt_data['word2idx']
        self.setup_validation_data(params.validation_file)

    def setup_validation_data(self, filename):
        """Read a validation set from a file."""
        src_seqs, tgt_indices = read_validation_file(
            path.join(self.data_dir, filename), self.src_indexer, self.tgt_indexer)
        src_n_vocab, tgt_n_vocab = self.src_emb.size()[0], self.tgt_emb.size()[0]
        src_seqs, tgt_indices = drop_oov_from_validation_set(
            src_seqs, tgt_indices, src_n_vocab, tgt_n_vocab)
        self.src_seqs = torch.LongTensor(pad(src_seqs))
        self.tgt_indices = torch.LongTensor(tgt_indices)
        if torch.cuda.is_available():
            self.src_seqs = self.src_seqs.cuda()
            self.tgt_indices = self.tgt_indices.cuda()

    def cosine_similarity(self, g, src_data, tgt_data):
        """Calculate cosine similarities."""
        src_vecs = src_data['F'](Variable(self.src_seqs), src_data['E'])
        tgt_vecs = tgt_data['E'](Variable(self.tgt_indices))
        sims = F.cosine_similarity(g(src_vecs), tgt_vecs).data
        if sims.is_cuda:
            sims = sims.cpu()
        return sims.numpy()
