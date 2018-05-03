#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import nn
import torch


PAD = 0

class SubwordEmbedding(nn.Module):
    def __init__(self, dim_emb, dim_h=64, n_layers=1, dropout=0.1):
        """Convert a numpy array into an embedding layer."""
        super(SubwordEmbedding, self).__init__()
        layers = [nn.Dropout(dropout)]
        for i in range(n_layers + 1):
            dim_in = dim_emb if i == 0 else dim_h
            dim_out = dim_emb if i == n_layers else dim_h
            layers.append(nn.Linear(dim_in, dim_out, bias=False))
            if i < n_layers:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        self.mapping = nn.Sequential(*layers)
        for i in range(1, len(self.mapping), 3):
            nn.init.eye_(self.mapping[i].weight.data)
        self.eval()  # testing mode by default

    def forward(self, idx_seqs, emb, mean=True, transform=True):
        """Return word vectors corresponding to sequences of subword IDs."""
        vecs = emb(idx_seqs).detach()
        if transform:
            vecs = self.mapping(vecs)
        vecs = vecs.sum(dim=1)
        if mean:  # take average
            vecs /= (idx_seqs != PAD).sum(dim=1).float().view((-1, 1))
        return vecs

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))


class Embedding(nn.Module):
    """Embedding wrapper"""
    def __init__(self, emb_array, requirs_grad=False):
        """Convert a numpy array into an embedding layer."""
        super(Embedding, self).__init__()
        num_embeddings, embedding_dim = emb_array.shape
        self.emb = nn.Embedding(num_embeddings, embedding_dim,
                                padding_idx=PAD, sparse=True)
        self.emb.weight.data.copy_(torch.from_numpy(emb_array).float())
        self.emb.weight.requires_grad=False

    def forward(self, idx_seqs, mean=True):
        """Return word vectors corresponding to sequences of subword IDs."""
        return self.emb(idx_seqs)

    def center(self):
        """Cener embeddings."""
        mean = self.emb.weight.data.mean(0, keepdim=True)
        self.emb.weight.data.sub_(mean.expand_as(self.emb.weight.data))

    def size(self):
        """Return a size of the embedding."""
        return self.emb.weight.size()
