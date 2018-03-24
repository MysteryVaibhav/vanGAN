from torch import optim
import numpy as np
import sys
import torch
import torch.utils.data
from tqdm import trange
from torch.autograd import Variable
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.sampler import WeightedRandomSampler

from model import Generator
from model import Discriminator
from properties import *
from timeit import default_timer as timer
from util import init_logger
from util import get_embeddings

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
#from train_validate import get_precision_k


def init_xavier(m):
    if type(m) == torch.nn.Linear:
        fan_in = m.weight.size()[1]
        fan_out = m.weight.size()[0]
        std = np.sqrt(6.0 / (fan_in + fan_out))
        m.weight.data.normal_(0,std)


def train(**kwargs):
    gpu = kwargs.get('gpu', False)
    flag_xavier = kwargs.get('flag_xavier', False)
    gan_model = kwargs.get('gan_model', 0)
    logger = kwargs.get('logger', init_logger('Training'))

    # Load data
    src, trg = get_embeddings()   # Vocab x Embedding_dimension
    src = torch.FloatTensor(src)
    trg = torch.FloatTensor(trg)

    # TODO: reflect word frequencies?
    weights = np.ones(most_frequent_sampling_size) / most_frequent_sampling_size
    src_sampler = BatchSampler(WeightedRandomSampler(weights, iters_in_epoch),
                              mini_batch_size, drop_last=True)
    trg_sampler = BatchSampler(WeightedRandomSampler(weights, iters_in_epoch),
                              mini_batch_size, drop_last=True)

    # Create models
    g = Generator(input_size=g_input_size, output_size=g_output_size)
    d_trg = Discriminator(  # Descriminator (source-side)
        input_size=d_input_size, hidden_size=d_hidden_size, output_size=d_output_size)
    d_src = None  # Descriminator (target-side)
    if gan_model == 0:  # Model 1: Undirectional Transformation
        pass
    elif gan_model == 2:  # Model 2: Bidirectional Transformation
        d_src = Discriminator(  # Descriminator (source-side)
            input_size=g_output_size, hidden_size=d_hidden_size, output_size=d_output_size)
    else:  # Model 3: Adversarial Autoencoder
        pass

    if flag_xavier:
        init_xavier(d_trg)
        if d_src is not None:
            init_xavier(d_src)

    # Define loss function and optimizers
    d_trg_optimizer = optim.Adam(d_trg.parameters(), lr=d_learning_rate)
    if d_src is not None:
        d_src_optimizer = optim.Adam(d_src.parameters(), lr=d_learning_rate)
    g_optimizer = optim.Adam(g.parameters(), lr=g_learning_rate)

    if gpu:
        # Move the network and the optimizer to the GPU
        g = g.cuda()
        d_trg = d_trg.cuda()
        if d_src is not None:
            d_src = d_src.cuda()

    for epoch in range(num_epochs):
        for itr, (batch_src, batch_trg) in enumerate(zip(src_sampler, trg_sampler), start=1):
            embs_src = Variable(src[batch_src])
            embs_trg = Variable(trg[batch_trg])
            if gpu:
                embs_src = embs_src.cuda()
                embs_trg = embs_trg.cuda()

            # Generator
            g_optimizer.zero_grad()  # Reset the gradients
            embs_trg_mapped = g(embs_src)  # Target embs mapped from source embs
            g_loss = d_trg(embs_src).log().neg().mean()  # source-to-target
            if d_src is not None:
                embs_src_mapped = g(embs_trg, src2trg=False)  # Src embs mapped from trg embs
                g_loss += d_src(embs_src_mapped).log().neg().mean()  # target-to-source
            g_loss.backward()
            g_optimizer.step()

            # Descriminator (target-side)
            d_trg_optimizer.zero_grad()
            preds = d_trg(embs_trg)
            hits = int(sum(preds >= 0.5))
            d_trg_loss = preds.log().neg().mean()  # -log(D(Y))
            preds = (1 - d_trg(embs_trg_mapped.detach()))
            hits += int(sum(preds >= 0.5))
            d_trg_loss += preds.log().neg().mean()  # -log(1 - D(G(X)))
            d_trg_loss.backward()
            d_trg_optimizer.step()
            d_trg_acc = hits / float(embs_trg.size(0) + embs_trg_mapped.size(0))

            # Descriminator (source-side)
            if d_src is not None:
                d_src_optimizer.zero_grad()
                d_src_loss = d_src(embs_src).log().neg().mean()  # -log(D(Y))
                d_src_loss += (1 - d_src(embs_src_mapped.detach())).log().neg().mean()  # -log(1 - D(G(X)))
                d_src_loss.backward()
                d_src_optimizer.step()

            if itr % 50 == 0:
                status = ['[{}/{}]'.format(itr, epoch)]
                status.append('G: {:.5f}'.format(float(g_loss.data)))
                status.append('D(trg): {:.5f} {:.3f}'.format(float(d_trg_loss.data), d_trg_acc))
                if d_src is not None:
                    status.append('D(src): {:.5f}'.format(float(d_src_loss.data)))
                logger.info(' '.join(status))
    return g


def orthogonalize(W):
    W.copy_((1 + beta) * W - beta * W.mm(W.transpose(0, 1).mm(W)))

    
def clip(d, clip):
    if clip > 0:
        for x in d.parameters():
            x.data.clamp_(-clip, clip)


if __name__ == '__main__':
    generator = train()
