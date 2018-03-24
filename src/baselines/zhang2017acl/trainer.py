from os import path
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
from tqdm import trange
import numpy as np
import torch
import torch.utils.data

from evaluate import get_precision_k
from model import Discriminator
from model import Generator
from properties import *
from timeit import default_timer as timer
from util import WeightedSampler
from util import get_embeddings
from util import get_frequencies
from util import get_true_dict
from util import init_logger
from util import downsample_frequent_words

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
    dir_model = kwargs.get('dir_model', None)
    flag_normalize = kwargs.get('normalize', True)
    flag_xavier = kwargs.get('flag_xavier', False)
    gan_model = kwargs.get('gan_model', 3)
    gpu = kwargs.get('gpu', False)
    logger = kwargs.get('logger', init_logger('Training'))

    # Load data
    logger.info('SRC: {}, TRG: {}'.format(lang_src, lang_trg))
    src, trg = get_embeddings(lang_src=lang_src, lang_trg=lang_trg,
                              normalize=flag_normalize)   # Vocab x Embedding_dimension
    src = torch.FloatTensor(src)
    trg = torch.FloatTensor(trg)

    logger.info('Get true dict')
    true_dict = get_true_dict()

    try:
        weights_src, weights_trg = get_frequencies(lang_src=lang_src, lang_trg=lang_trg)
        logger.info('Use frequencies for sampling')
        weights_src = downsample_frequent_words(weights_src)
        weights_trg = downsample_frequent_words(weights_trg)
        weights_src[most_frequent_sampling_size:] = 0.0
        weights_trg[most_frequent_sampling_size:] = 0.0
        weights_src /= weights_src.sum()
        weights_trg /= weights_trg.sum()
    except FileNotFoundError:
        weights_src = np.ones(most_frequent_sampling_size) / most_frequent_sampling_size
        weights_trg = np.ones(most_frequent_sampling_size) / most_frequent_sampling_size
    sampler_src = WeightedSampler(weights_src)
    sampler_trg = WeightedSampler(weights_trg)
    iter_src = sampler_src.get_iterator(mini_batch_size)
    iter_trg = sampler_trg.get_iterator(mini_batch_size)

    # Create models
    g = Generator(input_size=g_input_size, output_size=g_output_size)
    d_trg = Discriminator(  # Discriminator (source-side)
        input_size=g_output_size, hidden_size=d_hidden_size, output_size=d_output_size)
    d_src = None  # Discriminator (target-side)
    lambda_r = 0.0  # Coefficient of reconstruction loss
    if gan_model == 1:  # Model 1: Undirectional Transformation
        logger.info('Model 1')
    elif gan_model == 2:  # Model 2: Bidirectional Transformation
        d_src = Discriminator(  # Discriminator (source-side)
            input_size=g_input_size, hidden_size=d_hidden_size, output_size=d_output_size)
        logger.info('Model 2')
    else:  # Model 3: Adversarial Autoencoder
        lambda_r = kwargs.get('lambda_r', 1.0)
        logger.info('Model 3: lambda = {}'.format(lambda_r))

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

    best_orth_score = 10000000  # Orthogonality score

    for itr, (batch_src, batch_trg) in enumerate(zip(iter_src, iter_trg), start=1):
        embs_src = Variable(src[batch_src])
        embs_trg = Variable(trg[batch_trg])
        if gpu:
            embs_src = embs_src.cuda()
            embs_trg = embs_trg.cuda()

        # Generator
        embs_trg_mapped = g(embs_src)  # target embs mapped from source embs
        g_loss = -d_trg(embs_trg_mapped).log().mean()  # discriminate in the trg side

        if d_src is not None:  # Model 2
            embs_src_mapped = g(embs_trg, trg2src=True)  # src embs mapped from trg embs
            g_loss += -d_src(embs_src_mapped).log().mean()  # target-to-source

        if lambda_r > 0:  # Model 3
            embs_src_r = g(embs_trg_mapped, trg2src=True)  # reconstructed src embs
            g_loss_r = 1.0 - F.cosine_similarity(embs_src, embs_src_r).mean()
            g_loss += lambda_r * g_loss_r

        ## Update
        g_optimizer.zero_grad()  # reset the gradients
        g_loss.backward()
        g_optimizer.step()

        # Discriminator (target-side)
        preds = d_trg(embs_trg)
        hits = int(sum(preds >= 0.5))
        d_trg_loss = -preds.log().sum()  # -log(D(Y))
        preds = d_trg(embs_trg_mapped.detach())
        hits += int(sum(preds < 0.5))
        d_trg_loss += -(1.0 - preds).log().sum()  # -log(1 - D(G(X)))
        d_trg_loss /= embs_trg.size(0) + embs_trg_mapped.size(0)
        d_trg_acc = hits / float(embs_trg.size(0) + embs_trg_mapped.size(0))

        ## Update
        d_trg_optimizer.zero_grad()
        d_trg_loss.backward()
        d_trg_optimizer.step()

        # Discriminator (source-side)
        if d_src is not None:
            preds = d_src(embs_src)
            hits = int(sum(preds >= 0.5))
            d_src_loss = -preds.log().sum()  # -log(D(X))
            preds = d_src(embs_src_mapped.detach())
            hits = int(sum(preds < 0.5))
            d_src_loss += -(1.0 - preds).log().sum()  # -log(1 - D(G(Y)))
            d_src_loss /= embs_src.size(0) + embs_src_mapped.size(0)
            d_src_acc = hits / float(embs_src.size(0) + embs_src_mapped.size(0))

            ## Update
            d_src_optimizer.zero_grad()
            d_src_loss.backward()
            d_src_optimizer.step()

        if itr % 100 == 0:
            status = ['[{}]'.format(itr)]
            status.append('G: {:.5f}'.format(float(g_loss.data)))
            if lambda_r > 0:
                status.append('G(r): {:.5f}'.format(float(g_loss_r.data)))
            status.append('D(trg): {:.5f} {:.3f}'.format(float(d_trg_loss.data), d_trg_acc))
            if d_src is not None:
                status.append('D(src): {:.5f} {:.3f}'.format(float(d_src_loss.data), d_src_acc))
            WW = g.W.t().matmul(g.W)
            if gpu:
                WW = WW.cpu()
            orth_score = np.linalg.norm(WW.data.numpy() - np.identity(WW.size(0)))
            status.append('Orth.: {:.2f}'.format(orth_score))
            logger.info(' '.join(status))
            if dir_model is not None and best_orth_score > orth_score:
                filename = path.join(dir_model, 'g{}_best.pth'.format(gan_model))
                logger.info('Save a model to ' + filename)
                best_orth_score = orth_score
                g.save(filename)
                for k in [1,]:
                    prec = get_precision_k(k, g, true_dict, method='nn', gpu=gpu)
                    print('P@{} : {}'.format(k, prec))
        if itr % 1000 == 0:
            for k in [1, 5]:
                prec = get_precision_k(k, g, true_dict, method='nn', gpu=gpu)
                print('P@{} : {}'.format(k, prec))

        if itr > max_iters:
            break
    return g
