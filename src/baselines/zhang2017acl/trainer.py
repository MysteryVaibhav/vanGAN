from os import path
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
from tqdm import trange
import json
import numpy as np
import torch
import torch.utils.data

from model import Discriminator
from model import Generator
from properties import *
from timeit import default_timer as timer
from util import WeightedSampler
from util import get_frequencies
from util import init_logger
from util import downsample_frequent_words


class Trainer:
    def __init__(self, params):
        self.logger = init_logger('Trainer')
        self.params = params

    def train(self, src_emb, tgt_emb, evaluator, **kwargs):
        params = self.params
        # Load data
        if not path.exists(params.data_dir):
            raise "Data path doesn't exists: %s" % params.data_dir

        gan_model = kwargs.get('gan_model', 3)
        dir_model = params.model_dir

        # Embeddings
        src = src_emb
        tgt = tgt_emb

        # Define samplers
        vocab_size = params.most_frequent_sampling_size
        try:
            if params.uniform_sampling:
                raise FileNotFoundError
            weights_src, weights_tgt = get_frequencies()
            self.logger.info('Use frequencies for sampling')
            weights_src = downsample_frequent_words(weights_src)
            weights_tgt = downsample_frequent_words(weights_tgt)
            weights_src[vocab_size:] = 0.0
            weights_tgt[vocab_size:] = 0.0
            weights_src /= weights_src.sum()
            weights_tgt /= weights_tgt.sum()
        except FileNotFoundError:
            weights_src = np.ones(vocab_size) / vocab_size
            weights_tgt = np.ones(vocab_size) / vocab_size

        sampler_src = WeightedSampler(weights_src)
        sampler_tgt = WeightedSampler(weights_tgt)
        iter_src = sampler_src.get_iterator(mini_batch_size)
        iter_tgt = sampler_tgt.get_iterator(mini_batch_size)

        # Create models
        g = Generator(input_size=params.g_input_size,
                      output_size=params.g_output_size)
        d_tgt = Discriminator(  # Discriminator (source-side)
            input_size=params.g_output_size, hidden_size=params.d_hidden_size)
        d_src = None  # Discriminator (target-side)
        lambda_r = 0.0  # Coefficient of reconstruction loss
        if params.gan_model == 1:  # Model 1: Undirectional Transformation
            self.logger.info('Model 1')
        elif gan_model == 2:  # Model 2: Bidirectional Transformation
            d_src = Discriminator(  # Discriminator (source-side)
                input_size=g_input_size, hidden_size=d_hidden_size)
            self.logger.info('Model 2')
        else:  # Model 3: Adversarial Autoencoder
            lambda_r = params.lambda_r
            self.logger.info('Model 3: lambda = {}'.format(lambda_r))

        # Define loss function and optimizers
        g_optimizer = optim.Adam(g.parameters(), lr=g_learning_rate)
        d_tgt_optimizer = optim.Adam(d_tgt.parameters(), lr=params.d_learning_rate)
        if d_src is not None:
            d_src_optimizer = optim.Adam(d_src.parameters(), lr=d_learning_rate)

        if torch.cuda.is_available:
            # Move the network and the optimizer to the GPU
            g = g.cuda()
            d_tgt = d_tgt.cuda()
            if d_src is not None:
                d_src = d_src.cuda()

        lowest_loss = 10000000  # lowest loss value (standard for saving checkpoint)

        for itr, (batch_src, batch_tgt) in enumerate(zip(iter_src, iter_tgt), start=1):
            if src.weight.is_cuda:
                batch_src = batch_src.cuda()
                batch_tgt = batch_tgt.cuda()
            embs_src = src(batch_src)
            embs_tgt = tgt(batch_tgt)

            # Generator
            embs_tgt_mapped = g(embs_src)  # target embs mapped from source embs
            g_loss = -(d_tgt(embs_tgt_mapped, inject_noise=False) + 1e-16).log().mean()  # discriminate in the trg side
            if d_src is not None:  # Model 2
                embs_src_mapped = g(embs_tgt, tgt2src=True)  # src embs mapped from trg embs
                g_loss += -(d_src(embs_src_mapped, inject_noise=False) + 1e-16).log().mean()  # target-to-source

            if lambda_r > 0:  # Model 3
                embs_src_r = g(embs_tgt_mapped, tgt2src=True)  # reconstructed src embs
                g_loss_r = 1.0 - F.cosine_similarity(embs_src, embs_src_r).mean()
                g_loss += lambda_r * g_loss_r

            ## Update
            g_optimizer.zero_grad()  # reset the gradients
            g_loss.backward()
            g_grad_norm = float(g.W._grad.norm(2, dim=1).mean())
            g_optimizer.step()

            # Discriminator (target-side)
            preds = d_tgt(embs_tgt)
            hits = int(sum(preds >= 0.5))
            d_tgt_loss = -(preds + 1e-16).log().sum()  # -log(D(Y))
            preds = d_tgt(embs_tgt_mapped.detach())
            hits += int(sum(preds < 0.5))
            d_tgt_loss += -(1.0 - preds + 1e-16).log().sum()  # -log(1 - D(G(X)))
            d_tgt_loss /= embs_tgt.size(0) + embs_tgt_mapped.size(0)
            d_tgt_acc = hits / float(embs_tgt.size(0) + embs_tgt_mapped.size(0))

            ## Update
            d_tgt_optimizer.zero_grad()
            d_tgt_loss.backward()
            d_tgt_optimizer.step()

            # Discriminator (source-side)
            if d_src is not None:
                preds = d_src(embs_src)
                hits = int(sum(preds >= 0.5))
                d_src_loss = -(preds + 1e-16).log().sum()  # -log(D(X))
                preds = d_src(embs_src_mapped.detach())
                hits = int(sum(preds < 0.5))
                d_src_loss += -(1.0 - preds + 1e-16).log().sum()  # -log(1 - D(G(Y)))
                d_src_loss /= embs_src.size(0) + embs_src_mapped.size(0)
                d_src_acc = hits / float(embs_src.size(0) + embs_src_mapped.size(0))

                ## Update
                d_src_optimizer.zero_grad()
                d_src_loss.backward()
                d_src_optimizer.step()

            if itr % 100 == 0:
                itr_template = '[{:' + str(int(np.log10(max_iters)) + 1) + 'd}]'
                status = [itr_template.format(itr)]
                status.append('{:.2f}'.format(d_tgt_acc * 100))
                status.append('{:.3f}'.format(float(d_tgt_loss.data)))
                if d_src is not None:
                    status.append('{:.3f}'.format(d_src_acc))
                    status.append('{:.3f}'.format(float(d_src_loss.data)))
                status.append('{:.3f}'.format(float(g_loss.data)))
                if lambda_r > 0:
                    status.append('{:.3f}'.format(float(g_loss_r.data)))
                status.append('{:.3f}'.format(g_grad_norm))
                WW = g.W.t().matmul(g.W)
                if WW.is_cuda:
                    WW = WW.cpu()
                orth_score = np.linalg.norm(WW.data.numpy() - np.identity(WW.size(0)))
                status.append('{:.2f}'.format(orth_score))
                self.logger.info(' '.join(status))

                if itr % 1000 == 0:
                    filename = path.join(dir_model, 'g{}_checkpoint.pth'.format(gan_model))
                    self.logger.info('Save a model to ' + filename)
                    g.save(filename)
                    all_precisions = evaluator.get_all_precisions(g(src_emb.weight).data)
                    print(json.dumps(all_precisions))
            if itr > max_iters:
                break

            if dir_model is None:
                continue

            # Save checkpoint
            if itr > 10000 and lowest_loss > float(g_loss.data):
                filename = path.join(dir_model, 'g{}_best.pth'.format(gan_model))
                self.logger.info('Save a model to ' + filename)
                lowest_loss = float(g_loss.data)
                g.save(filename)
                all_precisions = evaluator.get_all_precisions(g(src_emb.weight).data)
                print(json.dumps(all_precisions))
        filename = path.join(dir_model, 'g{}_final.pth'.format(gan_model))
        self.logger.info('Save a model to ' + filename)
        g.save(filename)
        return g
