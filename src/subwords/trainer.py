# from properties import *
from datetime import timedelta
from os import path
from timeit import default_timer as timer
from torch.autograd import Variable
from tqdm import trange  # Run `pip install tqdm`
import copy
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random
import sys
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from evaluator import Evaluator
from model import Generator
from model import Discriminator
from monitor import Monitor
from util import *


class DiscHyperparameters:
    def __init__(self, params):
        self.dropout_inp = params.dropout_inp
        self.dropout_hidden = params.dropout_hidden
        self.leaky_slope = params.leaky_slope
        self.add_noise = params.add_noise
        self.noise_mean = params.noise_mean
        self.noise_var = params.noise_var


class GenHyperparameters:
    def __init__(self, params):
        self.leaky_slope = params.leaky_slope
        self.context = params.context


class Trainer:
    def __init__(self, params):
        self.params = params
        self.knn_emb = None
        self.suffix_str = None
        
    def initialize_exp(self, seed):
        if seed >= 0:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

    def train(self, src_data, tgt_data):
        params = self.params
        print(params)
        penalty = 10.0  # penalty on cosine similarity
        print('Subword penalty {}'.format(penalty))
        # Load data
        if not os.path.exists(params.data_dir):
            raise "Data path doesn't exists: %s" % params.data_dir

        src_lang = params.src_lang
        tgt_lang = params.tgt_lang
        self.suffix_str = src_lang + '_' + tgt_lang

        evaluator = Evaluator(params, src_data=src_data, tgt_data=tgt_data)
        monitor = Monitor(params, src_data=src_data, tgt_data=tgt_data)

        # Initialize subword embedding transformer
        # print('Initializing subword embedding transformer...')
        # src_data['F'].eval()
        # src_optimizer = optim.SGD(src_data['F'].parameters())
        # for _ in trange(128):
        #     indices = np.random.permutation(src_data['seqs'].size(0))
        #     indices = torch.LongTensor(indices)
        #     if torch.cuda.is_available():
        #         indices = indices.cuda()
        #     total_loss = 0
        #     for batch in indices.split(params.mini_batch_size):
        #         src_optimizer.zero_grad()
        #         vecs0 = src_data['vecs'][batch]  # original
        #         vecs = src_data['F'](src_data['seqs'][batch], src_data['E'])
        #         loss = F.mse_loss(vecs0, vecs)
        #         loss.backward()
        #         total_loss += float(loss)
        #         src_optimizer.step()
        # print('Done: final loss = {:.2f}'.format(total_loss))

        src_optimizer = optim.SGD(src_data['F'].parameters(), lr=params.sw_learning_rate, momentum=0.9)
        print('Src optim: {}'.format(src_optimizer))
        # Loss function
        loss_fn = torch.nn.BCELoss()

        # Create models
        g = Generator(input_size=params.g_input_size, hidden_size=params.g_hidden_size,
                      output_size=params.g_output_size)

        if self.params.model_file:
            print('Load a model from ' + self.params.model_file)
            g.load(self.params.model_file)

        d = Discriminator(input_size=params.d_input_size, hidden_size=params.d_hidden_size,
                          output_size=params.d_output_size, hyperparams=get_hyperparams(params, disc=True))
        seed = params.seed
        self.initialize_exp(seed)

        if not params.disable_cuda and torch.cuda.is_available():
            print('Use GPU')
            # Move the network and the optimizer to the GPU
            g.cuda()
            d.cuda()
            loss_fn = loss_fn.cuda()

        if self.params.model_file is None:
            print('Initializing G based on distribution')
            # if the relative change of loss values is smaller than tol, stop iteration
            topn = 10000
            tol = 1e-5
            prev_loss, loss = None, None
            g_optimizer = optim.SGD(g.parameters(), lr=0.01, momentum=0.9)

            batches = src_data['seqs'][:topn].split(params.mini_batch_size)
            src_emb = torch.cat([src_data['F'](batch, src_data['E']).detach()
                                 for batch in batches])
            tgt_emb = tgt_data['E'].emb.weight[:topn]
            if not params.disable_cuda and torch.cuda.is_available():
                src_emb = src_emb.cuda()
                tgt_emb = tgt_emb.cuda()
            src_emb = F.normalize(src_emb)
            tgt_emb = F.normalize(tgt_emb)
            src_mean = src_emb.mean(dim=0).detach()
            tgt_mean = tgt_emb.mean(dim=0).detach()
            # src_std = src_emb.std(dim=0).deatch()
            # tgt_std = tgt_emb.std(dim=0).deatch()

            for _ in trange(1000):  # at most 1000 iterations
                prev_loss = loss
                g_optimizer.zero_grad()
                mapped_src_mean = g(src_mean)
                loss = F.mse_loss(mapped_src_mean, tgt_mean)
                loss.backward()
                g_optimizer.step()
                # Orthogonalize
                self.orthogonalize(g.map1.weight.data)
                loss = float(loss)
                if type(prev_loss) is float and abs(prev_loss - loss) / prev_loss <= tol:
                    break
            print('Done: final loss = {}'.format(float(loss)))
        evaluator.precision(g, src_data, tgt_data)
        sim = monitor.cosine_similarity(g, src_data, tgt_data)
        print('Cos sim.: {:3f} (+/-{:.3})'.format(sim.mean(), sim.std()))


        d_acc_epochs, g_loss_epochs = [], []

        # Define optimizers
        d_optimizer = optim.SGD(d.parameters(), lr=params.d_learning_rate)
        g_optimizer = optim.SGD(g.parameters(), lr=params.g_learning_rate)
        for epoch in range(params.num_epochs):
            d_losses, g_losses = [], []
            hit = 0
            total = 0
            start_time = timer()

            for mini_batch in range(0, params.iters_in_epoch // params.mini_batch_size):
                for d_index in range(params.d_steps):
                    d_optimizer.zero_grad()  # Reset the gradients
                    d.train()

                    X, y, _ = self.get_batch_data(src_data, tgt_data, g)
                    pred = d(X)
                    d_loss = loss_fn(pred, y)
                    d_loss.backward()
                    d_optimizer.step()

                    d_losses.append(d_loss.data.cpu().numpy())
                    discriminator_decision = pred.data.cpu().numpy()
                    hit += np.sum(discriminator_decision[:params.mini_batch_size] >= 0.5)
                    hit += np.sum(discriminator_decision[params.mini_batch_size:] < 0.5)

                    sys.stdout.write("[%d/%d] :: Discriminator Loss: %f \r" % (
                        mini_batch, params.iters_in_epoch // params.mini_batch_size, np.asscalar(np.mean(d_losses))))
                    sys.stdout.flush()

                    total += 2 * params.mini_batch_size * params.d_steps

                for g_index in range(params.g_steps):
                    # 2. Train G on D's response (but DO NOT train D on these labels)
                    g_optimizer.zero_grad()
                    src_optimizer.zero_grad()
                    d.eval()

                    X, y, src_vecs = self.get_batch_data(src_data, tgt_data, g)
                    pred = d(X)
                    g_loss = loss_fn(pred, 1 - y)
                    src_loss = F.mse_loss(*src_vecs)
                    if g_loss.is_cuda:
                        src_loss = src_loss.cuda()
                    loss = g_loss + penalty * src_loss
                    loss.backward()
                    g_optimizer.step()  # Only optimizes G's parameters
                    src_optimizer.step()

                    g_losses.append(g_loss.data.cpu().numpy())

                    # Orthogonalize
                    self.orthogonalize(g.map1.weight.data)

                    sys.stdout.write("[%d/%d] ::                                     Generator Loss: %f \r" % (
                        mini_batch, params.iters_in_epoch // params.mini_batch_size, np.asscalar(np.mean(g_losses))))
                    sys.stdout.flush()

                d_acc_epochs.append(hit / total)
                g_loss_epochs.append(np.asscalar(np.mean(g_losses)))
            print("Epoch {} : Discriminator Loss: {:.5f}, Discriminator Accuracy: {:.5f}, Generator Loss: {:.5f}, Time elapsed {:.2f} mins".format(epoch, np.asscalar(np.mean(d_losses)), hit / total, np.asscalar(np.mean(g_losses)), (timer() - start_time) / 60))

            filename = path.join(params.model_dir, 'g_e{}.pth'.format(epoch))
            print('Save a generator to ' + filename)
            g.save(filename)
            filename = path.join(params.model_dir, 's_e{}.pth'.format(epoch))
            print('Save a subword transformer to ' + filename)
            src_data['F'].save(filename)
            if (epoch + 1) % params.print_every == 0:
                evaluator.precision(g, src_data, tgt_data)
                sim = monitor.cosine_similarity(g, src_data, tgt_data)
                print('Cos sim.: {:3f} (+/-{:.3})'.format(sim.mean(), sim.std()))

        return g

    def orthogonalize(self, W):
        beta = self.params.beta
        W.copy_((1 + beta) * W - beta * W.mm(W.transpose(0, 1).mm(W)))

    def get_batch_data(self, src_data, tgt_data, g,
                       coef=0.01):
        params = self.params
        # Sample params.mini_batch_size vectors
        n_src = min(params.most_frequent_sampling_size, src_data['seqs'].size(0))
        n_tgt = min(params.most_frequent_sampling_size, tgt_data['idx2word'].shape[0])
        src_indices = torch.randperm(n_src)[:params.mini_batch_size]
        tgt_batch = torch.randperm(n_tgt)[:params.mini_batch_size]
        if src_data['E'].emb.weight.is_cuda:  # using GPU
            src_indices, tgt_batch = src_indices.cuda(), tgt_batch.cuda()
        src_batch = src_data['seqs'][src_indices]
        # Generate fake target-side vectors
        src_vecs0 = Variable(src_data['vecs'][src_indices])  # original
        src_vecs = src_data['F'](src_batch, src_data['E'])
        src_vecs = F.normalize(src_vecs)
        if g.map1.weight.is_cuda:
            fake = g(src_vecs.cuda())
            real = tgt_data['E'](tgt_batch).cuda()
        else:
            fake = g(src_vecs)
            real = tgt_data['E'](tgt_batch)
        real = F.normalize(real)
        X = torch.cat([fake, real], 0)
        y = torch.zeros(2 * params.mini_batch_size)
        if g.map1.weight.is_cuda:
            y = y.cuda()
        y[:params.mini_batch_size] = 1 - params.smoothing   # As per fb implementation
        y[params.mini_batch_size:] = params.smoothing
        return X, y, (src_vecs0, src_vecs)


def _init_xavier(m):
    if type(m) == torch.nn.Linear:
        fan_in = m.weight.size()[1]
        fan_out = m.weight.size()[0]
        std = np.sqrt(6.0 / (fan_in + fan_out))
        m.weight.data.normal_(0, std)


def _clip(d, clip):
    if clip > 0:
        for x in d.parameters():
            x.data.clamp_(-clip, clip)


def _get_eval_params(params):
    params = copy.deepcopy(params)
    params.ks = [1]
    params.methods = ['nn']
    params.models = ['adv']
    params.refine = ['without-ref']
    return params


def get_hyperparams(params, disc=True):
    if disc:
        return DiscHyperparameters(params)
    else:
        return GenHyperparameters(params)
