# from properties import *
from datetime import timedelta
from model import Generator, Discriminator
from timeit import default_timer as timer
from torch.autograd import Variable
from util import *
import copy
from monitor import Monitor
from evaluator import Evaluator
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
        penalty = 0.1  # penalty on cosine similarity
        # Load data
        if not os.path.exists(params.data_dir):
            raise "Data path doesn't exists: %s" % params.data_dir

        src_lang = params.src_lang
        tgt_lang = params.tgt_lang
        self.suffix_str = src_lang + '_' + tgt_lang

        monitor = Monitor(params, src_data=src_data, tgt_data=tgt_data)

        if torch.cuda.is_available():
            src_data.E = src_data.E.cuda()
            src_data.F = src_data.F.cuda()
            tgt_data.E = tgt_data.E.cuda()
            tgt_data.F = tgt_data.F.cuda()
            src_data.vecs = src_data.vecs.cuda()
            tgt_data.vecs = tgt_data.vecs.cuda()
            src_data.seqs = src_data.seqs.cuda()
            tgt_data.seqs = tgt_data.seqs.cuda()

        for _ in range(params.num_random_seeds):

            # Create models
            g = Generator(input_size=params.g_input_size, hidden_size=params.g_hidden_size,
                          output_size=params.g_output_size, hyperparams=get_hyperparams(params, disc=False))
            d = Discriminator(input_size=params.d_input_size, hidden_size=params.d_hidden_size,
                              output_size=params.d_output_size, hyperparams=get_hyperparams(params, disc=True))
            seed = random.randint(0, 1000)
            self.initialize_exp(seed)

            # Define loss function and optimizers
            loss_fn = torch.nn.BCELoss()
            d_optimizer = optim.SGD(d.parameters(), lr=params.d_learning_rate)
            g_optimizer = optim.SGD(g.parameters(), lr=params.g_learning_rate)
            src_optimizer = optim.SGD(src_data.F.parameters(), lr=params.d_learning_rate)
            # tgt_optimizer = optim.SGD(tgt.parameters(), lr=params.d_learning_rate) (not required?)

            if torch.cuda.is_available():
                # Move the network and the optimizer to the GPU
                g = g.cuda()
                d = d.cuda()
                loss_fn = loss_fn.cuda()
            # true_dict = get_true_dict(params.data_dir)
            d_acc_epochs = []
            g_loss_epochs = []

            # logs for plotting later
            log_file = open("log_src_tgt.txt", "w")     # Being overwritten in every loop, not really required
            log_file.write("epoch, dis_loss, dis_acc, g_loss\n")

            try:
                for epoch in range(params.num_epochs):
                    d_losses = []
                    g_losses = []
                    hit = 0
                    total = 0
                    start_time = timer()

                    for mini_batch in range(0, params.iters_in_epoch // params.mini_batch_size):
                        for d_index in range(params.d_steps):
                            d_optimizer.zero_grad()  # Reset the gradients
                            src_optimizer.zero_grad()
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
                            src_loss = 1 - F.cosine_similarity(*src_vecs).mean()
                            loss = g_loss * penalty * src_loss
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
                    print("Epoch {} : Discriminator Loss: {:.5f}, Discriminator Accuracy: {:.5f}, Generator Loss: {:.5f}, Time elapsed {:.2f} mins".
                          format(epoch, np.asscalar(np.mean(d_losses)), hit / total, np.asscalar(np.mean(g_losses)),
                                 (timer() - start_time) / 60))

                    # lr decay
                    g_optim_state = g_optimizer.state_dict()
                    old_lr = g_optim_state['param_groups'][0]['lr']
                    g_optim_state['param_groups'][0]['lr'] = max(old_lr * params.lr_decay, params.lr_min)
                    g_optimizer.load_state_dict(g_optim_state)
                    print("Changing the learning rate: {} -> {}".format(old_lr, g_optim_state['param_groups'][0]['lr']))
                    d_optim_state = d_optimizer.state_dict()
                    d_optim_state['param_groups'][0]['lr'] = max(d_optim_state['param_groups'][0]['lr'] * params.lr_decay, params.lr_min)
                    d_optimizer.load_state_dict(d_optim_state)

                    if (epoch + 1) % params.print_every == 0:
                        sim = monitor.cosine_similarity(g, src_data, tgt_data)
                        print('Cos sim.: {:3f} (+/-{:.3})'.format(sim.mean(), sim.std()))
                        # log_file.write("{},{:.5f},{:.5f},{:.5f}\n".format(epoch + 1, np.asscalar(np.mean(d_losses)), hit / total, np.asscalar(np.mean(g_losses))))
                        # log_file.write(str(all_precisions) + "\n")
                        # Saving generator weights

                        elems = ['generator_weights', self.suffix_str, 'seed', str(seed),
                                 'mf', str(epoch), 'lr', 'params.g_learning_rate']
                        torch.save(g.state_dict(), '_'.join(elems) + '.t7')

                # Save the plot for discriminator accuracy and generator loss
                fig = plt.figure()
                plt.plot(range(0, params.num_epochs), d_acc_epochs, color='b', label='discriminator')
                plt.plot(range(0, params.num_epochs), g_loss_epochs, color='r', label='generator')
                plt.ylabel('accuracy/loss')
                plt.xlabel('epochs')
                plt.legend()
                fig.savefig('d_g.png')

            except KeyboardInterrupt:
                print("Interrupted.. saving model !!!")
                torch.save(g.state_dict(), 'g_model_interrupt.t7')
                torch.save(d.state_dict(), 'd_model_interrupt.t7')
                log_file.close()
                exit()

            log_file.close()

        return g

    def orthogonalize(self, W):
        params = self.params
        W.copy_((1 + params.beta) * W - params.beta * W.mm(W.transpose(0, 1).mm(W)))

    def get_batch_data(self, src_data, tgt_data, g,
                       coef=0.01):
        params = self.params
        # Sample params.mini_batch_size vectors
        n = min(params.most_frequent_sampling_size, src_data.seqs.size(0))
        src_indices = torch.randperm(n)[:params.mini_batch_size]
        tgt_indices = torch.randperm(n)[:params.mini_batch_size]
        if src_data.E.emb.weight.is_cuda:  # using GPU
            src_indices, tgt_indices = src_indices.cuda(), tgt_indices.cuda()
        src_batch, tgt_batch = src_data.seqs[src_indices], tgt_data.seqs[tgt_indices]
        # Generate fake target-side vectors
        src_vecs0 = Variable(src_data.vecs[src_indices])  # original
        src_vecs = src_data.F(Variable(src_batch), src_data.E)
        fake = g(src_vecs)
        real = tgt_data.F(Variable(tgt_batch), tgt_data.E, transform=False)
        X = torch.cat([fake, real], 0)
        y = to_variable(torch.zeros(2 * params.mini_batch_size))
        y[:params.mini_batch_size] = 1 - params.smoothing   # As per fb implementation
        y[params.mini_batch_size:] = params.smoothing
        return X, y, (src_vecs0, src_vecs)


def construct_input(knn_emb, indices, src_emb, attn, use_cuda=False):
    if torch.cuda.is_available() and use_cuda:
        indices = indices.cuda()
    knn = knn_emb(indices).type(torch.LongTensor)
    if torch.cuda.is_available() and use_cuda:
        knn = knn.cuda()
    H = src_emb(knn)
    h = src_emb(to_variable(indices, use_cuda=use_cuda))
    p = F.softmax(attn(H, h), dim=1)
    c = torch.matmul(H.transpose(1, 2), p.unsqueeze(2)).squeeze()
    return torch.cat([h, c], 1)


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


# Finds top-k neighbours and puts them in a matrix
def get_knn_list(src_ids, emb, params, suffix_str, method='knn'):
    xq = emb(src_ids).data
    xb = emb.weight.data
    xb = xb/xb.norm(2, 1)[:, None]
    xq = xq/xq.norm(2, 1)[:, None]

    k = params.K + 1

    if method == 'knn':
        _, knn = eval.get_knn_indices(k, xb, xq)

    else:
        r_source = eval.common_csls_step(params.csls_k, xb, xq)
        try:
            r_target = pickle.load(open('r_target_' + suffix_str + '.pkl', 'rb'))
        except FileNotFoundError:
            print("Calculating r_target...")
            start_time = time.time()
            xb = emb.weight.data
            xb = xb / xb.norm(2, 1)[:, None]
            r_target = eval.common_csls_step(params.csls_k, xb, xb)
            with open('r_target_' + suffix_str + '.pkl', 'wb') as fp:
                pickle.dump(r_target, fp, pickle.HIGHEST_PROTOCOL)
            print("Time taken for making r_target: %.2f" % (time.time() - start_time))
        if torch.cuda.is_available:
            r_source = r_source.cuda()
            r_target = r_target.cuda()
        knn = eval.csls(k, xb, xq, r_source, r_target)
    return modify_knn(knn, src_ids)


def modify_knn(knn, src_ids):
    r, c = knn.size()
    last_id = knn[:, c - 1]
    knn[knn == src_ids[:, None].repeat(1, c)] = last_id
    return knn[:, :-1]


def get_knn_embedding(params, src_emb, suffix_str):
    start_time_begin = time.time()
    # max_top = params.most_frequent_sampling_size
    max_top = params.top_frequent_words
    # Construct knn list embedding layer
    if torch.cuda.is_available():
        bs = 4096
    else:
        bs = 1500
    knn_list = None
    for i in range(0, max_top, bs):
        start_time = time.time()
        lo = i
        hi = min(max_top, i + bs)
        src_ids = torch.arange(lo, hi).type(torch.LongTensor)
        if torch.cuda.is_available():
            src_ids = src_ids.cuda()
        temp = get_knn_list(src_ids, src_emb, params, suffix_str, method='csls')

        if knn_list is None:
            knn_list = temp
        else:
            knn_list = torch.cat([knn_list, temp], 0)
        print("Time taken for iteration %d: %.2f" % (i, time.time() - start_time))
    knn_list = knn_list.cpu().numpy()
    with open('full_knn_list_' + suffix_str + '.pkl', 'wb') as fp:
        pickle.dump(knn_list, fp, pickle.HIGHEST_PROTOCOL)
    print("Total time taken for constructing knn list: %.2f" % (time.time() - start_time_begin))
    return knn_list
