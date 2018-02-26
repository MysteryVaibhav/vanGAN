import sys
import torch
import numpy as np
import torch.utils.data
from properties import *
import torch.optim as optim
from util import get_embeddings
from model import Generator, Discriminator
from timeit import default_timer as timer
#from validation_faiss import *


def to_tensor(numpy_array):
    return torch.from_numpy(numpy_array).float()


def to_variable(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return torch.autograd.Variable(tensor)


def get_batch_data(en, it, g, detach=False, forGenerator = False):
    en_len = en.shape[0]
    it_len = it.shape[0]
    random_en_indices = np.random.permutation(en_len)
    random_it_indices = np.random.permutation(it_len)
    en_batch = en[random_en_indices[:mini_batch_size]]
    it_batch = it[random_it_indices[:mini_batch_size]]
    fake = g(to_variable(to_tensor(en_batch)))
    if detach:
        fake = fake.detach()
    if forGenerator:
        input = fake
        output = to_variable(torch.FloatTensor(mini_batch_size).zero_())
        output[:] = 1 - smoothing
    else:
        real = to_variable(to_tensor(it_batch))
        input = torch.cat([fake, real], 0)
        output = to_variable(torch.FloatTensor(2 * mini_batch_size).zero_())
        output[:mini_batch_size] = smoothing
        output[mini_batch_size:] = 1-smoothing
    return input, output


def train():
    # Load data
    en, it = get_embeddings()  # Vocab x Embedding_dimension

    # Create models
    g = Generator(input_size=g_input_size, hidden_size=g_hidden_size, output_size=g_output_size)
    d = Discriminator(input_size=d_input_size, hidden_size=d_hidden_size, output_size=d_output_size)

    # Define loss function and optimizers
    lr = d_learning_rate
    loss_fn = torch.nn.BCELoss()
    d_optimizer = optim.SGD(d.parameters(), lr=lr, weight_decay=0.95)
    # g_optimizer = optim.SGD(g.parameters(), lr=lr, weight_decay=0.95)
    # d_optimizer = optim.Adam(d.parameters(), lr=lr, betas=optim_betas)
    g_optimizer = optim.Adam(g.parameters(), lr=lr, betas=optim_betas)

    if torch.cuda.is_available():
        # Move the network and the optimizer to the GPU
        g = g.cuda()
        d = d.cuda()
        loss_fn = loss_fn.cuda()

    d_acc = []
    #true_dict = get_true_dict()
    for epoch in range(num_epochs):
        d_losses = []
        g_losses = []
        hit = 0
        total = 0
        start_time = timer()
        for mini_batch in range(0, len(en) // mini_batch_size):
            for d_index in range(d_steps):
                d_optimizer.zero_grad()  # Reset the gradients
                # noise = torch.normal(torch.ones(mini_batch_size, d_input_size) * 5,
                #                      torch.ones(mini_batch_size, d_input_size) * 2)

                input, output = get_batch_data(en, it, g, detach=True)
                # print("Input: ", input)
                l1, a1, l2, a2, l3, pred = d(input)
                # print("Layer-1 Linear: ", l1)
                # print("Layer-1 Activation: ", a1)
                # print("Layer-2 Linear: ", l2)
                # print("Layer-2 Activation: ", a2)
                # print("Layer-3 Linear: ", l3)
                # print("Pred: ", pred)
                d_loss = loss_fn(pred, output)
                d_loss.backward()  # compute/store gradients, but don't change params
                d_losses.append(d_loss.data.cpu().numpy())
                discriminator_decision = pred.data.cpu().numpy()
                # print("Fake data: ", discriminator_decision[
                #                         :mini_batch_size])
                # print("Real data: ", discriminator_decision[mini_batch_size:])
                hit += np.sum(discriminator_decision[:mini_batch_size] < 0.5)
                # print("Hits fake: ", hit)
                hit += np.sum(discriminator_decision[mini_batch_size:] >= 0.5)
                # print("Hits real: ", hit)
                d_optimizer.step()  # Only optimizes D's parameters; changes based on stored gradients from backward()
                sys.stdout.write("[%d/%d] :: Discriminator Loss: %f \r" % (
                    mini_batch, len(en) // mini_batch_size, np.asscalar(np.mean(d_losses))))
                sys.stdout.flush()
            total += 2 * mini_batch_size * d_steps

            for g_index in range(g_steps):
                # 2. Train G on D's response (but DO NOT train D on these labels)
                g_optimizer.zero_grad()

                input, output = get_batch_data(en, it, g, detach=False, forGenerator=True)
                l1, a1, l2, a2, l3, pred = d(input)
                g_loss = loss_fn(pred, output)
                g_loss.backward()
                g_losses.append(g_loss.data.cpu().numpy())
                g_optimizer.step()  # Only optimizes G's parameters
                # Orthogonalize
                orthogonalize(g.map1.weight.data)
                sys.stdout.write("[%d/%d] :: Generator Loss: %f \r" % (
                    mini_batch, len(en) // mini_batch_size, np.asscalar(np.mean(g_losses))))
                sys.stdout.flush()

        d_acc.append(hit / total)
        print(
            "Epoch {} : Discriminator Loss: {:.5f}, Discriminator Accuracy: {:.5f}, Generator Loss: {:.5f}, Time elapsed {:.2f} mins".
            format(epoch, np.asscalar(np.mean(d_losses)), hit/total,
                   np.asscalar(np.mean(g_losses)),
                   (timer() - start_time) / 60))
        torch.save(g.state_dict(), 'generator_weights_{}.t7'.format(epoch))
        torch.save(g.state_dict(), 'discriminator_weights_{}.t7'.format(epoch))
        # if epoch % 5 == 0:
        #     print(get_precision_k(10, g, true_dict))
        #     if epoch != 0:
        #         lr = lr / 2
    return g


def orthogonalize(W):
    W.copy_((1 + beta) * W - beta * W.mm(W.transpose(0, 1).mm(W)))


if __name__ == '__main__':
    generator = train()