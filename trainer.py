import sys
import torch
import numpy as np
import torch.utils.data
from properties import *
import torch.optim as optim
from util import get_embeddings
from model import Generator, Discriminator
from timeit import default_timer as timer


def to_tensor(numpy_array):
    return torch.from_numpy(numpy_array).float()


def to_variable(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return torch.autograd.Variable(tensor)


class CustomDataSet(torch.utils.data.TensorDataset):
    def __init__(self, data):
        self.data = data
        self.num_of_samples = len(self.data)

    def __len__(self):
        return self.num_of_samples

    def __getitem__(self, idx):
        return to_tensor(self.data[idx])


def train():
    # Load data
    en, it = get_embeddings()   # Vocab x Embedding_dimension

    # Create data-loaders
    en_data_loader = torch.utils.data.DataLoader(CustomDataSet(en), batch_size=mini_batch_size, shuffle=True)
    it_data_loader = torch.utils.data.DataLoader(CustomDataSet(it), batch_size=mini_batch_size, shuffle=True)

    # Create models
    g = Generator(input_size=g_input_size, hidden_size=g_hidden_size, output_size=g_output_size)
    d = Discriminator(input_size=d_input_size, hidden_size=d_hidden_size, output_size=d_output_size)

    # Define loss function and optimizers
    loss_fn = torch.nn.BCELoss()
    d_optimizer = optim.Adam(d.parameters(), lr=d_learning_rate, betas=optim_betas)
    g_optimizer = optim.Adam(g.parameters(), lr=g_learning_rate, betas=optim_betas)

    if torch.cuda.is_available():
        # Move the network and the optimizer to the GPU
        g = g.cuda()
        d = d.cuda()
        loss_fn = loss_fn.cuda()

    d_acc = []
    for epoch in range(num_epochs):
        d_losses = []
        g_losses = []
        hit = 0
        total = 0
        start_time = timer()
        en_iter = iter(en_data_loader)
        mini_batch = 1
        for d_real_data in it_data_loader:
            # Inspired from https://github.com/devnag/pytorch-generative-adversarial-networks/blob/master/gan_pytorch.py
            for d_index in range(d_steps):
                # 1. Train D on real+fake
                d.zero_grad()  # Reset the gradients

                #  1A: Train D on real
                noise = torch.normal(torch.ones(mini_batch_size, d_input_size) * 5,
                                     torch.ones(mini_batch_size, d_input_size) * 2)
                d_real_data = to_variable(torch.mul(d_real_data, noise))  # Could add some noise to the real data later

                d_real_decision = d(d_real_data)
                real_discriminator_decision = d_real_decision.data.cpu().numpy()
                hit += np.sum(real_discriminator_decision < 0.5)
                d_real_error = loss_fn(d_real_decision, to_variable(torch.zeros(mini_batch_size, 1)))  # ones = true
                d_real_error.backward()  # compute/store gradients, but don't change params
                d_losses.append(d_real_error.data.cpu().numpy())

                #  1B: Train D on fake
                d_gen_input = to_variable(next(en_iter))
                d_fake_data = g(d_gen_input).detach()  # detach to avoid training G on these labels
                d_fake_decision = d(d_fake_data)  # Add noise later
                fake_discriminator_decision = d_fake_decision.data.cpu().numpy()
                hit += np.sum(fake_discriminator_decision >= 0.5)
                d_fake_error = loss_fn(d_fake_decision, to_variable(torch.ones(mini_batch_size, 1)))  # zeros = fake
                d_fake_error.backward()
                d_losses.append(d_fake_error.data.cpu().numpy())
                d_optimizer.step()  # Only optimizes D's parameters; changes based on stored gradients from backward()
                sys.stdout.write("[%d/%d] :: Discriminator Loss: %f \r" % (
                    mini_batch, len(en) // mini_batch_size, np.asscalar(np.mean(d_losses))))
                sys.stdout.flush()
                mini_batch += 1
                total += 2*mini_batch_size

        mini_batch = 1
        it_iter = iter(it_data_loader)
        for gen_input in en_data_loader:
            for g_index in range(g_steps):
                # 2. Train G on D's response (but DO NOT train D on these labels)
                g.zero_grad()

                gen_input = to_variable(gen_input)
                g_fake_data = g(gen_input)
                g_fake_decision = d(g_fake_data)  # Add noise later
                g_error = loss_fn(g_fake_decision,
                                  to_variable(torch.zeros(mini_batch_size, 1)))  # we want to fool, so pretend it's all genuine
                g_losses.append(g_error.data.cpu().numpy())
                g_error.backward()

                g_input_real = to_variable(next(it_iter))
                real_decision = d(g_input_real)
                g_real_error = loss_fn(real_decision, to_variable(torch.ones(mini_batch_size, 1)))
                g_real_error.backward()
                g_losses.append(g_real_error.data.cpu().numpy())
                g_optimizer.step()  # Only optimizes G's parameters

                sys.stdout.write("[%d/%d] :: Generator Loss: %f \r" % (
                    mini_batch, len(en) // mini_batch_size, np.asscalar(np.mean(g_losses))))
                sys.stdout.flush()
                mini_batch += 1
        d_acc.append(hit/total)
        print("Epoch {} : Discriminator Loss: {:.5f}, Discriminator Accuracy: {:.5f}, Generator Loss: {:.5f}, Time elapsed {:.2f} mins".
              format(epoch, np.asscalar(np.mean(d_losses)), hit/total, np.asscalar(np.mean(g_losses)),
                     (timer() - start_time) / 60))
        torch.save(g.state_dict(), 'generator_weights_{}.t7'.format(epoch))
    return g


if __name__ == '__main__':
    generator = train()