import sys
import torch
import numpy as np
import torch.utils.data
from properties import *
import torch.optim as optim
from util import get_embeddings
from model import Generator, Discriminator
from timeit import default_timer as timer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def to_tensor(numpy_array):
    return torch.from_numpy(numpy_array).float()


def to_variable(tensor, volatile=False):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return torch.autograd.Variable(tensor, volatile)


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
        output[:] = smoothing
    else:
        real = to_variable(to_tensor(it_batch))
        input = torch.cat([fake, real], 0)
        output = to_variable(torch.FloatTensor(2 * mini_batch_size).zero_())
        output[:mini_batch_size] = smoothing
        output[mini_batch_size:] = 1 - smoothing
    return input, output


def init_xavier(m):
    if type(m) == torch.nn.Linear:
        fan_in = m.weight.size()[1]
        fan_out = m.weight.size()[0]
        std = np.sqrt(6.0 / (fan_in + fan_out))
        m.weight.data.normal_(0,std)


def train():
    # Load data
    en, it = get_embeddings()   # Vocab x Embedding_dimension

    # Create models
    g = Generator(input_size=g_input_size, hidden_size=g_hidden_size, output_size=g_output_size)
    d = Discriminator(input_size=d_input_size, hidden_size=d_hidden_size, output_size=d_output_size)

    init_xavier(g)
    init_xavier(d)
    
    # Define loss function and optimizers
    loss_fn = torch.nn.BCELoss()
    d_optimizer = optim.SGD(d.parameters(), lr=d_learning_rate, weight_decay=0.001)
    g_optimizer = optim.SGD(g.parameters(), lr=g_learning_rate, weight_decay=0.001)

    if torch.cuda.is_available():
        # Move the network and the optimizer to the GPU
        g = g.cuda()
        d = d.cuda()
        loss_fn = loss_fn.cuda()

    d_acc_epochs = []
    g_loss_epochs = []
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
                pred = d(input)
                d_loss = loss_fn(pred, output)
                d_loss.backward()  # compute/store gradients, but don't change params
                d_losses.append(d_loss.data.cpu().numpy())
                discriminator_decision = pred.data.cpu().numpy()
                hit += np.sum(discriminator_decision[:mini_batch_size] < 0.5)
                hit += np.sum(discriminator_decision[mini_batch_size:] >= 0.5)
                d_optimizer.step()  # Only optimizes D's parameters; changes based on stored gradients from backward()
                
                # Clip weights
                clip(d, clip_value)
                
                sys.stdout.write("[%d/%d] :: Discriminator Loss: %f \r" % (
                    mini_batch, len(en) // mini_batch_size, np.asscalar(np.mean(d_losses))))
                sys.stdout.flush()
            
            total += 2*mini_batch_size*d_steps

            for g_index in range(g_steps):
                # 2. Train G on D's response (but DO NOT train D on these labels)
                g_optimizer.zero_grad()

                input, output = get_batch_data(en, it, g, detach=False, forGenerator=True)
                pred = d(input)
                g_loss = loss_fn(pred, 1 - output)
                g_loss.backward()
                g_losses.append(g_loss.data.cpu().numpy())
                g_optimizer.step()  # Only optimizes G's parameters
                
                # Orthogonalize
                orthogonalize(g.map1.weight.data)
                
                sys.stdout.write("[%d/%d] :: Generator Loss: %f \r" % (
                    mini_batch, len(en) // mini_batch_size, np.asscalar(np.mean(g_losses))))
                sys.stdout.flush()

        d_acc_epochs.append(hit / total)
        g_loss_epochs.append(np.asscalar(np.mean(g_losses)))
        print("Epoch {} : Discriminator Loss: {:.5f}, Discriminator Accuracy: {:.5f}, Generator Loss: {:.5f}, Time elapsed {:.2f} mins".
              format(epoch, np.asscalar(np.mean(d_losses)), hit/total, np.asscalar(np.mean(g_losses)),
                     (timer() - start_time) / 60))
        if epoch % 5 == 0:
            torch.save(g.state_dict(), 'generator_weights_{}.t7'.format(epoch))
    
    # Save the plot for discriminator accuracy and generator loss
    fig = plt.figure()
    plt.plot(range(0, num_epochs), d_acc_epochs, color='b', label='discriminator')
    plt.plot(range(0, num_epochs), g_loss_epochs, color='r', label='generator')
    plt.ylabel('accuracy/loss')
    plt.xlabel('epochs')
    plt.legend()
    fig.savefig('d_g.png')
    return g


def orthogonalize(W):
    W.copy_((1 + beta) * W - beta * W.mm(W.transpose(0, 1).mm(W)))

    
def clip(d, clip):
    if clip > 0:
        for x in d.parameters():
            x.data.clamp_(-clip, clip)


if __name__ == '__main__':
    generator = train()