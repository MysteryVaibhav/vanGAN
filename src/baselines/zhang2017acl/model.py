from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(input_size, output_size))
        nn.init.orthogonal(self.W)

    def forward(self, x, trg2src=True):
        if trg2src:
            return x.matmul(self.W.t())  # target to source
        return x.matmul(self.W)  # source to target

    def save(self, filename):
        torch.save(self.state_dict(), filename)


class Discriminator(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=500, sigma=0.5):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.activation1 = nn.ReLU()
        self.map2 = nn.Linear(hidden_size, output_size)
        self.sigma = sigma

    def forward(self, x):
        gpu = (type(x.data) is torch.cuda.FloatTensor)  # Using GPU?

        # Inject multiplicative Gaussian noise to input
        noise = Variable(torch.FloatTensor(x.size()).normal_(1.0, self.sigma),
                         volatile=False)
        if gpu:
            noise = noise.cuda()
        h = self.activation1(self.map1(x * noise))

        # Inject multiplicative Gaussian noise to hidden units
        noise = Variable(torch.FloatTensor(h.size()).normal_(1.0, self.sigma),
                         volatile=False)
        if gpu:
            noise = noise.cuda()
        return F.sigmoid(self.map2(h * noise)).view(-1)
