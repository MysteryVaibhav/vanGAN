import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(input_size, output_size))
        nn.init.orthogonal(self.W)

    def forward(self, x, src2trg=True):
        if src2trg:  # source to target
            return x.matmul(self.W.t())
        return x.matmul(self.W.t())  # target to source


class Discriminator(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=500, sigma=0.5, **kwargs):
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
