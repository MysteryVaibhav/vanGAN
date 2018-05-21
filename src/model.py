import torch
import torch.nn as nn
import torch.nn.functional as F
# from properties import *
import math


class Generator(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, hyperparams=None):
        super(Generator, self).__init__()

        self.context = hyperparams.context
        assert self.context in [0, 1, 2]
        
        self.non_linear = hyperparams.non_linear

        if self.context == 0 or self.context == 2:
            self.map1 = nn.Linear(input_size, output_size, bias=False)
            #nn.init.orthogonal(self.map1.weight)
            nn.init.eye(self.map1.weight)   # As per the fb implementation initialization
        elif self.context == 1:
            self.map1 = nn.Linear(input_size, output_size, bias=False)
            self.map1.weight.data = torch.cat([torch.eye(output_size), torch.zeros(output_size, output_size)], 0).transpose(0, 1)

            # print(input_size)
            # print(output_size)
            # print(self.map1.weight.data)
            # print(self.map1.data)

            if self.non_linear == 1:
                leaky_slope = hyperparams.leaky_slope
                self.activation1 = nn.LeakyReLU(leaky_slope)
                self.map2 = nn.Linear(300, 300, bias=True)

    def forward(self, x):
        if self.context == 0 or self.context == 2:
            return self.map1(x)
        else:
            if self.non_linear == 1:
                return self.map2(self.activation1(self.map1(x)))
            else:
                return self.map1(x)


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hyperparams):
        dropout_inp = hyperparams.dropout_inp
        dropout_hidden = hyperparams.dropout_hidden
        leaky_slope = hyperparams.leaky_slope
        self.add_noise = hyperparams.add_noise
        self.noise_mean = hyperparams.noise_mean
        self.noise_var = hyperparams.noise_var

        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.drop1 = nn.Dropout(dropout_inp)
        self.activation1 = nn.LeakyReLU(leaky_slope)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.drop2 = nn.Dropout(dropout_hidden)    # As per the fb implementation
        self.activation2 = nn.LeakyReLU(leaky_slope)
        self.map3 = nn.Linear(hidden_size, output_size)

    def gaussian(self, ins, mean, stddev):
        noise = torch.autograd.Variable(ins.data.new(ins.size()).normal_(mean, stddev))
        return ins * noise

    def forward(self, x):
        if self.add_noise:
            x = self.gaussian(x, mean=self.noise_mean, stddev=self.noise_var)  # muliplicative guassian noise
        x = self.activation1(self.map1(self.drop1(x))) # Input dropout
        x = self.drop2(self.activation2(self.map2(x)))
        return F.sigmoid(self.map3(x)).view(-1)


class Attention(nn.Module):
    def __init__(self, atype='dot', **kwargs):
        super(Attention, self).__init__()
        self.atype = atype
        if self.atype == 'dot':
            pass
        elif self.atype == 'mlp':
            self.inp_size = 600
            self.hidden_size = 50
            self.map1 = nn.Linear(self.inp_size, self.hidden_size)
            self.v = nn.Parameter(torch.rand(self.hidden_size))
            stdv = 1. / math.sqrt(self.hidden_size)
            self.v.data.normal_(mean=0, std=stdv)
        elif self.atype == 'bilinear':
            self.input_size = 300
            self.map1 = nn.Linear(self.input_size, self.input_size, bias=False)
            nn.init.eye(self.map1.weight)

    def forward(self, H, h):
        if self.atype == 'dot':
            return torch.matmul(H, h[:, :, None]).squeeze()
        elif self.atype == 'mlp':
            B, k, d = H.size()
            h = h[:, :, None].transpose(1, 2).repeat(1, k, 1) # [B, d] -> [B, k, d]
            energy = F.tanh(self.map1(torch.cat([H, h], 2)))  # [B*k*2d]->[B*k*d]
            energy = energy.transpose(2, 1)  # [B*d*k]
            v = self.v.repeat(B, 1).unsqueeze(1)  # [B*1*d]
            energy = torch.bmm(v, energy)  # [B*1*k]
            return energy.squeeze(1)  # [B*T]
        elif self.atype == 'bilinear':
            h = self.map1(h)
            return torch.matmul(H, h[:, :, None]).squeeze()
        

class RankPredictor(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, leaky_slope):
        super(RankPredictor, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.activation1 = nn.LeakyReLU(leaky_slope)
        self.map2 = nn.Linear(hidden_size, output_size)

    def forward(self, h):
        h = self.map1(h)
        h = self.activation1(h)
        h = self.map2(h)
        return h
