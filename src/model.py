import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(input_size, output_size, bias=False)
        #nn.init.orthogonal(self.map1.weight)
        nn.init.eye(self.map1.weight)   # As per the fb implementation initialization

    def forward(self, x):
        return self.map1(x)


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.drop1 = nn.Dropout(0.1)
        self.activation1 = nn.LeakyReLU(0.2)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.drop2 = nn.Dropout(0)    # As per the fb implementation
        self.activation2 = nn.LeakyReLU(0.2)
        self.map3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.activation1(self.map1(self.drop1(x))) # Input dropout
        x = self.drop2(self.activation2(self.map2(x)))
        return F.sigmoid(self.map3(x)).view(-1)