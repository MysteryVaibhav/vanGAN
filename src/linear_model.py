import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(input_size, output_size, bias=False)
        # self.map1 = nn.Linear(input_size, hidden_size)
        # self.map2 = nn.Linear(hidden_size, hidden_size)
        # self.map3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x = F.relu(self.map1(x))
        # x = F.tanh(self.map2(x))
        # return self.map3(x)
        return self.map1(x)


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lerelu = nn.LeakyReLU()
        x = lerelu(self.map1(x))
        x = lerelu(self.map2(x))
        # x = F.relu(self.map1(x))
        # x = F.relu(self.map2(x))
        return F.sigmoid(self.map3(x))