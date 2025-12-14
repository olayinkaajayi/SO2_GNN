import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

###MLP with lienar output
class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, three_D=False):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''

        super(MLP, self).__init__()

        self.linear_or_not = True #default is linear model
        self.num_layers = num_layers
        self.three_D = three_D # To deal with batchnorm

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            #Linear model
            self.linear = nn.Linear(input_dim, output_dim, bias=True)
        else:
            #Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for i in range(num_layers):
                init.xavier_uniform_(self.linears[i].weight)
                init.constant_(self.linears[i].bias, 0)

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            #If linear model
            return self.linear(x)
        else:
            #If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = self.linears[layer](h)
                if self.three_D:
                    h = h.transpose(-1,-2)
                    h = self.batch_norms[layer](h)
                    h = h.transpose(-1,-2)
                else:
                    h = self.batch_norms[layer](h)
                h = F.relu(h)

            return self.linears[self.num_layers - 1](h)