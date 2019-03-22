
import torch
import torch.nn as nn

# Neural Network
class IMDB_NN_Model(torch.nn.Module):
    def __init__(self, name, d_in, H0, H1, d_out, dtype=torch.float, device='cpu'):
        super(IMDB_NN_Model, self).__init__()

        self.dtype = dtype
        self.device = device

        self.name = name
        self.H0 = H0
        self.H1 = H1

        #1 hidden layers
        self.lin_in = nn.Linear(d_in, H0)

        self.lin1 = nn.Linear(H0, H1)

        self.lin_out = nn.Linear(H1, d_out)

        self.activation_function = nn.Sigmoid()



    def forward(self, x):
        z1 = self.lin_in(x)
        a1 = self.activation_function(z1)

        z2 = self.lin1(a1)
        a2 = self.activation_function(z2)

        z3 = self.lin_out(a2)
        return (z3)

    def predict_prob(self, x):
        z1 = self.lin_in(x)
        a1 = self.activation_function(z1)

        z2 = self.lin1(a1)
        a2 = self.activation_function(z2)

        z3 = self.lin_out(a2)
        prob = self.activation_function(z3)
        return (prob)

    def predict(self, x):
        z1 = self.lin_in(x)
        a1 = self.activation_function(z1)

        z2 = self.lin1(a1)
        a2 = self.activation_function(z2)

        z3 = self.lin_out(a2)

        pred = torch.max(z3.data, 1)[1]
        return (pred)
