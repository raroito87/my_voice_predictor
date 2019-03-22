import torch

# Neural Network
class LogReg(torch.nn.Module):
    def __init__(self, name, d_in, d_out, dtype=torch.float, device='cpu'):
        super(LogReg, self).__init__()

        self.dtype = dtype
        self.device = device

        self.name = name
        self.lin = torch.nn.Linear(d_in, d_out)
        self.softmax = torch.nn.Softmax(d_out)


    def forward(self, x):
        z = self.lin(x)
        return (z)

    def predict_prob(self, x):
        z = self.forward(x)
        prob = self.softmax(z)
        return (prob)

    def predict(self, x):
        z = self.forward(x)
        pred = torch.max(z.data, 1)[1]
        return (pred)
