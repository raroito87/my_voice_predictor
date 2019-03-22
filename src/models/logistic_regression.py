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

    def get_batch(self, x, y, batch_idx, batch_size):
        batch_x = self.x[batch_idx * batch_size: (batch_idx + 1) * batch_size, :]
        batch_y = self.y[batch_idx * batch_size: (batch_idx + 1) * batch_size]

        return batch_x, batch_y

    def reshape_data(self, x):
        return x

    def get_args(self):
        return [self.name, self.d_in, self.d_out, self.dtype, self.device]
