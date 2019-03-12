import torch


# LogReg
class logreg(nn.Module):
    def __init__(self):
        super(logreg, self).__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(D_in, D_out)
        )

    def forward(self, x):#prediction ohne softmax
        ...
        return x

    def backward(self, x):#gradient  computation
        ...
        return x

    def predict(self, x):#prediction mit softmax
        ...
        return x