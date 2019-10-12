import torch
import torch.autograd as autograd
import torch.nn       as nn
import torch.optim    as optim

def is_enumerable(x):
    return hasattr(x, '__len__')

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm   = nn.LSTM(1, 256)
        self.output = nn.Linear(256, 6)
        self.add_module('lstm',   self.lstm)
        self.add_module('output', self.output)

    def __len__(self):
        return len(self.shape)

    def forward(self, X):
        X, _ = self.lstm.forward(X)
        return self.output.forward(X)
    
    def generate_action(self, current_distance):
        X = torch.tensor([[[current_distance]]], requires_grad=True)
        X = self.forward(X)
        y = torch.argmax(X)
        return y.item()
