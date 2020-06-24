'''
This file is part of Qwoppy.

Copyright (C) 2020 Chris Swinchatt

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
'''

import torch
import torch.autograd as autograd
import torch.nn       as nn
import torch.optim    as optim

def is_enumerable(x):
    return hasattr(x, '__len__')

class Agent(nn.Module):
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
