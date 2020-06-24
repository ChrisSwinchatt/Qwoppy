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

import qwoppy.ocr as ocr

class Settings:
    RNN    = nn.GRU
    STACK  = 1
    HIDDEN = 256

class Agent(nn.Module):
    def __init__(self, device, train=True):
        super().__init__()
        self.train  = train
        self.device = device
        self.rnn    = Settings.RNN(ocr.Constants.NUM_TOKENS, Settings.HIDDEN, Settings.STACK).to(device)
        self.out    = nn.Linear(Settings.HIDDEN, 6).to(device)
        hsize       = (Settings.STACK, 1, Settings.HIDDEN)
        if Settings.RNN.__name__ == 'LSTM':
            h = torch.randn(hsize, device=device)
            c = torch.randn(hsize, device=device)
            self.h = (h,c)
        else:
            self.h = torch.randn(hsize, device=device)
        self.optim         = optim.Adam(self.parameters(), lr=0.0001)
        self.loss          = nn.MSELoss()
        self.prev_distance = None

    def generate_action(self, X):
        if self.train and self.prev_distance is not None:
            # If there is a previous distance, compute gradients & backprop.
            loss = self.loss(X, self.prev_distance)**2
            loss.backward(retain_graph=True)
            self.optim.step()
        if self.train:
            self.optim.zero_grad()
        self.prev_distance = X
        # Generate action from the new distance.
        X, self.h = self.rnn.forward(X, self.h)
        X         = self.out.forward(X)
        y         = torch.argmax(X)
        return y.item()
