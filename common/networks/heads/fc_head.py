#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn

class FCHead(nn.Module):
    def __init__(self, base_neurons=[512, 256, 128], out_dim=3):
        super().__init__()
        layers = []
        for (inp_neurons, out_neurons) in zip(base_neurons[:-1], base_neurons[1:]):
            layers.append(nn.Linear(inp_neurons, out_neurons))
            layers.append(nn.ReLU())
        self.final_layer = nn.Linear(out_neurons, out_dim)
        self.decoder = nn.Sequential(*layers)

    def forward(self, inp):
        decoded = self.decoder(inp)
        out = self.final_layer(decoded)

        return out