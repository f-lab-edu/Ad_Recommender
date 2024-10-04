
import torch.nn as nn
from activation import *


class DNN(nn.Module):
    def __init__(self, inputs_dim, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False,
                 init_std=0.0001, dice_dim=3, seed=1024, device='cpu'):
        super(DNN, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        if len(hidden_units) == 0:
            raise ValueError("hidden_units is empty")
        hidden_units = [inputs_dim] + list(hidden_units)

        self.linears = nn.ModuleList([nn.Linear(hidden_units[i+1]) for i in range(len(hidden_units)-1)])

        if self.use_bn:
            self.bn = nn.ModuleList([nn.BatchNorm1d(hidden_units[i+1]) for i in range(len(hidden_units)-1)])

        self.activation_layers = activation_layer(activation)

        for name, tensor in self.linears.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)

        self.to(device)

    def forward(self, inputs):
        x = inputs
        for i in range(len(self.linears)):
            fc = self.linears[i](x)

            if self.use_bn:
                fc = self.bn[i](fc)

            fc = self.activation_layers[i](fc)
            fc = self.dropout(fc)
            x = fc
        return x
