import torch
import torch.nn as nn
import collections
from torch.nn.modules.flatten import Flatten
import numpy as np


def pad_K_dim(x, pad):
    padding = [0] * (len(x.shape) * 2 + 2)
    padding[-3] = pad
    return nn.functional.pad(x[:, None, ...], padding, mode='constant', value=0)


def extend_Z(x, vals):
    K = x.shape[1]
    pad = np.prod(x.shape[2:])
    x = pad_K_dim(x, pad)
    x[:, K:, ...] = vals

    return x


class ToZ(nn.Module):

    def __init__(self, eps):
        super(ToZ, self).__init__()
        self.eps = eps

    def forward(self, x):
        return extend_Z(x, self.eps)


class Normalization(nn.Module):

    def __init__(self, device):
        super(Normalization, self).__init__()
        self.mean = torch.FloatTensor([0.1307]).view((1, 1, 1, 1)).to(device)
        self.sigma = torch.FloatTensor([0.3081]).view((1, 1, 1, 1)).to(device)

    def forward(self, x):
        return (x - self.mean) / self.sigma


class FullyConnected(nn.Module):

    def __init__(self, device, input_size, fc_layers):
        super(FullyConnected, self).__init__()

        layers = [Normalization(device), Flatten()]
        prev_fc_size = input_size * input_size
        for i, fc_size in enumerate(fc_layers):
            layers += [nn.Linear(prev_fc_size, fc_size)]
            if i + 1 < len(fc_layers):
                layers += [nn.ReLU()]
            prev_fc_size = fc_size
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Conv(nn.Module):

    def __init__(self, device, input_size, conv_layers, fc_layers, n_class=10):
        super(Conv, self).__init__()

        self.input_size = input_size
        self.n_class = n_class

        layers = [Normalization(device)]
        prev_channels = 1
        img_dim = input_size

        for n_channels, kernel_size, stride, padding in conv_layers:
            layers += [
                nn.Conv2d(prev_channels, n_channels, kernel_size, stride=stride, padding=padding),
                nn.ReLU(),
            ]
            prev_channels = n_channels
            img_dim = img_dim // stride
        layers += [Flatten()]

        prev_fc_size = prev_channels * img_dim * img_dim
        for i, fc_size in enumerate(fc_layers):
            layers += [nn.Linear(prev_fc_size, fc_size)]
            if i + 1 < len(fc_layers):
                layers += [nn.ReLU()]
            prev_fc_size = fc_size
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class NNFullyConnectedZ(nn.Module):

    def __init__(self, device, input_size, fc_layers, eps):
        super(NNFullyConnectedZ, self).__init__()

        layers = [Normalization(device), ToZ(eps), Flatten(start_dim=2)]
        prev_fc_size = input_size * input_size
        for i, fc_size in enumerate(fc_layers):
            layers += [LinearZ(prev_fc_size, fc_size)]
            if i + 1 < len(fc_layers):
                layers += [ReLUZLinear(fc_size)]
            prev_fc_size = fc_size
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class NNConvZ(nn.Module):
    """
    Copy of Conv Module for the image. This module builds up the corresponding network for the Zonotope.
    """
    def __init__(self, device, input_size, conv_layers, fc_layers, eps, n_class=10):
        super(NNConvZ, self).__init__()

        self.input_size = input_size
        self.n_class = n_class

        layers = [Normalization(device), ToZ(eps)]
        prev_channels = 1
        height = width = input_size

        for n_channels, kernel_size, stride, padding in conv_layers:
            height, width = self._compute_resulting_height_width(height, width, stride, padding)
            layers += [
                ConvZ(prev_channels, n_channels, kernel_size, stride=stride, padding=padding),
                ReLUZConv(n_channels, height, width),
            ]
            prev_channels = n_channels
        layers += [Flatten(start_dim=2)]

        prev_fc_size = prev_channels * height * width
        for i, fc_size in enumerate(fc_layers):
            layers += [LinearZ(prev_fc_size, fc_size)]
            if i + 1 < len(fc_layers):
                layers += [ReLUZLinear(fc_size)]
            prev_fc_size = fc_size
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class LinearZ(nn.Linear):
    def load_state_dict(self, state_dict_in):
        state_dict = collections.OrderedDict([('weight', nn.Parameter(state_dict_in['weight'], requires_grad=False)),
                                              ('bias', nn.Parameter(torch.tensor([0.0]), requires_grad=False))])
        self.bias = state_dict_in['bias']
        self.layer.load_state_dict(state_dict)

    def forward(self, x):
        # input (N, K, input_size)
        print(x.shape)
        N, K, input_size = x.shape
        x = x.view(N * K, input_size)
        out = super(LinearZ, self).forward(x).view(N, K, -1)
        out[:, 0, :] += self.bias

        return out


class ConvZ(nn.Conv2d):
    def load_state_dict(self, state_dict_in):
        state_dict = collections.OrderedDict([('weight', nn.Parameter(state_dict_in['weight'], requires_grad=False)),
                                              ('bias', nn.Parameter(torch.tensor([0.0]), requires_grad=False))])
        self.bias = state_dict_in['bias']
        self.conv2d.load_state_dict(state_dict)

    def forward(self, x):
        # TODO: check this
        # input is (N, K, c_in, H, W) want to apply same conv2 for each k in range(K)
        # easy solution: just fold all K in batch dimension, and leverage pytorch's threaded computation of multiple
        # batch members. Btw in our setting, we always assume N=1.
        # We are essentially treating each epsilon layer as an image. However, do not treat these as batch members, as
        # we need to increase K in the ReLU
        # TODO: check whether we really need to fold in *and* out, can't we do all of this in the batch dimension?

        N, K, c_in, H, W = x.shape
        x = x.view(N * K, c_in, H, W)

        out = super(ConvZ, self).forward(x)

        NK, c_out, H_out, W_out = out.shape
        out = out.view(N, K, c_out, H_out, W_out)
        out[:, 0, :, :, :] += self.bias

        return out


class ReLUZ(nn.Module):

    def __init__(self):
        super(ReLUZ, self).__init__()
        self.relu = nn.ReLU()

    @staticmethod
    def lower_bound(x):
        return torch.abs(x[:, 0, ...]) - torch.sum(torch.abs(x[:, 1:, ...]), dim=1)

    @staticmethod
    def upper_bound(x):
        return torch.abs(x[:, 0, ...]) + torch.sum(torch.abs(x[:, 1:, ...]), dim=1)

    def weights_init(self):
        # TODO: Initialize weights here
        nn.init.constant_(self.lambdas, 0.5)

    def forward(self, x):
        # TODO: I don't know if the following is computed in parallel, if written like this
        # input is (N, K, c_in, H, W) or (N, K, fc_size)

        l, u = self.lower_bound(x), self.upper_bound(x)

        # TODO: check if broadcasting of lambdas works as expected
        l_0_u = nn.functional.relu(u) * nn.functional.relu(-l)
        out = nn.functional.relu(l) * x + l_0_u * self.lambdas * x

        out[:, 0, ...] -= (self.lambdas * l / 2)[:, 0, ...]

        # TODO: this seems wrong
        return extend_Z(out, - self.lambdas * l / 2 * l_0_u)


class ReLUZConv(ReLUZ):
    def __init__(self, n_channels, height, width):
        super(ReLUZConv, self).__init__()

        self.lambdas = nn.Parameter(torch.ones([1, 1, n_channels, height, width]))
        self.lambdas.requires_grad_()


class ReLUZLinear(ReLUZ):
    def __init__(self, fc_size):
        super(ReLUZLinear, self).__init__()

        self.lambdas = nn.Parameter(torch.ones([1, 1, fc_size]))
        self.lambdas.requires_grad_()


