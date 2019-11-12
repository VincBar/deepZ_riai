import torch
import torch.nn as nn
import collections
from torch.nn.modules.flatten import Flatten


class Normalization(nn.Module):

    def __init__(self, device):
        super(Normalization, self).__init__()
        self.mean = torch.FloatTensor([0.1307]).view((1, 1, 1, 1)).to(device)
        self.sigma = torch.FloatTensor([0.3081]).view((1, 1, 1, 1)).to(device)

    def forward(self, x):
        return (x - self.mean) / self.sigma


class NormalizationZ(Normalization):

    def __init__(self, device):
        super(Normalization, self).__init__()
        self.mean = torch.FloatTensor([0.1307]).view((1, 1, 1, 1, 1)).to(device)
        self.sigma = torch.FloatTensor([0.3081]).view((1, 1, 1, 1, 1)).to(device)


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
    """
    Copy of fullyConnected Module for the image. This module builds up the corresponding network for the Zonotope.
    """
    def __init__(self, device, input_size, fc_layers):
        super(NNFullyConnectedZ, self).__init__()

        layers = [Normalization(device), Flatten(start_dim=2)]
        prev_fc_size = input_size * input_size
        for i, fc_size in enumerate(fc_layers):
            layers += [LinearZ(prev_fc_size, fc_size)]
            if i + 1 < len(fc_layers):
                #why fc_size
                layers += [ReLUZLinear(fc_size)]
            prev_fc_size = fc_size
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class NNConvZ(nn.Module):
    """
    Copy of Conv Module for the image. This module builds up the corresponding network for the Zonotope.
    """
    def __init__(self, device, input_size, conv_layers, fc_layers, n_class=10):
        super(NNConvZ, self).__init__()

        self.input_size = input_size
        self.n_class = n_class

        layers = [Normalization(device)]
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
        out = x.matmul(self.weight.t())
        out[0, :] += self.bias
        # fix return out ?
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

    @staticmethod
    def lower_bound(x):
        return x[0, :] - torch.sum(torch.abs(x[1:, :]), dim=0)

    #bugfix, we dont want the absolut value of x[0]
    @staticmethod
    def upper_bound(x):
        return x[0, :] + torch.sum(torch.abs(x[1:, :]), dim=0)

    def weights_init(self):
        # TODO: Initialize weights here
        for module in self.modules():
            nn.init.normal_(module.weight, mean=0, std=1)
            nn.init.constant_(module.bias, 0)


class ReLUZConv(ReLUZ):
    def __init__(self, n_channels, height, width):
        super(ReLUZConv, self).__init__()

        self.lambdas = nn.Parameter(torch.ones([1, n_channels, 1, height, width]))
        self.lambdas.requires_grad_()

        self.relu = nn.ReLU()

    def forward(self, x):
        # input is (N, K, c_in, H, W)
        N, K, c_in, H, W = x.shape

        # TODO: I don't know if this is computed in parallel, if written like this
        l, u = self.lower_bound(x), self.upper_bound(x)

        # TODO: check if broadcasting of lambdas works as expected
        l_0_u = nn.functional.relu(u) * nn.functional.relu(-l)
        out = nn.functional.relu(l) * x + l_0_u * self.lambdas * x
        out[:, 0, :, :, :] -= self.lambdas[:, 0, :, :, :] * x / 2

        out = nn.functional.pad(input=out, pad=(0, 0, 0, W * H, 0, 0, 0, 0, 0, 0), mode='constant', value=0)

        out[:, K:, :, :, :] = - self.lambdas * l / 2 * l_0_u

        return out


class ReLUZLinear(ReLUZ):
    def __init__(self, fc_size):
        super(ReLUZLinear, self).__init__()

        self.lambdas = nn.Parameter(torch.eye(fc_size))
        self.lambdas.requires_grad_()

        self.relu = nn.ReLU()

    def forward(self, x):
        # input is (N, K, fc_size)
        #N, K, fc_size = x.shape

        # TODO: I don't know if this is computed in parallel, if written like this
        l, u = self.lower_bound(x), self.upper_bound(x)

        # # TODO: check if broadcasting of lambdas works as expected
        l_0_u = torch.le(torch.mul(l,u),0)
        u_0 = torch.ge(u,0)

        out = x - torch.mul(x,l_0_u) + x.matmul(self.lambdas)/2
        out[0,:]= - self.lambdas.matmul(l)/2

        # out[:, :, 0] -= self.lambdas[:, :, 0] * x / 2
        #
        # fc_size = x[1]
        # K = x.shape[2]
        # out = nn.functional.pad(input=out, pad=(0, 0, 0, fc_size, 0, 0), mode='constant', value=0)
        #
        # out[:, K:, :] = - self.lambdas * l / 2 * l_0_u

        return out



