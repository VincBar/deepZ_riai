import torch
import torch.nn as nn
from torch.nn.modules.flatten import Flatten
import numpy as np


def lower_bound(x):
    return x[:, 0, ...] - torch.sum(torch.abs(x[:, 1:, ...]), dim=1)


def upper_bound(x):
    return x[:, 0, ...] + torch.sum(torch.abs(x[:, 1:, ...]), dim=1)


def pad_K_dim(x, pad):
    """
    Extend the K dimension of input x by pad
    :param x:
    :param pad:
    :return:
    """
    padding = [0] * len(x.shape) * 2
    padding[-3] = pad
    return nn.functional.pad(x, padding, mode='constant', value=0)


def extend_Z(x, vals):
    """
    Extend the K dimension of input x by the number of ReLU's put on an affine layer in the original NN and update the
    values in the K dim by vals. (see j=K+i and j=else in case distinction in 2.2 in project paper).
    :param x:
    :param vals: one-dimensional tensor
    :return:
    """
    K = x.shape[1]
    pad = np.prod(x.shape[2:])
    x = pad_K_dim(x, pad)

    if isinstance(vals, float):
        vals *= torch.ones(pad)

    # TODO: check this!!
    x[:, K:, ...] = torch.diagflat(vals).view([1, pad] + list(x.shape[2:]))
    return x


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


class ZModule(nn.Module):
    def initialize(self):
        for layer in self.layers:
            if isinstance(layer, ReLUZ):
                layer.lambdas = nn.init.constant_(layer.lambdas, 0.5)


class NNFullyConnectedZ(ZModule):
    """
    Copy of FullyConnected Module for the image. This module builds up the corresponding network for the Zonotope. This is done
    by just replacing all modules by the modules adapted to passing Zonotopes (transformers). We replace ReLU and Linear
    and introduce a new layer ToZ. Normalization was not replaced as it is a constant operation.
    """
    def __init__(self, device, input_size, fc_layers, eps, target):
        super(NNFullyConnectedZ, self).__init__()

        layers = [Normalization(device), ToZ(eps), Flatten(start_dim=2)]
        prev_fc_size = input_size * input_size
        for i, fc_size in enumerate(fc_layers):
            layers += [LinearZ(prev_fc_size, fc_size)]
            if i + 1 < len(fc_layers):
                layers += [ReLUZLinear(fc_size)]
            prev_fc_size = fc_size
        layers += [EndLayerZ(target)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class NNConvZ(ZModule):
    """
    Copy of Conv Module for the image. This module builds up the corresponding network for the Zonotope. This is done
    by just replacing all modules by the modules adapted to passing Zonotopes (transformers). We replace ReLU, Conv2d,
    Linear and introduce a new layer ToZ. Normalization was not replaced as it is a constant operation.
    """
    def __init__(self, device, input_size, conv_layers, fc_layers, eps, target, n_class=10):
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
        layers += [EndLayerZ(target)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ToZ(nn.Module):
    """
    This layer takes an input tensor of shape (N, ...) and inserts the K dimension for the initial zonotope of the image
    with perturbation eps. The output will be (N, K, ...) where K = nr of input nodes (fc_size, height * width).
    """
    def __init__(self, eps):
        super(ToZ, self).__init__()
        self.eps = eps

    def forward(self, x):
        return extend_Z(x[:, None, ...], self.eps)


class LinearZ(nn.Linear):
    """
    This layer replaces the linear layer in the original NN. It does so by computing the original linear layer for each
    entry in the K dim. This is achieved by treating each entry in the K dim as a seperate member of a batch by folding
    the K dim into the batch dim. This should leverage PyTorch's parallel computation.
    """

    def forward(self, x):
        # input (N, K, input_size)
        N, K, input_size = x.shape
        x = x.view(N * K, input_size)
        out = super(LinearZ, self).forward(x).view(N, K, -1)
        out[:, 0, :] += self.bias

        return out


class ConvZ(nn.Conv2d):
    """
    This layer replaces the conv2d layer in the original NN. It does so by computing the original conv2d layer for each
    entry in the K dim. This is achieved by treating each entry in the K dim as a seperate member of a batch by folding
    the K dim into the batch dim. This should leverage PyTorch's parallel computation.
    """

    def forward(self, x):
        # TODO: check whether we really need to fold in *and* out, can't we do all of this in the batch dimension?

        N, K, c_in, H, W = x.shape
        x = x.view(N * K, c_in, H, W)

        out = super(ConvZ, self).forward(x)

        NK, c_out, H_out, W_out = out.shape
        out = out.view(N, K, c_out, H_out, W_out)
        out[:, 0, :, :, :] += self.bias

        return out


class EndLayerZ(nn.Module):
    """
    This layer computes the difference between the pseudo-probability outputs for all digits and the target digit.
    """

    def __init__(self, target):
        super(EndLayerZ, self).__init__()
        self.target = target

    def forward(self, x):
        N, K, size = x.shape

        out = torch.zeros([N, size])
        for i in range(size):
            if i == self.target:
                continue

            out[..., i] = lower_bound(x[..., self.target] - x[..., i])

        return out


class ReLUZ(nn.Module):
    """
    This layer replaces the ReLU layer. Contrarily, to the ReLU layer it has learnable parameters (lambdas) such that
    we need to subclass it for the proper definition of lambdas.shape depending on whether it connects to a LinearZ or
    to a ConvZ.

    ReLUZ extends the K dimension. I don't know if we can bound the behaviour because especially in the ReLUZConv case
    this can lead to extremely large K dim sizes. On the other hand, I don't know how much of a computational burden
    this is.
    """
    def __init__(self):
        super(ReLUZ, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        # TODO: I don't know if the following is computed in parallel, if written like this
        # input is (N, K, c_in, H, W) or (N, K, fc_size)

        l, u = lower_bound(x), upper_bound(x)
        _l = l.new(np.heaviside(l.numpy(), 0))
        l_0_u = u.new(np.heaviside(u.numpy(), 0)) * l.new(np.heaviside((-l).numpy(), 0))

        # TODO: check if broadcasting of lambdas works as expected
        out = _l * x + l_0_u * self.lambdas * x
        out[:, 0, ...] -= (self.lambdas * l / 2)[:, 0, ...]

        return extend_Z(out, (- self.lambdas * l / 2 * l_0_u).flatten(start_dim=0))


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


class PairwiseLoss(nn.Module):
    def __init__(self, net, trained_digit):
        super(PairwiseLoss, self).__init__()
        self.net = net
        self.trained_digit = trained_digit

    def forward(self, x):
        return - self.net(x)[..., self.trained_digit]


class GlobalLoss(nn.Module):
    def __init__(self, net):
        super(GlobalLoss, self).__init__()
        self.net = net

    def forward(self, x):
        return - torch.sum(self.net(x), dim=1)
