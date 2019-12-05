import torch
import torch.nn as nn
from torch.nn.modules.flatten import Flatten
import numpy as np


def lower_bound(x):
    return x[0, ...] - torch.sum(torch.abs(x[1:, ...]), dim=0)


def upper_bound(x):
    return x[0, ...] + torch.sum(torch.abs(x[1:, ...]), dim=0)


def is_scalar(x):
    try:
        float(x)
    except ValueError:
        return False

    return True


def heaviside(a, zero_pos=False):
    """
    :param a: any dimensional pytorch tensor
    :return: 0,1 identifier if input larger 0
    """
    if zero_pos:
        a = a + np.finfo(float).eps
    return torch.relu(torch.sign(a))


def pad_K_dim(x, pad):
    """
    Extend the K dimension of input x by pad
    :param x:
    :param pad:
    :return:
    """
    padding = [0] * len(x.shape) * 2
    padding[-1] = pad
    return nn.functional.pad(x, padding, mode='constant', value=0)


def extend_Z(x, vals, l_0_u):
    """
    Extend the K dimension of input x by the number of ReLU's put on an affine layer in the original NN and update the
    values in the K dim by vals. (see j=K+i and j=else in case distinction in 2.2 in project paper).
    :param x:
    :param vals: one-dimensional tensor
    :return:
    """
    K = x.shape[0]
    if is_scalar(vals):
        pad2 = np.prod(x.shape[1:])
        vals = vals * torch.ones(pad2)

    # TODO: check this!!

    x[K:, ...] = vals[l_0_u.bool().flatten(), ...]
    return x


class ClipLambdas(object):
    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'lambdas'):
            lambdas = module.lambdas.data
            lambdas = lambdas.clamp(0, 1)
            module.lambdas.data = lambdas


class WeightFixer(object):
    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            module.weight.requires_grad = False
        if hasattr(module, 'bias'):
            module.bias.requires_grad = False


def check_lambdas(net):
    ret = torch.tensor([True])
    for key, val in net.state_dict().items():
        pre, nr, param = key.split('.')
        if param == 'lambdas':
            up = torch.all(val <= 1)
            lo = torch.all(val >= 0)
            ret = ret & lo & up
            if not lo:
                print('lo', val[val < 0])
            if not up:
                print('up', val[val > 1])
    return ret


class Normalization(nn.Module):

    def __init__(self, device):
        super(Normalization, self).__init__()
        self.mean = torch.FloatTensor([0.1307]).view((1, 1, 1, 1)).to(device)
        self.sigma = torch.FloatTensor([0.3081]).view((1, 1, 1, 1)).to(device)

    def forward(self, x):
        return (x - self.mean) / self.sigma


class EpsNorm(nn.Module):
    def __init__(self):
        super(EpsNorm, self).__init__()
        self.sigma = 0.3081

    def forward(self, x):
        x[1:, ...] = x[1:, ...] / self.sigma
        return x


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
            layers += [nn.Conv2d(prev_channels, n_channels, kernel_size, stride=stride, padding=padding), nn.ReLU(), ]
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
    def initialize(self, inputs, eps):
        out = inputs

        for i, layer in enumerate(self.layers):
            if isinstance(layer, ToZLinear):
                inp = inputs.flatten(start_dim=1)
                layer.eps[inp < eps] = (inp[inp < eps] + eps) / 2
                layer.eps[inp > (1 - eps)] = (1 - inp[inp > (1 - eps)] + eps) / 2
            if isinstance(layer, ToZConv):
                inp = inputs
                layer.eps[inp < eps] = (inp[inp < eps] + eps) / 2
                layer.eps[inp > (1 - eps)] = (1 - inp[inp > (1 - eps)] + eps) / 2

            if isinstance(layer, ReLUZ):
                with torch.no_grad():
                    l = lower_bound(out)
                    u = upper_bound(out)

                    layer.lambdas.copy_(u / (u - l))
                    layer.lambdas.requires_grad = True

            if isinstance(layer, LinearZ) or isinstance(layer, ConvZ):
                layer.weight.requires_grad = False
                layer.bias.requires_grad = False

            out = self.layers[i](out)


class NNFullyConnectedZ(ZModule):
    """
    Copy of FullyConnected Module for the image. This module builds up the corresponding network for the Zonotope. This is done
    by just replacing all modules by the modules adapted to passing Zonotopes (transformers). We replace ReLU and Linear
    and introduce a new layer ToZ. Normalization was not replaced as it is a constant operation.
    """

    def __init__(self, device, input_size, fc_layers, eps, target):
        super(NNFullyConnectedZ, self).__init__()
        prev_fc_size = input_size * input_size

        layers = [Normalization(device), ToZLinear(eps, prev_fc_size), Flatten(start_dim=1), EpsNorm()]
        for i, fc_size in enumerate(fc_layers):
            layers += [LinearZ(prev_fc_size, fc_size)]
            if i + 1 < len(fc_layers):
                layers += [ReLUZLinear(fc_size)]
            prev_fc_size = fc_size
        layers += [EndLayerZ(target, prev_fc_size)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class NNConvZ(ZModule):
    """
    Copy of Conv Module for the image. This module builds up the corresponding network for the Zonotope. This is done
    by just replacing all modules by the modules adapted to passing Zonotopes (transformers). We replace ReLU, Conv2d,
    Linear and introduce a new layer ToZ. Normalization was not replaced as it is a constant operation.
    """

    def __init__(self, device, input_size, conv_layers, fc_layers, n_class=10, eps=0, target=0):
        super(NNConvZ, self).__init__()

        self.input_size = input_size
        self.n_class = n_class

        prev_channels = 1
        height = width = input_size

        layers = [Normalization(device), ToZConv(eps, prev_channels, height, width), EpsNorm()]

        for n_channels, kernel_size, stride, padding in conv_layers:
            height, width = self._compute_resulting_height_width(height, width, kernel_size, stride, 2 * padding)
            layers += [ConvZ(prev_channels, n_channels, kernel_size, stride=stride, padding=padding),
                       ReLUZConv(n_channels, height, width), ]
            prev_channels = n_channels
        layers += [Flatten(start_dim=1)]

        prev_fc_size = prev_channels * height * width
        for i, fc_size in enumerate(fc_layers):
            layers += [LinearZ(prev_fc_size, fc_size)]
            if i + 1 < len(fc_layers):
                layers += [ReLUZLinear(fc_size)]
            prev_fc_size = fc_size
        layers += [EndLayerZ(target, prev_fc_size)]
        self.layers = nn.Sequential(*layers)

    @staticmethod
    def _compute_resulting_height_width(height, width, kernel_size, stride, padding):
        height = (height - kernel_size + padding + stride) // stride
        width = (width - kernel_size + padding + stride) // stride

        return height, width

    def forward(self, x):
        return self.layers(x)


class ToZ(nn.Module):
    """
    This layer takes an input tensor and inserts the K dimension for the initial zonotope of the image
    with perturbation eps. The output will be (K, ...) where K = nr of input nodes (fc_size, height * width).
    """

    def __init__(self):
        super(ToZ, self).__init__()

    def forward(self, x):
        pad = np.prod(x.shape[1:])
        return extend_ToZ(x, self.eps, torch.ones([pad]))


class ToZConv(ToZ):
    def __init__(self, eps, c, h, w):
        super(ToZConv, self).__init__()
        self.eps = eps * torch.ones([1, c, h, w])
        pad=c*h*w
        self.eps = torch.diagflat(self.eps).view([pad] + [c,h,w])


class ToZLinear(ToZ):
    def __init__(self, eps, fc_size):
        super(ToZLinear, self).__init__()
        self.eps = torch.ones([1, fc_size]) * eps


class LinearZ(nn.Linear):
    """
    This layer replaces the linear layer in the original NN. It does so by computing the original linear layer for each
    entry in the K dim. This is achieved by treating each entry in the K dim as a seperate member of a batch by folding
    the K dim into the batch dim. This should leverage PyTorch's parallel computation.
    """

    def __init__(self, *args, **kwargs):
        super(LinearZ, self).__init__(*args, **kwargs)
        self.weight.requires_grad = False
        self.bias.requires_grad = False

    def forward(self, x):
        out = nn.functional.linear(x, self.weight, bias=None)
        out[0, :] += self.bias
        return out


class ConvZ(nn.Conv2d):
    """
    This layer replaces the conv2d layer in the original NN. It does so by computing the original conv2d layer for each
    entry in the K dim. This is achieved by treating each entry in the K dim as a seperate member of a batch by folding
    the K dim into the batch dim. This should leverage PyTorch's parallel computation.
    """

    def __init__(self, *args, **kwargs):
        super(ConvZ, self).__init__(*args, **kwargs)
        self.weight.requires_grad = False
        self.bias.requires_grad = False

    def forward(self, x):
        out = nn.functional.conv2d(x, weight=self.weight, bias=None, stride=self.stride, padding=self.padding)
        out[0, :, :, :] += self.bias[:, None, None]
        return out


class EndLayerZ(nn.Module):
    """
    This layer computes the difference between the pseudo-probability outputs for all digits and the target digit.
    """

    def __init__(self, target, size):
        super(EndLayerZ, self).__init__()
        self.target = target

        self.weight = torch.zeros([size, size])
        self.weight[:, target] = 1
        self.weight -= torch.diag(torch.ones(size))

    def forward(self, x):
        x = nn.functional.linear(x, self.weight, bias=None)
        out = lower_bound(x)
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
        # input is (K, c_in, H, W) or (K, fc_size)

        l, u = lower_bound(x)[None, :], upper_bound(x)[None, :]
        _l = heaviside(l)
        l_0_u = (heaviside(u) * heaviside(-l))

        d_1 = -l * self.lambdas
        d_2 = u * (1 - self.lambdas)

        # TODO: check if lambdas are bounded between [0,1]
        # check completed
        # TODO: check if broadcasting of lambdas works as expected
        # check completed see test_conv_pad

        # compute shift
        d = torch.max(d_1, d_2)

        out = _l * x + l_0_u * self.lambdas * x
        out[0, ...] += l_0_u[0, ...] * (d / 2)[0, ...]
        return

        # # TODO: I don't know if the following is computed in parallel, if written like this  # # input is (K, c_in, H, W) or (K, fc_size)  #  # l_t, u_t = lower_bound(x)[None, :], upper_bound(x)[None, :]  # _l_t = heaviside(l_t)  # l_0_u_t = (heaviside(u_t) * heaviside(-l_t))  #  # lambda_crit_t = u_t / (u_t - l_t)  # is_lower = self.lambdas < lambda_crit_t  # is_larger = torch.logical_not(is_lower)  #  # # TODO: check if lambdas are bounded between [0,1]  # # TODO: check if broadcasting of lambdas works as expected  # # check completed see test_conv_pad  #  # # compute shift  # d_t = torch.zeros(self.lambdas.shape)  # d_t[is_larger] = - l_t[is_larger]  # d_t[is_lower] = (1 - self.lambdas[is_lower])/self.lambdas[is_lower] * u_t[is_lower]  #  # out_t = _l_t * x + l_0_u_t * self.lambdas * x  # out_t[0, ...] += l_0_u_t[0, ...] * (self.lambdas * d_t / 2)[0, ...]  # torch.all(out_t==out)  #  # return extend_Z(out_t, self.lambdas * d_t/2 * l_0_u_t, l_0_u_t)


class ReLUZConv(ReLUZ):
    def __init__(self, n_channels, height, width, *args, **kwargs):
        super(ReLUZConv, self).__init__(*args, **kwargs)
        # TODO: Currently all lambdas are initialized as one.
        # Maybe the initalization can be learned number specific, smallest area
        # TODO: Only add rows that are actually relevant
        lambdas_tmp = nn.Parameter(torch.ones([1, n_channels, height, width]))
        lambdas_tmp.requires_grad_()
        pad=n_channels*height*width
        self.lambdas = torch.diagflat(lambdas_tmp).view([pad] + [n_channels,height,width])

class ReLUZLinear(ReLUZ):
    def __init__(self, fc_size, *args, **kwargs):
        super(ReLUZLinear, self).__init__(*args, **kwargs)

        lambdas_tmp = nn.Parameter(torch.ones([1, fc_size]))
        lambdas_tmp.requires_grad_()
        self.lambdas = torch.diagflat(lambdas_tmp).view([fc_size] + [fc_size])


class PairwiseLoss(nn.Module):
    def __init__(self, trained_digit):
        super(PairwiseLoss, self).__init__()
        self.trained_digit = trained_digit
        self.non_verified = [self.trained_digit]

    def forward(self, x):
        loss = - x[self.trained_digit]
        is_verified = torch.sum(heaviside(x)[torch.LongTensor(self.non_verified)]) > 0
        return loss, is_verified


class GlobalLoss(nn.Module):
    def __init__(self, reg):
        super(GlobalLoss, self).__init__()
        self.reg = reg

    def forward(self, x):
        loss = - torch.sum(x) + self.reg / x.shape[0] * torch.sum(torch.pdist(x.view((x.shape[0], 1)), p=1))
        is_verified = torch.prod(heaviside(x, zero_pos=True)).bool()

        return loss, is_verified
