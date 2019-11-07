import torch
import torch.nn as nn
import collections

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

        layers = [Normalization(device), nn.Flatten()]
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
        layers += [nn.Flatten()]

        prev_fc_size = prev_channels * img_dim * img_dim
        for i, fc_size in enumerate(fc_layers):
            layers += [nn.Linear(prev_fc_size, fc_size)]
            if i + 1 < len(fc_layers):
                layers += [nn.ReLU()]
            prev_fc_size = fc_size
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ConvZ(nn.Module):
    def __init__(self, weight, bias, prev_channels, n_channels, kernel_size, stride, padding):
        super(ConvZ, self).__init__()

        self.conv2d = nn.Conv2d(prev_channels, n_channels, kernel_size, stride=stride, padding=padding)

        # load weights, make sure they are fixed
        state_dict = collections.OrderedDict([('weight', nn.Parameter(weight, requires_grad=False)),
                                              ('bias', nn.Parameter(torch.tensor([0.0]), requires_grad=False))])
        self.bias = bias
        self.conv2d.load_state_dict(state_dict)

    def forward(self, x):
        # TODO: check this
        # input is (N, c_in, K, H, W) want to apply same conv2 for each k in range(K)
        # easy solution: just fold all K in batch dimension, and leverage pytorch's threaded computation of multiple
        # batch members. Btw in our setting, we always assume N=1.
        # We are essentially treating each epsilon layer as an image. However, do not treat these as batch members, as
        # we need to increase K in the ReLU
        # TODO: check whether we really need to fold in *and* out, can't we do all of this in the batch dimension?

        K = x.shape[2]
        N = x.shape[0]
        x = x.view(N * K, x.shape[1], x.shape[3], x.shape[4])

        out = self.conv2d(x)

        out = out.view(N, out.shape[1], K, out.shape[2], out.shape[3], o)
        out[:, :, 0, :, :] += self.bias

        return out


class ReLUZ(nn.Module):
    def __init__(self, n_channels, height, width):
        super(ReLUZ).__init__()

        self.lambdas = nn.Parameter(torch.tensor([1, n_channels, 1, height, width]))
        self.relu = nn.ReLU()

    @staticmethod
    def lower_bound(x):
        return torch.abs(x[:, :, 0, :, :]) - torch.sum(torch.abs(x[:, :, 1:, :, :]), dim=2)

    @staticmethod
    def upper_bound(x):
        return torch.abs(x[:, :, 0, :, :]) + torch.sum(torch.abs(x[:, :, 1:, :, :]), dim=2)

    def forward(self, x):
        # TODO: I don't know if this is computed in parallel, if written like this
        l, u = self.lower_bound(x), self.upper_bound(x)

        # TODO: check if broadcasting of lambdas works as expected
        l_0_u = nn.functional.relu(u) * nn.functional.relu(u)
        out = nn.functional.relu(l) * x + l_0_u * self.lambdas * x
        out[:, :, 0, :, :] -= self.lambdas[:, :, 0, :, :] * x / 2

        width = x.shape[-1]
        height = x.shape[-2]
        K = x.shape[2]
        out = nn.functional.pad(input=out, pad=(0, 0, 0, 0, 0, width * height, 0, 0, 0, 0), mode='constant', value=0)

        out[:, :, K:, :, :] = - self.lambdas * l / 2 * l_0_u

        return out



