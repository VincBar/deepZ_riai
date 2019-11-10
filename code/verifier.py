import argparse
import torch
from networks import FullyConnected, Conv, NNFullyConnectedZ
import numpy as np
from collections import OrderedDict

DEVICE = 'cpu'
INPUT_SIZE = 28


def analyze(net, inputs, true_label):
    net(inputs)
    return 0


def load_and_initialize_Z(net, state_dict):
    # there is one layer more in netZ (ToZ), so shift layer names.
    state_dict_shifted = OrderedDict([('layers.' + str(int(key.split('.')[1]) + 1) + '.' + key.split('.')[2],
                                       val.requires_grad_()) for key, val in state_dict.items()])
    net.load_state_dict(state_dict_shifted, strict=False)

    # Freeze weights and biases
    for param in state_dict_shifted.keys():
        obj = net
        for attr in param.split('.'):
            try:
                attr = int(attr)
                obj = obj[attr]
            except ValueError:
                obj = getattr(obj, attr)

        obj.requires_grad = False

    # initialize lambdas
    net.initialize()


def main():
    parser = argparse.ArgumentParser(description='Neural network verification using DeepZ relaxation')
    parser.add_argument('--net',
                        type=str,
                        choices=['fc1', 'fc2', 'fc3', 'fc4', 'fc5', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5'],
                        required=True,
                        help='Neural network to verify.')
    parser.add_argument('--spec', type=str, required=True, help='Test case to verify.')
    args = parser.parse_args()

    with open(args.spec, 'r') as f:
        lines = [line[:-1] for line in f.readlines()]
        true_label = int(lines[0])
        pixel_values = [float(line) for line in lines[1:]]
        eps = float(args.spec[:-4].split('/')[-1].split('_')[-1])

    if args.net == 'fc1':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 10]).to(DEVICE)
        netZ = NNFullyConnectedZ(DEVICE, INPUT_SIZE, [100, 10], eps).to(DEVICE)
    elif args.net == 'fc2':
        net = FullyConnected(DEVICE, INPUT_SIZE, [50, 50, 10]).to(DEVICE)
    elif args.net == 'fc3':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 10]).to(DEVICE)
    elif args.net == 'fc4':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 10]).to(DEVICE)
    elif args.net == 'fc5':
        net = FullyConnected(DEVICE, INPUT_SIZE, [400, 200, 100, 100, 10]).to(DEVICE)
    elif args.net == 'conv1':
        net = Conv(DEVICE, INPUT_SIZE, [(32, 4, 2, 1)], [100, 10], 10).to(DEVICE)
    elif args.net == 'conv2':
        net = Conv(DEVICE, INPUT_SIZE, [(32, 4, 2, 1), (64, 4, 2, 1)], [100, 10], 10).to(DEVICE)
    elif args.net == 'conv3':
        net = Conv(DEVICE, INPUT_SIZE, [(32, 3, 1, 1), (32, 4, 2, 1), (64, 4, 2, 1)], [150, 10], 10).to(DEVICE)
    elif args.net == 'conv4':
        net = Conv(DEVICE, INPUT_SIZE, [(32, 4, 2, 1), (64, 4, 2, 1)], [100, 100, 10], 10).to(DEVICE)
    elif args.net == 'conv5':
        net = Conv(DEVICE, INPUT_SIZE, [(16, 3, 1, 1), (32, 4, 2, 1), (64, 4, 2, 1)], [100, 100, 10], 10).to(DEVICE)

    state_dict = torch.load('../mnist_nets/%s.pt' % args.net, map_location=torch.device(DEVICE))
    net.load_state_dict(state_dict)
    load_and_initialize_Z(netZ, state_dict)

    inputs = torch.FloatTensor(pixel_values).view(1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
    outs = net(inputs)
    pred_label = outs.max(dim=1)[1].item()
    assert pred_label == true_label

    if analyze(netZ, inputs, true_label):
        print('verified')
    else:
        print('not verified')


if __name__ == '__main__':
    main()
