import argparse
import torch
import time

import sys
sys.path.insert(0, '..')

from networks import FullyConnected, Conv, NNFullyConnectedZ, NNConvZ, PairwiseLoss, GlobalLoss, WeightFixer, \
    check_lambdas, ClipLambdas
from time import strftime, gmtime
from collections import OrderedDict
import numpy as np

use_cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda" if use_cuda else "cpu")
DEVICE = torch.device("cpu")
INPUT_SIZE = 28
NET_CHOICES = ['fc1', 'fc2', 'fc3', 'fc4', 'fc5', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5']


def check_weights(net_name, netZ):
    state_dict_check = torch.load('../mnist_nets/%s.pt' % net_name, map_location=torch.device(DEVICE))
    for key, val in netZ.state_dict().items():
        pre, nr, param = key.split('.')
        nr = str(int(nr) - 2)
        if param != 'lambdas':
            print(param,": All net parameters equal to original value ?",
                  torch.all(state_dict_check['.'.join([pre, nr, param])] == val))
            print(param,": Net parameter requires gradient ?", val.requires_grad)


def analyze(net, inputs, true_label, eps, pairwise=True, tensorboard=True, maxsec=None, time_info=False,
            global_init=False):

    if tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        tim = strftime("%Y-%m-%d-%H_%M_%S", gmtime())


    if pairwise:
        

        trained_digits = non_verified_digits = set(range(10)) - {true_label}
        losses = dict([(i, PairwiseLoss(trained_digit=i)) for i in trained_digits])
        start_time = time.time()
        in_time = True

        # initialize lambdas,
        # TODO: do we restart from scratch for each digit? we could try warm starting, maybe for 'similar' digits
        # the amount of test cases where we know the result is not sufficient
        # TODO: search good initialization
        # we take minimum area slopes which is in the setting of non-comparable settings reasonable
        net.initialize(inputs, eps)


        inputs[inputs < eps] = (inputs[inputs < eps] + eps) / 2
        inputs[inputs > (1 - eps)] = (1 + inputs[inputs > (1 - eps)] - eps) / 2

        # run an initialization based on the global loss
        # if False:
        #     # TODO: think hard about this one, we want to avoid local minima
        #     # the approach seems to be sufficient
        #     # optimizer = torch.optim.Adam(net.parameters(), lr=0.05)
        #     optimizer = (torch.optim.Adam, {'lr': 0.05})
        #     loss = GlobalLoss()

        #     writer = None
        #     if tensorboard:
        #         writer = SummaryWriter('../runs/global_init_' + tim)
        #     res, out = run_optimization(net, inputs, loss, optimizer, writer=writer, maxsec=maxsec)

        #     # non_verified_digits -= set(np.where(torch.logical_not(loss.non_verified_digits))[0])
        #     non_verified_digits = set(np.where(loss.non_verified_digits)[0])


        while not not non_verified_digits and in_time:
            optimizer = torch.optim.Adam(net.parameters(), lr=0.02)
            i = list(non_verified_digits)[0]

            writer = None
            if tensorboard:
                writer = SummaryWriter('../runs/pairwise_' + tim + '_digit' + str(i))

            remaining_time = None
            if maxsec is not None:
                remaining_time = maxsec - (time.time() - start_time)

            losses[i].non_verified = list(non_verified_digits)
            res, out = run_optimization(net, inputs, losses[i], optimizer, writer=writer, maxsec=remaining_time)

            # check which digits can be cancelled
            # print(np.where(out > 0)[0], non_verified_digits)
            non_verified_digits -= set(np.where(out > 0)[0])

            if maxsec is not None:
                in_time = (time.time() - start_time) < maxsec

    else:

        optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
        loss = GlobalLoss()
        start_time = time.time()
        net.initialize(inputs, eps)

        inputs[inputs < eps] = (inputs[inputs < eps] + eps) / 2
        inputs[inputs > (1 - eps)] = (1 + inputs[inputs > (1 - eps)] - eps) / 2
        
        writer = None
        if tensorboard:
            writer = SummaryWriter('../runs/global_' + tim)
        res, out = run_optimization(net, inputs, loss, optimizer, writer=writer, maxsec=maxsec)

    if time_info:
        return res, time.time() - start_time

    return res


def run_optimization(net, inputs, loss, optimizer, writer=None, maxsec=None):

    orig_optimizer = None
    if type(optimizer) is tuple:
        orig_optimizer, opt_args = optimizer
        optimizer = orig_optimizer(net.parameters(), **opt_args)

    is_verified = False
    counter = 0

    start_time = time.time()

    in_time = True
    continue_ = True
    clipper = ClipLambdas()
    while not is_verified and in_time and continue_:
        counter += 1
        net.zero_grad()

        out = net(inputs)
        lss, is_verified, continue_, reset_optim = loss(out)

        # print(lss, loss, out, id(optimizer))
        if not is_verified:
            if reset_optim and orig_optimizer:
                # opt_args['lr'] *= lr_lr
                optimizer = orig_optimizer(net.parameters(), **opt_args)

            lss.backward()
            optimizer.step()
            net.apply(clipper)

        # assert check_lambdas(net)

        if writer is not None:
            writer.add_scalar('training loss', lss, counter)

        if maxsec is not None:
            in_time = (time.time() - start_time) < maxsec

    if not in_time or not continue_:
        return 0, out

    return 1, out


def fix_weights(net):
    fixer = WeightFixer()
    net = net.apply(fixer)

    return net


def loadZ(net, state_dict):
    # there are two more layers in netZ (ToZ), so shift layer names.
    state_dict_shifted = OrderedDict([])
    for key, val in state_dict.items():
        pre, nr, param = key.split('.')
        nr = str(int(nr) + 2)
        state_dict_shifted['.'.join([pre, nr, param])] = val

    net.load_state_dict(state_dict_shifted, strict=False)
    net = fix_weights(net)
    return net


def _generate_nets(typ, eps, true_label, device, *args, **kwargs):
    if typ == 'fc':
        net = FullyConnected(device, *args, **kwargs).to(device)
        netZ = NNFullyConnectedZ(device, *args, **kwargs, eps=eps, target=true_label).to(device)
    elif typ == 'conv':
        net = Conv(device, *args, **kwargs).to(device)
        netZ = NNConvZ(device, *args, **kwargs, eps=eps, target=true_label).to(device)
    else:
        raise ValueError

    return net, netZ


def load_net(net_name, eps, target):
    if net_name == 'fc1':
        net, netZ = _generate_nets('fc', eps, target, DEVICE, INPUT_SIZE, [100, 10])
    elif net_name == 'fc2':
        net, netZ = _generate_nets('fc', eps, target, DEVICE, INPUT_SIZE, [50, 50, 10])
    elif net_name == 'fc3':
        net, netZ = _generate_nets('fc', eps, target, DEVICE, INPUT_SIZE, [100, 100, 10])
    elif net_name == 'fc4':
        net, netZ = _generate_nets('fc', eps, target, DEVICE, INPUT_SIZE, [100, 100, 100, 10])
    elif net_name == 'fc5':
        net, netZ = _generate_nets('fc', eps, target, DEVICE, INPUT_SIZE, [400, 200, 100, 100, 10])
    elif net_name == 'conv1':
        net, netZ = _generate_nets('conv', eps, target, DEVICE, INPUT_SIZE, [(32, 4, 2, 1)], [100, 10], 10)
    elif net_name == 'conv2':
        net, netZ = _generate_nets('conv', eps, target, DEVICE, INPUT_SIZE, [(32, 4, 2, 1), (64, 4, 2, 1)],
                                   [100, 10], 10)
    elif net_name == 'conv3':
        net, netZ = _generate_nets('conv', eps, target, DEVICE, INPUT_SIZE,
                                   [(32, 3, 1, 1), (32, 4, 2, 1), (64, 4, 2, 1)], [150, 10], 10)
    elif net_name == 'conv4':
        net, netZ = _generate_nets('conv', eps, target, DEVICE, INPUT_SIZE, [(32, 4, 2, 1), (64, 4, 2, 1)],
                                   [100, 100, 10], 10)
    elif net_name == 'conv5':
        net, netZ = _generate_nets('conv', eps, target, DEVICE, INPUT_SIZE,
                                   [(16, 3, 1, 1), (32, 4, 2, 1), (64, 4, 2, 1)], [100, 100, 10], 10)

    state_dict = torch.load('../mnist_nets/%s.pt' % net_name, map_location=torch.device(DEVICE))
    net.load_state_dict(state_dict)
    netZ = loadZ(netZ, state_dict)

    return net, netZ


def main():
    parser = argparse.ArgumentParser(description='Neural network verification using DeepZ relaxation')
    parser.add_argument('--net', type=str, choices=NET_CHOICES, required=True, help='Neural network to verify.')
    parser.add_argument('--spec', type=str, required=True, help='Test case to verify.')
    args = parser.parse_args()

    with open(args.spec, 'r') as f:
        lines = [line[:-1] for line in f.readlines()]
        true_label = int(lines[0])
        pixel_values = [float(line) for line in lines[1:]]
        eps = float(args.spec[:-4].split('/')[-1].split('_')[-1])

    net, netZ = load_net(args.net, eps, true_label)

    inputs = torch.FloatTensor(pixel_values).requires_grad_(False).view(1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
    outs = net(inputs)
    pred_label = outs.max(dim=1)[1].item()
    assert pred_label == true_label

    # torch.set_printoptions(linewidth=300, edgeitems=5)
    # start_time = time.time()

    if analyze(netZ, inputs, true_label, eps, pairwise=True, maxsec=None, tensorboard=False, global_init=True):
        print('verified')
        # print(time.time() - start_time)
    else:
        print('not verified')

    # check_weights(args.net, netZ)
    # check_lambdas(net)


if __name__ == '__main__':
    # torch.autograd.set_detect_anomaly(True)

    main()
