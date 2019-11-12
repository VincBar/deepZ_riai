import argparse
import torch
from networks import FullyConnected, Conv, NNFullyConnectedZ, NNConvZ, PairwiseLoss, GlobalLoss, get_child
from time import strftime, gmtime
from collections import OrderedDict

DEVICE = 'cpu'
INPUT_SIZE = 28


def analyze(net, inputs, true_label, pairwise=True, tensorboard=True):

    # TODO: think hard about this one, we want to avoid local minima
    optimizer = torch.optim.Adam(net.parameters(), lr=0.1)

    if tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        time = strftime("%Y-%m-%d-%H_%M_%S", gmtime())

    if pairwise:
        trained_digits = non_verified_digits = set(range(10)) - {true_label}
        losses = dict([(i, PairwiseLoss(net, trained_digit=i)) for i in trained_digits])

        while not not non_verified_digits:
            i = list(non_verified_digits)[0]

            # initialize lambdas,
            # TODO: do we restart from scratch for each digit? we could try warm starting, maybe for 'similar' digits
            # TODO: search good initialization
            net.initialize()

            writer = None
            if tensorboard:
                writer = SummaryWriter('../runs/pairwise_' + time + '_digit' + str(i))

            res = run_optimization(net, inputs, losses[i], optimizer, writer=writer)
            non_verified_digits -= {i}

    else:
        loss = GlobalLoss(net)
        net.initialize()

        writer = None
        if tensorboard:
            writer = SummaryWriter('../runs/global_' + time)

        res = run_optimization(net, inputs, loss, optimizer, writer=writer)

    return res


def run_optimization(net, inputs, loss, optimizer, writer=None):
    not_verified = True
    counter = 0

    while not_verified:
        counter += 1
        net.zero_grad()
        lss = loss(inputs)
        lss.backward()
        optimizer.step()

        if writer is not None:
            writer.add_scalar('training loss', lss, counter)

        if - lss > 0:
            not_verified = False

    return 1


def load_Z(net, state_dict):
    # there is one layer more in netZ (ToZ), so shift layer names.
    state_dict_shifted = OrderedDict([])
    for key, val in state_dict.items():
        pre, nr, param = key.split('.')
        nr = str(int(nr) + 1)
        state_dict_shifted['.'.join([pre, nr, param])] = val.requires_grad_(False)

    net.load_state_dict(state_dict_shifted, strict=False)


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
        netZ = NNFullyConnectedZ(DEVICE, INPUT_SIZE, [100, 10], eps, true_label).to(DEVICE)
    elif args.net == 'fc2':
        net = FullyConnected(DEVICE, INPUT_SIZE, [50, 50, 10]).to(DEVICE)
        netZ = NNFullyConnectedZ(DEVICE, INPUT_SIZE, [50, 50, 10], eps, true_label).to(DEVICE)
    elif args.net == 'fc3':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 10]).to(DEVICE)
    elif args.net == 'fc4':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 10]).to(DEVICE)
    elif args.net == 'fc5':
        net = FullyConnected(DEVICE, INPUT_SIZE, [400, 200, 100, 100, 10]).to(DEVICE)
        netZ = NNFullyConnectedZ(DEVICE, INPUT_SIZE, [400, 200, 100, 100, 10], eps, true_label).to(DEVICE)
    elif args.net == 'conv1':
        net = Conv(DEVICE, INPUT_SIZE, [(32, 4, 2, 1)], [100, 10], 10).to(DEVICE)
        netZ = NNConvZ(DEVICE, INPUT_SIZE, [(32, 4, 2, 1)], [100, 10], eps, true_label, 10).to(DEVICE)
    elif args.net == 'conv2':
        net = Conv(DEVICE, INPUT_SIZE, [(32, 4, 2, 1), (64, 4, 2, 1)], [100, 10], 10).to(DEVICE)
    elif args.net == 'conv3':
        net = Conv(DEVICE, INPUT_SIZE, [(32, 3, 1, 1), (32, 4, 2, 1), (64, 4, 2, 1)], [150, 10], 10).to(DEVICE)
    elif args.net == 'conv4':
        net = Conv(DEVICE, INPUT_SIZE, [(32, 4, 2, 1), (64, 4, 2, 1)], [100, 100, 10], 10).to(DEVICE)
    elif args.net == 'conv5':
        net = Conv(DEVICE, INPUT_SIZE, [(16, 3, 1, 1), (32, 4, 2, 1), (64, 4, 2, 1)], [100, 100, 10], 10).to(DEVICE)
        netZ = NNConvZ(DEVICE, INPUT_SIZE, [(16, 3, 1, 1), (32, 4, 2, 1), (64, 4, 2, 1)], [100, 100, 10], eps, true_label,
                       10).to(DEVICE)

    state_dict = torch.load('../mnist_nets/%s.pt' % args.net, map_location=torch.device(DEVICE))
    net.load_state_dict(state_dict)
    load_Z(netZ, state_dict)

    inputs = torch.FloatTensor(pixel_values).requires_grad_(False).view(1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
    outs = net(inputs)
    pred_label = outs.max(dim=1)[1].item()
    assert pred_label == true_label

    if analyze(netZ, inputs, true_label, pairwise=True):
        print('verified')
    else:
        print('not verified')


if __name__ == '__main__':
    main()
