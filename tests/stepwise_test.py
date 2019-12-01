import torch
import sys
sys.path.append('D:/Dokumente/GitHub/RAI_proj/code_nn')
from code.networks import NNFullyConnectedZ,PairwiseLoss
from code.verifier import loadZ, analyze,run_optimization
from collections import OrderedDict
DEVICE='cpu'
import numpy as np


def main():
    nu=0.060
    weight_l1 = np.array([[-1, 2,-1,0], [1, 1,0.8,-1],[-1,1,-1,0], [2, -2,0,-1]])
    bias_l1 = np.array([0,0,1,0])

    weight_l2 = np.array([[-2, 1,0 ,-2], [3,1,2,0]])
    bias_l2 = np.array([0,-20])

    label=0
    netZ = NNFullyConnectedZ(DEVICE, 2, [4,2], nu, label).to(DEVICE)
    state_dict=OrderedDict()
    state_dict["layers.4.weight"]=torch.Tensor(weight_l1).double().to(DEVICE)
    state_dict["layers.4.bias"]=torch.Tensor(bias_l1).double().to(DEVICE)

    state_dict["layers.6.weight"]=torch.Tensor(weight_l2).double().to(DEVICE)
    state_dict["layers.6.bias"]=torch.Tensor(bias_l2).double().to(DEVICE)
    for key, val in state_dict.items():
        pre, nr, param = key.split('.')
        state_dict['.'.join([pre, nr, param])] = val.requires_grad_(False)
    print(state_dict)
    netZ.load_state_dict(state_dict,strict=False)
    print(netZ.layers[6].weight)

    netZ=netZ.double().to(DEVICE)
    x_00=0.3
    x_01=-0.4
    x_10=-0.9
    x_11=0.1

    sigma= 0.3081
    inputs = torch.from_numpy(np.array([[x_00, x_01], [x_10, x_11]])).double().to(DEVICE)
    x=netZ.layers[0](inputs)
    print(x)
    print("mu",0.1307,"sigma", sigma)
    x=netZ.layers[1](x)
    print(x)
    x=netZ.layers[2](x)
    print(x)
    x=netZ.layers[3](x)
    print(x)
    print(nu/sigma)
    x=netZ.layers[4](x)

    print(x)
    x=netZ.layers[5](x)
    print(x)
    x=netZ.layers[6](x)
    print(x)
    x=netZ.layers[7](x)
    print(x)

    optimizer = torch.optim.Adam(netZ.parameters(), lr=0.1)
    trained_digits = non_verified_digits = set(range(2)) - {label}
    losses = dict([(i, PairwiseLoss( trained_digit=i)) for i in trained_digits])
    while not not non_verified_digits:
        i = list(non_verified_digits)[0]
        res = run_optimization(netZ, inputs, losses[i], optimizer)
        non_verified_digits -= {i}
        with torch.no_grad():
            x = netZ(inputs)
            print(x)

if __name__ == '__main__':
    main()
