
import torch
import sys
sys.path.append('D:/Dokumente/GitHub/RAI_proj/code')
from networks import NNFullyConnectedZ
from verifier import load_Z, analyze
from collections import OrderedDict
DEVICE='cpu'

netZ = NNFullyConnectedZ(DEVICE, 2, [2, 2], 0.999999, 1).to(DEVICE)
state_dict=OrderedDict()
state_dict["layers.3.weight"]=torch.Tensor([[1,0],[0,2],[-1,0],[0,2]]).to(DEVICE).t()
state_dict["layers.3.bias"]=torch.Tensor([0,0]).to(DEVICE)
state_dict["layers.5.weight"]=torch.Tensor([[1,-1],[-1,2]]).to(DEVICE)
state_dict["layers.5.bias"]=torch.Tensor([0,0]).to(DEVICE)
print(state_dict)

load_Z(netZ, state_dict)
inputs=torch.FloatTensor([0.9,0.5,0.2,0.8]).requires_grad_(False).view(1, 1, 2, 2).to(DEVICE)
netZ.load_state_dict(state_dict,strict=False)
inp=netZ.layers[0](inputs)
inp2=netZ.layers[1](inputs)
inp3=netZ.layers[2](inp2)
inp4=netZ.layers[3](inp3)
inp5=netZ.layers[4](inp4)
inp6=netZ.layers[5](inp5)
inp7=netZ.layers[6](inp6)

#TODO: check what does normalization do
analyze(netZ, inputs, -1, pairwise=True)
