import sys
import torch
import torch.nn as nn

sys.path.append("D:/Dokumente/GitHub/RAI_proj/code_nn/")
from networks import FullyConnected, Conv, NNFullyConnectedZ

DEVICE="cpu"
INPUT_SIZE=28

with open("D:/Dokumente/Universitaet/Statistik/ML/ReliableAI/riai2019_project/test_cases/fc1/img0_0.06000.txt",
          'r') as f:
    lines = [line[:-1] for line in f.readlines()]
    true_label = int(lines[0])
    pixel_values = [float(line) for line in lines[1:]]
    eps = float(0.06000)


net = FullyConnected(DEVICE, INPUT_SIZE, [100, 10]).to(DEVICE)
netZ = NNFullyConnectedZ(DEVICE, INPUT_SIZE, [100, 10]).to(DEVICE)

net.load_state_dict(torch.load('D:/Dokumente/GitHub/RAI_proj/mnist_nets/%s.pt' % "fc1", map_location=torch.device(DEVICE)))
netZ.load_state_dict(torch.load('D:/Dokumente/GitHub/RAI_proj/mnist_nets/%s.pt' % "fc1", map_location=torch.device(DEVICE)), strict=False)

netZ = NNFullyConnectedZ(DEVICE, INPUT_SIZE, [100, 10]).to(DEVICE)
x = torch.FloatTensor(pixel_values).view(1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)


#outs = net(inputs)
#pred_label = outs.max(dim=1)[1].item()
#assert pred_label == true_label
import numpy as np
x = torch.FloatTensor(pixel_values).to(DEVICE).view(INPUT_SIZE*INPUT_SIZE,1)
inputs= torch.FloatTensor(np.diag(np.ones(INPUT_SIZE*INPUT_SIZE)*eps))


out=netZ.layers[2](inputs)

out.shape
res1=netZ.layers[3](out)

