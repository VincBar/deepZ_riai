
import torch
import sys
sys.path.append('D:/Dokumente/GitHub/RAI_proj/code_nn')
from networks import NNFullyConnectedZ
from verifier import load_Z, analyze
from collections import OrderedDict
DEVICE='cpu'
import numpy as np
def heaviside(a, zero_pos=False):
    """
    :param a: any dimensional pytorch tensor
    :return: 0,1 identifier if input larger 0
    """
    if zero_pos:
        a = a + np.finfo(float).eps
    return max(0,np.sign(a))

def test():
    x_00=0.9
    x_01=0.5
    x_10=0.2
    x_11=-0.5
    nu=0.8
    lam_1 =1
    lam_2=1
    mu=0.1307
    s=0.3081
    weight = np.array([[1, 1], [0, 0], [-1, 0], [0, 2]])
    netZ = NNFullyConnectedZ(DEVICE, 2, [2], nu, 0).to(DEVICE)
    state_dict=OrderedDict()
    state_dict["layers.3.weight"]=torch.Tensor(weight).to(DEVICE).t()
    state_dict["layers.3.bias"]=torch.Tensor([0,0]).to(DEVICE)
    #state_dict["layers.5.weight"]=torch.Tensor([[1,-1],[-1,2]]).to(DEVICE)
    #state_dict["layers.5.bias"]=torch.Tensor([0,0]).to(DEVICE)
    #print(state_dict)



    inputs=np.array([[x_00,x_01],[x_10,x_11]])
    load_Z(netZ, state_dict)
    inputs=torch.FloatTensor(inputs).requires_grad_(False).view(1, 1, 2, 2).to(DEVICE)
    netZ.load_state_dict(state_dict,strict=False)
    inp=netZ.layers[0](inputs)
    inp2=netZ.layers[1](inp)
    inp3=netZ.layers[2](inp2)
    inp4=netZ.layers[3](inp3)
    inp5=netZ.layers[4](inp4)
    #inp6=netZ.layers[5](inp5)
    #inp7=netZ.layers[6](inp6)
    print(inp5)

    #TODO: check what does normalization do
    #analyze(netZ, inputs, -1, pairwise=True)



    inputs_h=np.array([[x_00,x_01],[x_10,x_11]])
    inp_h= (inputs_h-mu)/s
    inp2_h=np.append(inp_h.reshape(1,-1),np.eye(4)*nu,axis=0)
    inp3_h=np.matmul(inp2_h,weight)
    l_1=inp3_h[0][0]-np.sum(np.abs(inp3_h.transpose()[0][1:]))
    l_2=inp3_h[0][1]-np.sum(np.abs(inp3_h.transpose()[1][1:]))
    u_1=inp3_h[0][0]+np.sum(np.abs(inp3_h.transpose()[0][1:]))
    u_2=inp3_h[0][1]+np.sum(np.abs(inp3_h.transpose()[1][1:]))

    trafo=np.eye(2)
    trafo[0,0]=heaviside(u_1)
    trafo[1,1]=heaviside(u_2)
    trafo2=np.eye(2)
    trafo2[0][0]=heaviside(-l_1*u_1)
    trafo2[1][1]=heaviside(-l_2*u_2)
    print(inp4)
    print(inp3_h)
    inp3_h[0,0]+=heaviside(-l_1*u_1)*(-inp3_h[0,0]-(lam_1*(-inp3_h[0,0]+l_1/2)))
    inp3_h[0,1]+=heaviside(-l_2*u_2)*(-inp3_h[0,1]-(lam_2*(-inp3_h[0,1]+l_2/2)))
    inp4_h=np.append(inp3_h,trafo2,axis=0)
    inp4_h=np.matmul(inp4_h,trafo)
    print(inp5,inp4_h)

    inp5_h=np.matmul(inp4_h,np.array([1,-1]))
    #print(inp5_h)
    #print(inp5_h[0]-np.sum(np.abs(inp5_h[1:])))
    print(inp3)
    print(inp2_h)

    # #hand calc:
    # import numpy as np
    # res=np.array([lam_1*((x_00-x_10)/(2*s)-nu)*heaviside((x_00-x_10)/s+2*nu)-lam_2*((x_00-mu)/s+2*(x_00-mu)/2-3/2*nu)*heaviside((x_00-mu)/s+2*(x_11-mu)/s+3*nu),1*heaviside((x_00-x_10)/s+2*nu)-1*heaviside((x_00-mu)/s+2*(x_11-mu)/s+3*nu),0,-1*heaviside((x_00-x_10)/s+2*nu),-2*heaviside(0,(x_00-mu)/s+2*(x_11-mu)/s+3*nu),-lam_1*((x_00-x_10)/s-2*nu)/2*heaviside(0,(x_00-x_10)/s+2*nu),-lam_2*((x_00-mu)/s+2*(x_11-mu)/s-3*nu)/2*heaviside(0,(x_00-mu)/s+2*(x_11-mu)/s+3*nu)])
    # print(res[0]-np.sum(np.abs(res[1:])))


if __name__ == '__main__':
    test()

