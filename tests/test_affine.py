import torch
import sys
sys.path.append('D:/Dokumente/GitHub/RAI_proj/code_nn')
from code.networks import LinearZ

# Let x the current zonotope. In the fully connected case x.shape = (K, fc_size).
# Now, we want to transform x with the weight matrix b into x_out with x_out.shape = (K,fc_size_out)


def test():
    K = 3
    fc_size = 5
    fc_size_out = 4
    x = torch.ones((K, fc_size)).double()
    l = LinearZ(fc_size,fc_size_out)
    l.weight = torch.nn.Parameter(torch.arange(fc_size*fc_size_out).double().view((fc_size_out,fc_size)),requires_grad=False)
    l.bias = torch.nn.Parameter(torch.ones(fc_size_out))
    #TODO: Check if direct imp. difers from functional
    x_out=l(x)
    x_check=x.matmul(torch.arange(fc_size_out*fc_size).double().view((fc_size_out, fc_size)).t())
    x_check[0,:] += torch.ones(fc_size_out)
    print(x_check == x_out)

if __name__ == '__main__':
    test()

