import torch
import sys
sys.path.append('D:/Dokumente/GitHub/RAI_proj/code')
from networks import ReLUZLinear

# Let x be the current zonotope. In the linear case x.shape = (K, fc_size).
# Now we want to apply the DeepZ relaxation on a linear Layer. In our current approach it should add fc_size parameters
# to x_out.shape =(K+fc_size,fc_size). This should be improved, to less unecessary memory allocation


def test():
    K = 3
    fc_size = 5
    x = torch.ones((K, fc_size)).double()
    reluz = ReLUZLinear(fc_size)
    x_out=reluz(x)
    print(x_out.shape[0]-K==fc_size)
    # TODO: Currently all lambdas are initialized as one. Maybe the initalization can be learned number specific.
    print(x_out)

if __name__ == '__main__':
    test()