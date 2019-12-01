import torch
import sys
sys.path.append('D:/Dokumente/GitHub/RAI_proj/code')
from code_nn.networks import ReLUZLinear, extend_Z

# Let x be the current zonotope. In the linear case x.shape = (K, fc_size).
# Now we want to apply the DeepZ relaxation on a linear Layer. In our current approach it should add fc_size parameters
# to x_out.shape =(K+fc_size,fc_size). This should be improved, to less unecessary memory allocation



def test():
    K = 3
    fc_size = 5
    x = torch.ones((K, fc_size)).double()
    x[0,0] = -10
    x[0,1] = 10
    reluz = ReLUZLinear(fc_size)
    # TODO: ERROR in propagation of RELUZ a_00 should be 0 but is 6, corrected
    # TODO: ERROR in propagation of RELUZ a_01 should be 10 but is 6, corrected maybe online description is wrong
    x_out=reluz(x)

    print(x_out.shape[0]-K==fc_size)

    print(x_out)

    x_true=torch.Tensor([[ 0.0000, 10.0000,  1.5000,  1.5000,  1.5000],
            [ 0.0000,  1.0000,  1.0000,  1.0000,  1.0000],
            [ 0.0000,  1.0000,  1.0000,  1.0000,  1.0000],
            [ 0.0000,  0.0000,  0.5000,  0.0000,  0.0000],
            [ 0.0000,  0.0000,  0.0000,  0.5000,  0.0000],
            [ 0.0000,  0.0000,  0.0000,  0.0000,  0.5000]])

    assert torch.all((x_out==x_true))

if __name__ == '__main__':
    test()