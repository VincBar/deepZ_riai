import torch
import sys
sys.path.append('D:/Dokumente/GitHub/RAI_proj/code_nn')
from networks import ReLUZConv,extend_Z

# Let x be the current zonotope. In the conv case x.shape = (K, c,h,w).
# Now we want to apply the DeepZ relaxation on the convolution layer. In our current approach it should add fc_size parameters
# to x_out.shape =(K+h*w,c,h,w). This should be improved, to less unecessary memory allocation



def test():
    K = 3
    c = 2
    h = 2
    w = 2
    x = (torch.ones((K, c, h , w))).double()
    x[0,1,0,0]=10
    x[0,1,0,1]=-10
    x[1,1,1,0]=2
    x[2,1,1,0]=-2
    reluz = ReLUZConv(c,h,w)
    x_out=reluz(x)
    # TODO: Currently all lambdas are initialized as one. Maybe the initalization can be learned number specific.
    print(x_out)
    print(x_out[0,1,...])#=[[10,0],[2.5,1.5]]
    print(x_out[1,1,...])#=[[1,0],[2,1]]
    print(x_out[2,1,...])#=[[1,0],[-2,1]]

    # TODO: Check for lambda not one

    print(x_out[3,1,...]==torch.Tensor([[0,0],[0,0]]))#=[[0,0],[0,0]]
    print(x_out[4,1,...]==torch.Tensor([[0,0],[0,0]]))#=[[0,0],[0,0]]
    print(x_out[5,1,...]==torch.Tensor([[0,0],[0,0]]))#=[[0,0],[0,0]]
    print(x_out[10, 1, ...] == torch.Tensor([[0, 0], [0, .50]]))

    # TODO: Reduce amount of lambdas necessary

if __name__ == '__main__':
    test()