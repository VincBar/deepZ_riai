import torch
import sys
sys.path.append('D:/Dokumente/GitHub/RAI_proj/code_nn')
from code.networks import extend_Z

# Let x the current zonotope. In the conv case x.shape = (K, C, H, W).
# Now, we want the tensor cast b with b.shape = (C, H, W) into x tensor c with c.shape = (C * H * W, C, H, W)
# such that c_kchw = b_chw if k = lin_order(c, h, w) else 0. Then we can extend x to x.shape = (K + C * H * W, C, H, W).
# The following shows that extend_Z achieves this. Specifically, we require c_kchw = b_chw for all k >= K if c_kchw is
# non-zero.

def test():
    x = torch.ones((1, 3, 3, 3))
    b = torch.arange(27).view((3, 3, 3))
    c = b.flatten(start_dim=0)
    x_ = extend_Z(x,c)
    x_2 = extend_Z(x,b)
    x_3 = extend_Z(x, 3.)

    print(x_[:, 1, 2, 1])
    print(b[1, 2, 1])

    # print(x_==x_2)
    print(x_3[:, 1, 2, 1])

if __name__ == '__main__':
    test()

