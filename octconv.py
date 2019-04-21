'''
naive implementation of Octave conv
assert input_size=2**n
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class OctConv(nn.Module):
    def __init__(self, nChannels, nOutChannels, stride=1, alphas=(0.5, 0.5)):
        super(OctConv, self).__init__()
        self.alpha_in, self.alpha_out = alphas
        assert 0 <= self.alpha_in <= 1, "Alphas must be in [0, 1]"
        assert 0 <= self.alpha_out <= 1, "Alphas must be in [0, 1]"

        # num channels
        nCh_h = int(nChannels * (1 - self.alpha_in))
        nOutCh_h = int(nOutChannels * (1 - self.alpha_out))
        nCh_l = nChannels - nCh_h
        nOutCh_l = nOutChannels - nOutCh_h

        # filters
        self.H2H = nn.Conv2d(nCh_h, nOutCh_h, kernel_size=3, stride=stride, padding=1, bias=False)
        if nOutCh_l!=0:
            self.H2L = nn.Conv2d(nCh_h, nOutCh_l, kernel_size=3, stride=stride, padding=1, bias=False)
        if nCh_l!=0 and nOutCh_l!=0: 
            self.L2L = nn.Conv2d(nCh_l, nOutCh_l, kernel_size=3, stride=stride, padding=1, bias=False)
        if nCh_l!=0:
            self.L2H = nn.Conv2d(nCh_l, nOutCh_h, kernel_size=3, stride=stride, padding=1, bias=False)


    def forward(self, x):
        if self.alpha_in == 0:
            h_in = x
            h2h = self.H2H(h_in)
            h2l = self.H2L(F.avg_pool2d(h_in,2))
            return h2h, h2l

        elif self.alpha_out == 0:
            h_in, l_in = x
            h2h = self.H2H(h_in)
            l2h = F.interpolate(self.L2H(l_in), scale_factor=2, mode='nearest')
            return h2h + l2h
             
        else:
            h_in, l_in = x
            h2h = self.H2H(h_in)
            h2l = self.H2L(F.avg_pool2d(h_in,2))
            l2l = self.L2L(l_in)
            l2h = F.interpolate(self.L2H(l_in), scale_factor=2, mode='nearest')
            return h2h + l2h, h2l + l2l


def main():
    oct_in  = OctConv(3,8,1,(0  ,0.5))
    oct_    = OctConv(8,8,2,(0.5,0.5))
    oct_out = OctConv(8,1,1,(0.5,0  ))
    
    net = nn.Sequential(oct_in,oct_,oct_out)
    dummy_x = torch.randn(2,3,32,32)
    print(net)

    if torch.cuda.is_available():
        net = net.cuda()
        dummy_x = dummy_x.cuda()
    print(net(dummy_x).size())


if __name__ == '__main__':
    main()