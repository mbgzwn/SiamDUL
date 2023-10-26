from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['SiamFC']

from siamfc.backbones import NONLocalBlock2D, SELayer1, SELayer, ECALayer, SimAM


class SiamFC(nn.Module):

    def __init__(self, out_scale=0.001):
        super(SiamFC, self).__init__()
        self.out_scale = out_scale
        # self.att1 = nn.Sequential(NONLocalBlock2D(256))
        self.att = nn.Sequential(SELayer1(256))
        # self.att2 = nn.Sequential(SimAM(256))

    def forward(self,z, x):
        a = self.att(z)
        z = z + a
        # out1 = self._fast_xcorr(z1, x1) * self.out_scale
        # out2 = self._fast_xcorr(z2, x2) * self.out_scale
        # out = 0.3 * out1 + 0.7 * out2
        # return out
        return self._fast_xcorr(z, x) * self.out_scale

    def _fast_xcorr(self, z, x):
        # fast cross correlation
        # nz和nx为batchsize，即这次喂进去的图片的张数
        nz = z.size(0)  # 1
        nx, c, h, w = x.size()  # 8 128 22 22
        x = x.view(-1, nz * c, h, w)
        out = F.conv2d(x, z, groups=nz)
        out = out.view(nx, -1, out.size(-2), out.size(-1))
        return out

# if __name__ == '__main__':
#     x = torch.rand((8, 512, 22, 22))
#     z = torch.rand((1, 512, 6, 6))
#
#
#     def fast_xcorr(self, z, x):
#         # fast cross correlation
#         # nz和nx为batchsize，即这次喂进去的图片的张数
#         nz = z.size(0)
#         nx, c, h, w = x.size()
#         x = x.view(-1, nz * c, h, w)
#         print(x.shape)
#         out = F.conv2d(x, z, groups=nz)
#         print(out.shape)
#         out = out.view(nx, -1, out.size(-2), out.size(-1))
#         print(out.shape)
#         return out
#
#
#     fast_xcorr(x, z, x)
