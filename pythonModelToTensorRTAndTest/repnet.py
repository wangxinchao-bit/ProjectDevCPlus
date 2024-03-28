import os
import torch
import torch.nn as nn
import torch.nn.functional as F
def bilinear_interpolate(input, scale_factor):
    # 获取输入张量的尺寸
    _, _, H, W = input.shape

    # 计算输出张量的尺寸
    new_H = int(H * scale_factor.item())
    new_W = int(W * scale_factor.item())

    # 使用 torch.nn.functional.interpolate 进行双线性插值
    interpolated_tensor = torch.nn.functional.interpolate(input, size=(new_H, new_W), mode='bilinear')

    return interpolated_tensor


class repnet(nn.Module):
    def __init__(self,upscale=2):
        super(repnet, self).__init__()

        self.upscale = upscale
        num_feat = 32
        down_scale = 2

        if self.upscale == 2:
           num_repconv = 1
        else :
           num_repconv = 2

        # downsample
        self.downsampler = nn.PixelUnshuffle(down_scale)

        self.body = nn.ModuleList()
        # the first conv

        self.body.append(nn.Conv2d(3*upscale*down_scale, num_feat, 3, 1, 1))

        activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.body.append(activation)

        # the body structure
        for _ in range(num_repconv):

            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(activation)

        # the last conv

        self.body.append(nn.Conv2d(num_feat, 3 * upscale*down_scale * upscale*down_scale, 3, 1, 1))

        # upsample
        self.upsampler = nn.PixelShuffle(upscale*down_scale)

    def forward(self, x):
        out = self.downsampler(x)

        # for i in range(0, len(self.body)):
        #     out = self.body[i](out)
        for i, module in enumerate(self.body):
            out = module(out)

        out = self.upsampler(out)
        print(out.shape)
        base = F.interpolate(x, scale_factor=2.0, mode='bilinear')
        # base = bilinear_interpolate(x, scale_factor= torch.tensor([2.0]) )
        # print(base.shape)
        out += base
        return out