# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

from torch import nn
import torch
import torch.nn.functional as F


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(self, in_channels, out_channels, ksize=3, stride=1, bias=True):
        super().__init__()
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            bias=bias,
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv(x))


def tanh_range(l=0.5, r=2.0):
    def get_activation(left, right):
        def activation(x):
            return (torch.tanh(x) * 0.5 + 0.5) * (right - left) + left

        return activation

    return get_activation(l, r)


class AdaptiveModule(nn.Module):
    def __init__(self, in_ch=3, nf=32, tm_pts_num=8, gamma_range=None):
        super().__init__()

        self.tm_pts_num = tm_pts_num
        self.gamma_range = gamma_range or [1.0, 4.0]

        self.head1 = BaseConv(in_ch, nf, ksize=3, stride=2)
        self.body1 = BaseConv(nf, nf * 2, ksize=3, stride=2)
        self.body2 = BaseConv(nf * 2, nf * 4, ksize=3, stride=2)
        self.body3 = BaseConv(nf * 4, nf * 2, ksize=3)
        self.pooling = nn.AdaptiveAvgPool2d(1)

        self.image_adaptive_gamma = nn.Sequential(
            nn.Linear(nf * 2, nf * 4, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Linear(nf * 4, 3, bias=False),
        )

        self.head2 = BaseConv(in_ch, nf, ksize=3, stride=2)
        self.body_local1 = BaseConv(nf, nf * 2, ksize=3, stride=2)

        self.image_adaptive_local = nn.Sequential(
            nn.Conv2d(nf * 2, nf * 2, 3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf * 2, tm_pts_num * 3 * 2, 3, stride=1, padding=1, bias=True),
        )

        self.out_layer = nn.Sequential(
            nn.Conv2d(3, 16, 1, stride=1, padding=0, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(16, 3, 1, stride=1, padding=0, bias=False),
        )

    def apply_gamma(self, img, ia_params):
        ia_params = tanh_range(self.gamma_range[0], self.gamma_range[1])(
            ia_params
        )[..., None, None]
        out_image = img ** (1.0 / ia_params)
        return out_image

    def apply_local(self, img, ltm):
        n1, _, h1, w1 = img.shape
        n2, _, h2, w2 = ltm.shape
        assert n1 == n2 and h1 == h2 and w1 == w2, f"LTM has invalid shape < {ltm.shape} >!"

        ltm = tanh_range(0.1, 99.9)(ltm).reshape(n1, 6, self.tm_pts_num, h1, w1)
        ltm1, ltm2 = torch.split(ltm, 3, dim=1)
        ltm1 = ltm1 / torch.sum(ltm1, dim=2, keepdim=True)
        ltm2 = ltm2 / torch.sum(ltm1 * ltm2, dim=2, keepdim=True)
        ltm1_ = [ltm1[:, :, :i].sum(2) for i in range(self.tm_pts_num + 1)]

        total_image = 0
        for i, point in enumerate(torch.split(ltm2, 1, dim=2)):
            total_image += (
                torch.minimum(
                    torch.clamp(img - ltm1_[i], 0, None),
                    ltm1_[i + 1] - ltm1_[i],
                )
                * point.squeeze(2)
            )

        return total_image

    def forward(self, img):
        img_down = F.interpolate(img, (256, 256), mode="bilinear", align_corners=True)
        h, w = img.shape[-2:]

        fea = self.head1(img_down)
        fea_s2 = self.body1(fea)
        fea_s4 = self.body2(fea_s2)
        fea_s8 = self.body3(fea_s4)
        fea_gamma = self.pooling(fea_s8)
        fea_gamma = fea_gamma.view(fea_gamma.shape[0], fea_gamma.shape[1])
        para_gamma = self.image_adaptive_gamma(fea_gamma)
        out_gamma = self.apply_gamma(img, para_gamma)

        fea_local = self.head2(img_down)
        fea_local = self.body_local1(fea_local)
        param_local = self.image_adaptive_local(fea_local)
        param_local = F.interpolate(param_local, (h, w), mode="bilinear", align_corners=True)
        out_local = self.apply_local(img, param_local)

        out = self.out_layer((out_local + out_gamma) / 2)

        return out
