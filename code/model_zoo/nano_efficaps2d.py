# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
import torch.nn.functional as F

from . import utils as zutils
from ..params import GameParams, ModelParams
from .. import utils


def squash(input:torch.Tensor, eps:float=10e-21):
    n = torch.norm(input, dim=-1, keepdim=True)
    return (1 - 1 / (torch.exp(n) + eps)) * (input / (n + eps))


def length(input):
    return torch.sqrt(torch.sum(input**2, dim=-1) + 1e-8)


def mask(input):
    if type(input) is list:
        input, mask = input
    else:
        x = torch.sqrt(torch.sum(input**2, dim=-1))
        mask = F.one_hot(torch.argmax(x, dim=1), num_classes=x.shape[1]).float()

    masked = input * mask.unsqueeze(-1)
    return masked.view(input.shape[0], -1)




@zutils.register_model
class NanoEffiCaps2D(torch.jit.ScriptModule):
    __constants__ = ["c_prime", "h_prime", "w_prime", "net"]

    DEFAULT_NNSIZE = 2
    DEFAULT_NNKS = 3
    DEFAULT_STRIDE = 1
    DEFAULT_DILATION = 1
    DEFAULT_BN = False
    # DEFAULT_BN_AFFINE = False

    default_game_name = "Hex13"

    def __init__(self, game_params: GameParams, model_params: ModelParams):
        torch.jit.ScriptModule.__init__(self)
        if game_params.game_name is None:
            game_params.game_name = self.__class__.default_game_name
        self.game_name = game_params.game_name
        self.game_params = game_params
        info = zutils.get_game_info(game_params)
        c, h, w = self.c, self.h, self.w = info["feature_size"][:3]
        c_prime, h_prime, w_prime = self.c_prime, self.h_prime, self.w_prime = info[
            "action_size"
        ][:3]
        if h_prime != h or w_prime != w:
            raise RuntimeError(
                f'The game "{self.game_name}" is not eligible to a conv-computed logit '
                f'model such as "{self.__class__.__name__}" - try with '
                f'"{self.__class__.__name__.replace("ConvLogit", "FCLogit")}" instead'
            )

        # nn size
        if model_params.nnsize is None:
            model_params.nnsize = self.DEFAULT_NNSIZE
        nnsize = model_params.nnsize
        # kernel size
        if model_params.nnks is None:
            model_params.nnks = self.DEFAULT_NNKS
        nnks = model_params.nnks
        # stride
        stride = self.DEFAULT_STRIDE
        # dilation
        dilation = self.DEFAULT_DILATION
        # padding
        padding = zutils.get_consistent_padding_from_nnks(nnks=nnks, dilation=dilation)
        # batch norm
        if model_params.bn is None:
            model_params.bn = self.DEFAULT_BN
        bn = model_params.bn
        # # batch norm affine
        # if model_params.bn_affine is None:
        #     model_params.bn_affine = self.DEFAULT_BN_AFFINE
        # bn_affine = model_params.bn_affine
        bn_affine = bn
        self.model_params = model_params

        net = [
            nn.Conv2d(
                c,
                int(nnsize * c),
                nnks,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=not bn_affine,
            )
        ]
        if bn or bn_affine:
            net.append(
                nn.BatchNorm2d(int(nnsize * c), track_running_stats=True, affine=bn_affine)
            )
        self.net = nn.Sequential(*net)
        self.v = nn.Linear(int(nnsize * c) * h * w, 1)
        self.pi_logit = nn.Conv2d(
            int(nnsize * c), c_prime, nnks, stride=stride, padding=padding, dilation=dilation
        )

        self.primary_caps = PrimaryCapsLayer(
            in_channels=int(nnsize * c), kernel_size=3, num_capsules=18, dim_capsules=4
        )
        self.digit_caps = RoutingLayer(num_capsules=24, dim_capsules=self.c_prime)

    @torch.jit.script_method
    def _forward(self, x: torch.Tensor, return_logit: bool):
        print("-")
        print(x.shape)
        print("-")
        h = F.relu(self.net(x))
        print("--")
        print(h.shape)
        print("--")
        h = self.primary_caps(h)
        print("=")
        print(h.shape)
        print("=")
        h = self.digit_caps(h)
        print("==")
        print(h.shape)
        print("==")

        v = torch.tanh(self.v(h.flatten(1)))
        pi_logit = self.pi_logit(h).flatten(1)
        if return_logit:
            return v, pi_logit
        s = pi_logit.shape
        pi = F.softmax(pi_logit.flatten(1), 1).reshape(s)
        return v, pi

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        v, pi_logit = self._forward(x, True)
        pi_logit = pi_logit.view(-1, self.c_prime, self.h_prime, self.w_prime)
        reply = {"v": v, "pi_logit": pi_logit}
        return reply


class PrimaryCapsLayer(torch.jit.ScriptModule):
    """Create a primary capsule layer where the properties of each capsule are extracted
    using a 2D depthwise convolution.

    Args:
        in_channels (int): depthwise convolution's number of features
        kernel_size (int): depthwise convolution's kernel dimension
        num_capsules (int): number of primary capsules
        dim_capsules (int): primary capsule dimension
        stride (int, optional): depthwise convolution's strides. Defaults to 1.
    """

    def __init__(self, in_channels, kernel_size, num_capsules, dim_capsules, stride=1):
        torch.jit.ScriptModule.__init__(self)
        self.depthwise_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=in_channels,
            padding=1,
        )
        self.num_capsules = num_capsules
        self.dim_capsules = dim_capsules

    @torch.jit.script_method
    def forward(self, input):
        print("---")
        print(input.shape)
        print("---")
        output = self.depthwise_conv(input)
        print("----")
        print(output.shape)
        print("----")
        output = output.view(output.size(0), output.size(1),self.num_capsules, self.dim_capsules)
        print("-----")
        print(output.shape)
        print("-----")
        return squash(output)


class RoutingLayer(torch.jit.ScriptModule):
    """Self-attention routing layer using a fully-connected network, to create a parent
    layer of capsules.
    Args:

        num_capsules (int): number of primary capsules
        dim_capsules (int): primary capsule dimension
    """

    def __init__(self, num_capsules, dim_capsules):
        torch.jit.ScriptModule.__init__(self)
        self.W = nn.Parameter(torch.Tensor(num_capsules, 18, 4, dim_capsules))
        self.b = nn.Parameter(torch.zeros(num_capsules, 18, 1))
        self.num_capsules = num_capsules
        self.dim_capsules = dim_capsules
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.W)
        nn.init.zeros_(self.b)

    @torch.jit.script_method
    def forward(self, input):
        print(input.shape)
        u = torch.einsum(
            "...ji,kjiz->...kjz", input, self.W
        )  # u shape = (None, num_capsules, height*width*16, dim_capsules)
        print(u.shape)
        c = torch.einsum("...ij,...kj->...i", u, u)[
            ..., None
        ]  # b shape = (None, num_capsules, height*width*16, 1) -> (None, j, i, 1)
        print(c.shape)
        c = c / torch.sqrt(
            # torch.Tensor([self.dim_capsules]).type(torch.cuda.FloatTensor)
            torch.tensor([self.dim_capsules])
        )
        print(c.shape)
        c = torch.softmax(c, dim=1)
        c = c + self.b
        s = torch.sum(
            torch.mul(u, c), dim=-2
        )  # s shape = (None, num_capsules, dim_capsules)
        return squash(s)