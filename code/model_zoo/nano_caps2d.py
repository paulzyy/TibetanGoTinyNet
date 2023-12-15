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
from .layers import ConvSlimCapsule2D

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
class NanoCaps2D(torch.jit.ScriptModule):
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
                int(128),
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
        self.v = nn.Linear(int(192) * h * w, 1)
        self.pi_logit = nn.Conv2d(
            int(192), c_prime, nnks, stride=stride, padding=padding, dilation=dilation
        )

        self.share_weight = True

        self.primary_caps = ConvSlimCapsule2D(
            kernel_size=3,
            input_dim=1,
            output_dim=16,
            input_atoms=64,
            output_atoms=4,
            stride=1,
            padding=1,
            num_routing=1,
            share_weight=self.share_weight,
        )

        self.encoder_output_dim = [16, 16, 8, 8, 8, self.c_prime]
        self.encoder_output_atoms = [8, 8, 16, 16, 32, 64]
        self.encoder_conv_caps = nn.ModuleList()
        self.encoder_kernel_size = 3

        for i in range(len(self.encoder_output_dim)):
            if i == 0:
                input_dim = self.primary_caps.output_dim  # 16
                input_atoms = self.primary_caps.output_atoms  # 4
            else:
                input_dim = self.encoder_output_dim[i - 1]
                input_atoms = self.encoder_output_atoms[i - 1]

            # stride = 2 if i % 2 == 0 else 1
            stride = 1


            self.encoder_conv_caps.append(
                ConvSlimCapsule2D(
                    kernel_size=self.encoder_kernel_size,
                    input_dim=input_dim,
                    output_dim=self.encoder_output_dim[i],
                    input_atoms=input_atoms,
                    output_atoms=self.encoder_output_atoms[i],
                    stride=stride,
                    padding=1,
                    dilation=1,
                    num_routing=3,
                    share_weight=self.share_weight,
                )
            )


    @torch.jit.script_method
    def _forward(self, x: torch.Tensor, return_logit: bool):

        h = F.relu(self.net(x))


        conv_3_1_reshaped = h.view(
            -1, self.encoder_output_dim[3], self.encoder_output_atoms[3], h.shape[-2], h.shape[-1])  # [1, 8, 16, 8, 8]


        x = self.encoder_conv_caps[4](conv_3_1_reshaped)  # [1, 8, 32, 8, 8]
        # conv_cap_4_1 = x
        conv_cap_4_1 = self.encoder_conv_caps[5](x)  # [1, 3, 64, 8, 8]

        shape = conv_cap_4_1.size()
        h = conv_cap_4_1.view(shape[0], -1, shape[-2], shape[-1])  # [1, 192, 8, 8]


        v = torch.tanh(self.v(h.flatten(1)))
        print(v.shape)
        pi_logit = self.pi_logit(h).flatten(1)
        print(pi_logit.shape)
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


