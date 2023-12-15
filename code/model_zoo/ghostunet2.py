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

from .modules import ResidualConv, Upsample,GhostBottleneckV2,GhostBottleneck,SqueezeExcite

@zutils.register_model
class Ghostunet2SE(torch.jit.ScriptModule):
    __constants__ = [
        "c_prime",
        "h_prime",
        "w_prime",
        "nb_layers_per_net",
        "mono",
        "nb_unets_div_by_2",
        "unets",
    ]

    DEFAULT_NB_NETS = 5
    DEFAULT_NB_LAYERS_PER_NET = 3
    DEFAULT_NNSIZE = 2
    DEFAULT_NNKS = 3
    DEFAULT_STRIDE = 1
    DEFAULT_DILATION = 1
    DEFAULT_POOLING = False
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

        # nb unets
        if model_params.nb_nets is None:
            model_params.nb_nets = self.DEFAULT_NB_NETS
        nb_nets = model_params.nb_nets
        if nb_nets % 2 == 0:
            raise RuntimeError(
                f'The model "{self.__class__.__name__}" accepts only odd numbers '
                f'for "nb_nets" while it was set to {nb_nets}'
            )
        self.nb_unets_div_by_2 = nb_unets_div_by_2 = nb_nets // 2
        # nb layers per unet
        if model_params.nb_layers_per_net is None:
            model_params.nb_layers_per_net = self.DEFAULT_NB_LAYERS_PER_NET
        nb_layers_per_net = model_params.nb_layers_per_net
        self.nb_layers_per_net = nb_layers_per_net
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
        # pooling
        if model_params.pooling is None:
            model_params.pooling = self.DEFAULT_POOLING
        pooling = model_params.pooling
        # batch norm
        if model_params.bn is None:
            model_params.bn = self.DEFAULT_BN
        bn = model_params.bn
        # # batch norm affine
        # if model_params.bn_affine is None:
        #     model_params.bn_affine
        # bn_affine = model_params.bn_affine = self.DEFAULT_BN_AFFINE
        bn_affine = bn
        self.model_params = model_params

        filters = [72, 72, 72, 72]
        self.input_layer = nn.Sequential(
            nn.Conv2d(self.c, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(self.c, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = GhostBottleneck(72,72,72,se_ratio=0.)
        # self.residual_conv_1 = GhostModuleVV(72, 72, relu=True, mode='attn')
        self.residual_conv_2 = GhostBottleneck(72, 72, 72, se_ratio=0.)
        # self.residual_conv_2 = GhostModuleVV(72, 72, relu=True, mode='attn')
        self.residual_conv_3 = GhostBottleneck(72, 72, 72, se_ratio=0.)
        self.bridge = GhostBottleneck(72, 72, 72, se_ratio=0.)

        # self.upsample_1 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv1 = GhostBottleneck(filters[3] + filters[2], filters[2], filters[2], se_ratio=0.)


        # self.upsample_2 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv2 = GhostBottleneck(filters[2] + filters[1], filters[1], filters[1], se_ratio=0.)

        # self.upsample_3 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv3 = GhostBottleneck(filters[1] + filters[0], filters[0], filters[0],se_ratio=0.)

        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], c_prime, 1, 1),
        )

        self.v = nn.Linear(72 * h * w, 1)
        self.pi_logit = nn.Conv2d(
            int(72), c_prime, nnks, stride=stride, padding=padding, dilation=dilation
        )

        self.seblock1 = SqueezeExcite(72)
        self.seblock2 = SqueezeExcite(72)
        self.seblock3 = SqueezeExcite(72)

    @torch.jit.script_method
    def _forward(self, x: torch.Tensor, return_logit: bool):

        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        x4 = self.residual_conv_3(x3)
        # Bridge
        x5 = self.bridge(x4)

        # Decode
        x2 = self.seblock1(x2)

        x3 = self.seblock2(x3)
        x4 = self.seblock3(x4)
        # x4 = self.upsample_1(x4)
        x6 = torch.cat([x5, x4], dim=1)


        x7 = self.up_residual_conv1(x6)

        x8 = torch.cat([x7, x3], dim=1)

        x9 = self.up_residual_conv2(x8)

        x10 = torch.cat([x9, x2], dim=1)

        x11 = self.up_residual_conv3(x10)

        h = F.relu(x11)  # final activation
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
