from __future__ import absolute_import, division, print_function


import torch
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import BasicUNet
from monai.transforms import AsDiscrete, Compose, EnsureType
from monai.visualize.img2tensorboard import plot_2d_or_3d_image

from monai.networks.blocks import Convolution
import torch.nn.functional as F
from . import utils as zutils
from ..params import GameParams, ModelParams
from .. import utils
from torch import nn
from collections import OrderedDict

@zutils.register_model
class UNetMonai(torch.jit.ScriptModule):
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
    def __init__(
        self,
        game_params: GameParams,
        model_params: ModelParams,
    ):
        torch.jit.ScriptModule.__init__(self)

        if game_params.game_name is None:
            game_params.game_name = self.__class__.default_game_name
        self.game_name = game_params.game_name
        self.game_params = game_params
        info = zutils.get_game_info(game_params)
        c, h, w = self.c, self.h, self.w = info["feature_size"][:3]
        c_prime, h_prime, w_prime = self.c_prime, self.h_prime, self.w_prime = info["action_size"][:3]

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

        self.out_channels = self.c_prime
        self.in_channels = self.c

        self.v = nn.Linear(int(nnsize * c) * h * w, 1)
        self.pi_logit = nn.Conv2d(
            int(nnsize * c), c_prime, nnks, stride=stride, padding=padding, dilation=dilation
        )

        # Building model
        # self.model = BasicUNet(in_channels=self.in_channels, out_channels=self.out_channels)
        self.feature_extractor = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1",
                        Convolution(
                            dimensions=3,
                            in_channels=int(c / 9),
                            out_channels=int(nnsize * c / 9),
                            kernel_size=nnks,
                            strides=stride,
                            dilation=dilation,
                            padding=padding,
                            bias=True,
                        ),
                    ),
                    (
                        "conv2",
                        Convolution(
                            dimensions=3,
                            in_channels=int(nnsize * c / 9),
                            out_channels=int(nnsize * c / 9),
                            kernel_size=nnks,
                            strides=stride,
                            dilation=dilation,
                            padding=padding,
                            bias=True,
                        ),
                    ),
                    (
                        "conv3",
                        Convolution(
                            dimensions=3,
                            in_channels=int(nnsize * c / 9),
                            out_channels=int(nnsize * c / 9),
                            kernel_size=nnks,
                            strides=stride,
                            dilation=dilation,
                            padding=padding,
                            bias=True,
                        ),
                    ),
                ]
            )
        )

    @torch.jit.script_method
    def _forward(self, x: torch.Tensor, return_logit: bool):
        c = x.shape[1]
        x = x.view(int(x.shape[0]), int(c / 9), 9, int(x.shape[2]), int(x.shape[3]))
        h1 = self.feature_extractor(x)
        h1 = F.relu(h1)
        h = h1.view(h1.shape[0], h1.shape[1] * h1.shape[2], h1.shape[3], h1.shape[4])
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


