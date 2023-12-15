from __future__ import absolute_import, division, print_function

from collections import OrderedDict

import torch
import torch.nn.functional as F
from .layers import ConvSlimCapsule3D
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks import one_hot
from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Conv
from monai.transforms import AsDiscrete, Compose, EnsureType
from monai.visualize.img2tensorboard import plot_2d_or_3d_image
from torch import nn

from . import utils as zutils
from ..params import GameParams, ModelParams
from .. import utils



@zutils.register_model
class ModifiedUCaps3D(torch.jit.ScriptModule):
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

        print("---------")
        print("进入mod_ucaps_init")
        # 原项目中的一些参数
        self.share_weight = False
        self.connection = "skip"



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


        self.out_channels =  self.c_prime
        self.in_channels = self.c
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

        # self.save_hyperparameters()
        # self.in_channels = self.hparams.in_channels
        # self.out_channels = self.hparams.out_channels
        # self.share_weight = self.hparams.share_weight
        # self.connection = self.hparams.connection
        #
        # self.lr_rate = self.hparams.lr_rate
        # self.weight_decay = self.hparams.weight_decay
        #
        # self.cls_loss = self.hparams.cls_loss
        # self.margin_loss_weight = self.hparams.margin_loss_weight
        # self.rec_loss_weight = self.hparams.rec_loss_weight
        # self.class_weight = self.hparams.class_weight
        #
        # # Defining losses
        # self.classification_loss1 = MarginLoss(class_weight=self.class_weight, margin=0.2)
        #
        # if self.cls_loss == "DiceCE":
        #     self.classification_loss2 = DiceCELoss(softmax=True, to_onehot_y=True, ce_weight=self.class_weight)
        # elif self.cls_loss == "CE":
        #     self.classification_loss2 = DiceCELoss(
        #         softmax=True, to_onehot_y=True, ce_weight=self.class_weight, lambda_dice=0.0
        #     )
        # elif self.cls_loss == "Dice":
        #     self.classification_loss2 = DiceCELoss(softmax=True, to_onehot_y=True, lambda_ce=0.0)
        # self.reconstruction_loss = nn.MSELoss(reduction="none")
        #
        # self.val_frequency = self.hparams.val_frequency
        # self.val_patch_size = self.hparams.val_patch_size
        # self.sw_batch_size = self.hparams.sw_batch_size
        # self.overlap = self.hparams.overlap


        # Building model

        mono = [
            nn.Conv3d(
                int(c / 9),
                int(nnsize * c / 9),
                nnks,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=not bn_affine,
            )
        ]
        if bn or bn_affine:
            mono.append(
                nn.BatchNorm2d(int(nnsize * c), track_running_stats=True, affine=bn_affine)
            )
        self.mono = nn.Sequential(*mono)

        self.feature_extractor = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1",
                        Convolution(
                            dimensions=3,
                            # in_channels=self.in_channels,
                            in_channels= int( c / 9),
                            out_channels=16,
                            kernel_size=(2,3,3),
                            strides=1,
                            dilation=1,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (
                        "conv2",
                        Convolution(
                            dimensions=3,
                            in_channels=16,
                            out_channels=32,
                            kernel_size=3,
                            strides=1,
                            dilation=1,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (
                        "conv3",
                        Convolution(
                            dimensions=3,
                            in_channels=32,
                            out_channels=64,
                            kernel_size=3,
                            strides=1,
                            padding=1,
                            dilation=1,
                            bias=False,
                            act="tanh",
                        ),
                    ),
                ]
            )
        )

        self.primary_caps = ConvSlimCapsule3D(
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
        self._build_encoder()
        self._build_decoder()
        # self._build_reconstruct_branch()

        # For validation
        self.post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=True, n_classes=self.out_channels)])
        self.post_label = Compose([EnsureType(), AsDiscrete(to_onehot=True, n_classes=self.out_channels)])

        self.dice_metric = DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=False)

        self.example_input_array = torch.rand(1, self.in_channels, 32, 32, 32)

        self.v = nn.Linear(int(10 * c_prime) * h * w, 1)
        self.pi_logit = nn.Conv2d(
            int(10 * c_prime), c_prime, nnks, stride=stride, padding=padding, dilation=dilation
        )

    @torch.jit.script_method
    def _forward(self, x: torch.Tensor, return_logit: bool):

        c = x.shape[1]
        x = x.view(int(x.shape[0]), int(c / 9), 9, int(x.shape[2]), int(x.shape[3]))#x.shape[1, 4, 9, 9, 9]


        # Contracting
        x = self.feature_extractor(x)#x.shape[1, 64, 9, 9, 9]


        # fe_0 = self.feature_extractor.conv1(x)
        # fe_1 = self.feature_extractor.conv2(fe_0)
        # x = self.feature_extractor.conv3(fe_1)

        conv_1_1 = x
        conv_2_1 = self.encoder_convs[0](conv_1_1)# conv_2_1.shape[1, 128, 9, 9, 9]


        conv_2_1 = self.relu(conv_2_1)

        # conv_2_1 = self.encoder_convs[1](conv_2_1)
        # conv_2_1 = self.relu(conv_2_1)

        conv_3_1 = self.encoder_convs[1](conv_2_1)# conv_2_1.shape[1, 128, 9, 9, 9]
        conv_3_1 = self.relu(conv_3_1)


        # conv_3_1 = self.encoder_convs[3](conv_3_1)
        # conv_3_1 = self.relu(conv_3_1)

        conv_3_1_reshaped = conv_3_1.view(
            -1, self.encoder_output_dim[3], self.encoder_output_atoms[3], 
            conv_3_1.shape[-3], conv_3_1.shape[-2], conv_3_1.shape[-1])#[1, 8, 16, 9, 9, 9]


        x = self.encoder_conv_caps[4](conv_3_1_reshaped)#[1, 8, 32, 5, 5, 5]


        # conv_cap_4_1 = x
        conv_cap_4_1 = self.encoder_conv_caps[5](x)#[1, 3, 64, 5, 5, 5]


        shape = conv_cap_4_1.size()
        conv_cap_4_1 = conv_cap_4_1.view(shape[0], -1, shape[-3], shape[-2], shape[-1])#[1, 192, 5, 5, 5]


        # Expanding
        if self.connection == "skip":
            x = self.decoder_conv[0](conv_cap_4_1)#[1, 256, 10, 10, 10]

            x = torch.cat((x, conv_3_1), dim=1)

            x = self.decoder_conv[1](x)

            # x = self.decoder_conv[2](x)#这里好像进行了上采样，TJ 从[1, 128, 10, 8, 8] 变成了 [1, 128, 20, 16, 16]
            # print("--")
            # print(x.shape)
            # print("--")
            x = torch.cat((x, conv_2_1), dim=1)

            x = self.decoder_conv[3](x)

            # x = self.decoder_conv[4](x)
            # print("-----")
            # print(x.shape)
            # print("-----")
            x = torch.cat((x, conv_1_1), dim=1)


            # extend decover and skip connection
            # x = self.add_deconvs[0](x)
            # x = torch.cat((x, fe_1), dim=1)
            # x = self.add_deconvs[1](x)
            # x = torch.cat((x, fe_0), dim=1)

        h1 = self.decoder_conv[5](x)

        h = h1.view(h1.shape[0], h1.shape[1] * h1.shape[2], h1.shape[3], h1.shape[4])

        v = torch.tanh(self.v(h.flatten(1)))

        pi_logit = self.pi_logit(h).flatten(1)

        if return_logit:
            return v, pi_logit
        s = pi_logit.shape
        pi = F.softmax(pi_logit.flatten(1), 1).reshape(s)
        return v, pi


    def _build_encoder(self):
        self.encoder_convs = nn.ModuleList()
        self.encoder_convs.append(
            nn.Conv3d(64, 128, 3, stride=1, padding=1)
        )
        # self.encoder_convs.append(
        #     nn.Conv3d(128, 128, 5, stride=1, padding=2)
        # )
        self.encoder_convs.append(
            nn.Conv3d(128, 128, 3, stride=1, padding=1)
        )
        # self.encoder_convs.append(
        #     nn.Conv3d(128, 128, 5, stride=1, padding=2)
        # )
        for i in range(len(self.encoder_convs)):
            torch.nn.init.normal_(self.encoder_convs[i].weight, std=0.1)
        self.relu = nn.ReLU(inplace=True)

        self.encoder_conv_caps = nn.ModuleList()
        self.encoder_kernel_size = 3
        self.encoder_output_dim = [16, 16, 8, 8, 8, self.out_channels]
        self.encoder_output_atoms = [8, 8, 16, 16, 32, 64]

        for i in range(len(self.encoder_output_dim)):
            if i == 0:
                input_dim = self.primary_caps.output_dim  #16
                input_atoms = self.primary_caps.output_atoms  # 4
            else:
                input_dim = self.encoder_output_dim[i - 1]
                input_atoms = self.encoder_output_atoms[i - 1]

            stride = 2 if i % 2 == 0 else 1

            self.encoder_conv_caps.append(
                ConvSlimCapsule3D(
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


    def _build_decoder(self):
        # self.add_deconvs = nn.ModuleList()
        # self.add_deconvs.append(
        #     nn.ConvTranspose3d(128, 32, 1, 1)
        # )
        # self.add_deconvs.append(
        #     nn.ConvTranspose3d(64, 16, 1, 1)
        # )
        self.decoder_conv = nn.ModuleList()
        if self.connection == "skip":
            self.decoder_in_channels = [self.out_channels * self.encoder_output_atoms[-1], 384, 128, 256, 64, 128]
            self.decoder_out_channels = [256, 128, 128, 64, 64, self.out_channels]

        for i in range(6):
            if i == 5:
                self.decoder_conv.append(
                    Conv["conv", 3](self.decoder_in_channels[i], self.decoder_out_channels[i], kernel_size=1)
                )
                # self.decoder_conv.append(
                #     Conv["conv", 3](32, self.decoder_out_channels[i], kernel_size=1)
                # )
            elif i % 2 == 0:
                self.decoder_conv.append(
                    UpSample(
                        dimensions=3,
                        in_channels=self.decoder_in_channels[i],
                        out_channels=self.decoder_out_channels[i],
                        scale_factor=2,
                    )
                )
            else:
                self.decoder_conv.append(
                    Convolution(
                        dimensions=3,
                        kernel_size=3,
                        in_channels=self.decoder_in_channels[i],
                        out_channels=self.decoder_out_channels[i],
                        strides=1,
                        padding=1,
                        bias=False,
                    )
                )


    def lalalaforward(self, x: torch.Tensor, return_logit: bool):
        c = x.shape[1]
        x = x.view(int(x.shape[0]), int(c / 9), 9, int(x.shape[2]), int(x.shape[3]))
        h = self.mono(x)  # linear transformation only
        saved_h = [
            h
        ] * self.nb_unets_div_by_2  # saves output of last linear transformation
        layer_no = 0
        for unet in self.unets:
            sublayer_no = 0
            for net in unet:
                h = F.relu(h)  # activation on previous block
                h = net(h)
                if sublayer_no == self.nb_layers_per_net - 1:
                    if layer_no < self.nb_unets_div_by_2:
                        saved_h[layer_no] = h
                    elif layer_no > self.nb_unets_div_by_2:
                        h = h + saved_h[layer_no - self.nb_unets_div_by_2 - 1]
                sublayer_no = sublayer_no + 1
            layer_no = layer_no + 1
        h1 = F.relu(h)  # final activation
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



