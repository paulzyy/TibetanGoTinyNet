# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .generic_model import GenericModel
from .amazons_model import AmazonsModel
from .nano_fc_logit_model import NanoFCLogitModel
from .nano_conv_logit_model import NanoConvLogitModel
from .deep_conv_fc_logit_model import DeepConvFCLogitModel
from .deep_conv_conv_logit_model import DeepConvConvLogitModel
from .res_conv_fc_logit_model import ResConvFCLogitModel
from .res_conv_conv_logit_model import ResConvConvLogitModel
from .res_conv_conv_logit_pool_model import ResConvConvLogitPoolModel
from .res_conv_conv_logit_pool_model_v2 import ResConvConvLogitPoolModelV2
from .u_conv_fc_logit_model import UConvFCLogitModel
from .u_conv_conv_logit_model import UConvConvLogitModel
from .connect4_benchmark_model import Connect4BenchModel

from .mymodel2 import lalala
from .u_conv_conv_logit_3Dmodel import UConvConvLogitModel3D
from .u_conv_conv_logit_3Dmodelmonai import UConvConvLogitModel3Dmonai
from .mod_ucaps import ModifiedUCaps3D
from .mod_ucapsdaizhushi import ModifiedUCaps3Ddaizhushi
from .unetmonai import UNetMonai

from .utils import MODELS  # directory where models are registered

#在简单网络中使用efficaps2d
from .nano_efficaps2d import NanoEffiCaps2D

from .resunet import Resunet
from .resunetupdownsample import Resunetupdown # 多尺度
from .attenunet import attention # 初始得attentionnet
from .resunetatten import Resunetatten #往resnet里加attention   做实验用的
from .ghostunet1 import Ghostunet1 #用ghostV2替换了Resunet块
from .ghostunet2 import Ghostunet2SE #准备在Skip中加入SE
from .ghostunet3 import Ghostunet2SECap#准备Ghost中加入胶囊
from .ghostunet4 import Ghostunet2SECapCA#TibetanGoTinyNetv1
from .ghostunet451  import Ghostunet2SECapCA51 #TibetanGoTinyNetv2

from .nano_caps2d import NanoCaps2D #测试2Dcapsule