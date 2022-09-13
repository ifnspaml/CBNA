import torch.nn as nn

from models.cbna_vgg.networks.vgg_encoder import MyBatchNorm2d as BN_VGG
from models.cbna.networks.resnet_encoder import MyBatchNorm2d as BN_ResNet


class BNAdaptation(object):
    def __init__(self):
        pass

    def process(self, model, momentum):
        # Check for architecture of encoder, because layers are accessed differently for different architectures
        if model._get_name() == 'UBNAVGG':
            for module in model.common.encoder.encoder.features.modules():
                if type(module) == nn.BatchNorm2d or type(module) == BN_VGG:
                    module.momentum = momentum
        else:
            for module in model.common.encoder.encoder.modules():
                if type(module) == nn.BatchNorm2d or type(module) == BN_ResNet:
                    module.momentum = momentum
        return model