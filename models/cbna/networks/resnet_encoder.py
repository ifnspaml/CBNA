import torch
import torch.nn as nn
import torchvision.models as models

RESNETS = {
    18: models.resnet18,
    34: models.resnet34,
    50: models.resnet50,
    101: models.resnet101,
    152: models.resnet152
}


class MyBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, *args, **kwargs):
        super(MyBatchNorm2d, self).__init__(*args, **kwargs)

    def forward(self, input):
        running_mean_temp = input.mean(dim=(0, 2, 3))
        running_var_temp = input.var(dim=(0, 2, 3))
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * running_mean_temp
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * running_var_temp

        return super().forward(input)


class ResnetEncoder(nn.Module):
    """A ResNet that handles multiple input images and outputs skip connections"""

    def __init__(self, num_layers, pretrained, cbna_bn_inference):
        super().__init__()

        if num_layers not in RESNETS:
            raise ValueError(f"{num_layers} is not a valid number of resnet layers")

        self.encoder = RESNETS[num_layers](pretrained)

        if cbna_bn_inference:
            encoder_state_dict = self.encoder.state_dict()

            def replace_bn(model, old, new):
                for n, module in model.named_children():
                    if len(list(module.children())) > 0:
                        ## compound module, go inside it
                        replace_bn(module, old, new)

                    if isinstance(module, old):
                        ## simple module
                        setattr(model, n, new(module.num_features,
                                              momentum=module.momentum,
                                              affine=module.affine,
                                              track_running_stats=module.track_running_stats))

            replace_bn(self.encoder, nn.BatchNorm2d, MyBatchNorm2d)

            self.encoder.load_state_dict(encoder_state_dict)

        # Remove fully connected layer
        self.encoder.fc = None

        if num_layers > 34:
            self.num_ch_enc = (64, 256,  512, 1024, 2048)
        else:
            self.num_ch_enc = (64, 64, 128, 256, 512)

    def forward(self, l_0):
        l_0 = self.encoder.conv1(l_0)
        l_0 = self.encoder.bn1(l_0)
        l_0 = self.encoder.relu(l_0)

        l_1 = self.encoder.maxpool(l_0)
        l_1 = self.encoder.layer1(l_1)

        l_2 = self.encoder.layer2(l_1)
        l_3 = self.encoder.layer3(l_2)
        l_4 = self.encoder.layer4(l_3)

        return (l_0, l_1, l_2, l_3, l_4)
