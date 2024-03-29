import torch.nn as nn
import torchvision.models as models

VGG_BN = {
    11: models.vgg11_bn,
    13: models.vgg13_bn,
    16: models.vgg16_bn,
    19: models.vgg19_bn
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


class VggEncoder(nn.Module):
    """A ResNet that handles multiple input images and outputs skip connections"""

    def __init__(self, num_layers, pretrained, cbna_bn_inference):
        super().__init__()

        if num_layers not in VGG_BN:
            raise ValueError(f"{num_layers} is not a valid number of resnet layers")

        self.encoder = VGG_BN[num_layers](pretrained)

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

        # Prepare encoder for skip connections
        modules = dict()
        conv1_modules = list()
        conv2_modules = list()
        conv3_modules = list()
        conv4_modules = list()
        conv5_modules = list()
        modules['conv1'] = conv1_modules
        modules['conv2'] = conv2_modules
        modules['conv3'] = conv3_modules
        modules['conv4'] = conv4_modules
        modules['conv5'] = conv5_modules
        conv_numbering = 1
        module_numbering = 0
        for module in self.encoder.features.modules():
            # Dont add first module as this contains the whole sequential wrapper
            if module_numbering > 0 and conv_numbering < 6:
                modules['conv' + str(conv_numbering)].append(module)
            if type(module) == nn.MaxPool2d:
                conv_numbering += 1
            module_numbering += 1
        self.encoder.conv1 = nn.Sequential(*modules['conv1'])
        self.encoder.conv2 = nn.Sequential(*modules['conv2'])
        self.encoder.conv3 = nn.Sequential(*modules['conv3'])
        self.encoder.conv4 = nn.Sequential(*modules['conv4'])
        self.encoder.conv5 = nn.Sequential(*modules['conv5'])

        # Remove fully connected layer
        # self.encoder.fc = None
        self.num_ch_enc = (64, 128, 256, 512, 512)

    def forward(self, l_0):
        # l_0 = (l_0 - 0.45) / 0.225 # weird normalization from monodepth2, better do this in data preprocessing
        l_0 = self.encoder.conv1(l_0)
        l_1 = self.encoder.conv2(l_0)
        l_2 = self.encoder.conv3(l_1)
        l_3 = self.encoder.conv4(l_2)
        l_4 = self.encoder.conv5(l_3)

        return (l_0, l_1, l_2, l_3, l_4)
