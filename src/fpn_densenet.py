from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

# from torchvision.models.densenet import _DenseBlock, _Transition


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        self.n_in_features = num_input_features
        self.n_out_features = num_input_features + num_layers * growth_rate
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.n_in_features = num_input_features
        self.n_out_features = num_output_features
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class FPNDenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0):

        super(FPNDenseNet, self).__init__()

        # First convolution
        self.pre_conv = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=1, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        self.dense1 = _DenseBlock(num_layers=block_config[0], num_input_features=num_init_features,
                                  bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        self.trans1 = _Transition(num_input_features=self.dense1.n_out_features,
                                  num_output_features=self.dense1.n_out_features // 2)
        self.dense2 = _DenseBlock(num_layers=block_config[1], num_input_features=self.trans1.n_out_features,
                                  bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        self.trans2 = _Transition(num_input_features=self.dense2.n_out_features,
                                  num_output_features=self.dense2.n_out_features // 2)
        self.dense3 = _DenseBlock(num_layers=block_config[2], num_input_features=self.trans2.n_out_features,
                                  bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        self.trans3 = _Transition(num_input_features=self.dense3.n_out_features,
                                  num_output_features=self.dense3.n_out_features // 2)
        self.dense4 = _DenseBlock(num_layers=block_config[3], num_input_features=self.trans3.n_out_features,
                                  bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)

        # Final batch norm
        self.dense4_bn = nn.BatchNorm2d(self.dense4.n_out_features)

        # Establish FPN Structure
        n_fpn_channels = self.dense1.n_out_features

        self.dense1_lateral = nn.Conv2d(self.dense1.n_out_features, self.dense2.n_out_features, kernel_size=1)
        self.dense2_lateral = nn.Conv2d(self.dense2.n_out_features, self.dense3.n_out_features, kernel_size=1)
        self.dense3_lateral = nn.Conv2d(self.dense3.n_out_features, self.dense4.n_out_features, kernel_size=1)

        self.fpn2_up = nn.ConvTranspose2d(self.dense2.n_out_features, self.dense2.n_out_features, kernel_size=3,
                                          stride=2, padding=1, output_padding=1)
        self.fpn3_up = nn.ConvTranspose2d(self.dense3.n_out_features, self.dense3.n_out_features, kernel_size=3,
                                          stride=2, padding=1, output_padding=1)
        self.fpn4_up = nn.ConvTranspose2d(self.dense4.n_out_features, self.dense4.n_out_features, kernel_size=3,
                                          stride=2, padding=1, output_padding=1)

        self.fpn1_conv = nn.Conv2d(self.dense2.n_out_features, n_fpn_channels, kernel_size=3, padding=1)
        self.fpn2_conv = nn.Conv2d(self.dense3.n_out_features, n_fpn_channels, kernel_size=3, padding=1)
        self.fpn3_conv = nn.Conv2d(self.dense4.n_out_features, n_fpn_channels, kernel_size=3, padding=1)
        self.fpn4_conv = nn.Conv2d(self.dense4.n_out_features, n_fpn_channels, kernel_size=3, padding=1)

        self.head_cls = nn.Conv2d(n_fpn_channels, 3 * 2, 1)
        self.head_reg = nn.Conv2d(n_fpn_channels, 3 * 8, 1)

        # weights / bias initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)

    def forward(self, x):
        x = self.pre_conv(x)
        x = dense1 = self.dense1(x)
        x = self.trans1(x)
        x = dense2 = self.dense2(x)
        x = self.trans2(x)
        x = dense3 = self.dense3(x)
        x = self.trans3(x)
        dense4 = self.dense4_bn(self.dense4(x))

        fpn4 = self.fpn4_conv(dense4)
        fpn3 = self.fpn3_conv(self.fpn4_up(dense4) + self.dense3_lateral(dense3))
        fpn2 = self.fpn2_conv(self.fpn3_up(dense3) + self.dense2_lateral(dense2))
        fpn1 = self.fpn1_conv(self.fpn2_up(dense2) + self.dense1_lateral(dense1))

        fpn4_cls = self.head_cls(fpn4)
        fpn4_reg = self.head_reg(fpn4)
        fpn3_cls = self.head_cls(fpn3)
        fpn3_reg = self.head_reg(fpn3)
        fpn2_cls = self.head_cls(fpn2)
        fpn2_reg = self.head_reg(fpn2)
        fpn1_cls = self.head_cls(fpn1)
        fpn1_reg = self.head_reg(fpn1)

        return fpn1_cls, fpn1_reg, fpn2_cls, fpn2_reg, fpn3_cls, fpn3_reg, fpn4_cls, fpn4_reg
