# coding=utf-8
# Copyright 2018 jose.fonollosa@upc.edu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import print_function, division
import math
import torch.nn as nn


class VGG(nn.Module):

    def __init__(self, vgg_name, num_classes=31):
        super(VGG, self).__init__()
        self.features = make_layers(cfg[vgg_name])
        self.classifier = nn.Sequential(
            nn.Linear(1 * 70 * 512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x.unsqueeze_(1)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def make_layers(cfg, batch_norm=True, in_channels=1):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'U':
            layers += [nn.Upsample(scale_factor=2, mode='nearest')]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class Traditional(nn.Module):
    
    def __init__(self, num_classes=26, batch_norm=True, in_channels=1):
        super(Traditional, self).__init__()
        
        layers = [nn.Conv2d(in_channels, 64, kernel_size=(110, 62), padding=(54,30))]
        if batch_norm:
            layers += [nn.BatchNorm2d(64)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]        
        layers += [nn.Conv2d(64, 64, kernel_size=(55, 31), padding=(27,15))]        
        self.features = nn.Sequential(*layers)        
        self.classifier = nn.Sequential(
            nn.Linear(1*20295*64, 50),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(50, num_classes)
        )
        self._initialize_weights()
        
    def forward(self, x):
        x.unsqueeze_(1)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class TraditionalDilated(nn.Module):
    
    def __init__(self, num_classes=26, batch_norm=False, in_channels=1):
        super(TraditionalDilated, self).__init__()        
        layers = [nn.Conv2d(in_channels, 64, kernel_size=(11, 7), dilation=(9,4), padding=(45,12))]
        layers += [nn.BatchNorm2d(64)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        layers += [nn.Conv2d(64, 32, kernel_size=(6, 4), dilation=(9,4), padding=(23,6))]
        layers += [nn.BatchNorm2d(32), nn.ReLU(inplace=True)]                
        layers += [nn.Conv2d(32, 16, kernel_size=(6, 4), dilation=(9,4), padding=(23,6))]
        layers += [nn.BatchNorm2d(16), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(16, 8, kernel_size=(6, 4), dilation=(9,4), padding=(23,6))]
        layers += [nn.BatchNorm2d(8), nn.ReLU(inplace=True)]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        self.features = nn.Sequential(*layers)        
        self.classifier = nn.Sequential(
            nn.Linear(1*5208*8, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, num_classes)
        )
        self._initialize_weights()
        
    def forward(self, x):
        x.unsqueeze_(1)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class TraditionalDilated16(nn.Module):
    
    def __init__(self, num_classes=26, batch_norm=False, in_channels=1):
        super(TraditionalDilated16, self).__init__()        
        layers = [nn.Conv2d(in_channels, 64, kernel_size=(11, 7), dilation=(9,4), padding=(45,12))]
        layers += [nn.BatchNorm2d(64)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        layers += [nn.Conv2d(64, 32, kernel_size=(6, 4), dilation=(9,4), padding=(23,6))]
        layers += [nn.BatchNorm2d(32), nn.ReLU(inplace=True)]                
        layers += [nn.Conv2d(32, 16, kernel_size=(6, 4), dilation=(9,4), padding=(23,6))]
        layers += [nn.BatchNorm2d(16), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(16, 8, kernel_size=(6, 4), dilation=(9,4), padding=(23,6))]
        layers += [nn.BatchNorm2d(8), nn.ReLU(inplace=True)]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        self.features = nn.Sequential(*layers)        
        self.classifier = nn.Sequential(
            nn.Linear(1*5208*8, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, num_classes)
        )
        self._initialize_weights()
        
    def forward(self, x):
        x.unsqueeze_(1)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
