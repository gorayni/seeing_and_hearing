from __future__ import print_function, division
import torch.nn as nn


class LateFusion(nn.Module):
    def __init__(self, fusion_name, num_classes, arc=None):
        super(LateFusion, self).__init__()
        features_size = 0
        if 'rgb' in fusion_name:
            if arc == 'resnet18':
                features_size += 512
            else:
                features_size += 2048
        if 'flow' in fusion_name:
            if arc == 'resnet18':
                features_size += 512
            else:
                features_size += 2048
        if 'audio_vgg' in fusion_name:
            features_size += 4096
        if 'audio_traddil' in fusion_name:
            features_size += 256

        self.classifier = nn.Sequential(
            nn.Linear(features_size, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
