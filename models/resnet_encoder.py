# models/resnet_encoder.py

import torch.nn as nn
import torchvision.models as models
from .projection_head import ProjectionHead

class ResNetSimCLR(nn.Module):
    def __init__(self, base_model='resnet18', out_dim=128, pretrained=False):
        super(ResNetSimCLR, self).__init__()

        assert base_model in ['resnet18', 'resnet50'], "Only resnet18 or resnet50 are supported"

        if base_model == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
        else:
            self.backbone = models.resnet50(pretrained=pretrained)

        # Remove the final classification layer
        feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.projection_head = ProjectionHead(feat_dim, feat_dim, out_dim)

    def forward(self, x):
        h = self.backbone(x)
        z = self.projection_head(h)
        return h, z
