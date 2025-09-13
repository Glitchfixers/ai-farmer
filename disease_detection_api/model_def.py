import torch
import torch.nn as nn
from torchvision import models

class ImprovedCNNViTHybrid(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        self.backbone = models.resnet50(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        self.feature_dim = 2048
        self.patch_size = 7
        self.num_patches = 49
        self.embedding_dim = 768

        self.feature_projection = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Conv2d(self.feature_dim, self.embedding_dim, kernel_size=1),
            nn.BatchNorm2d(self.embedding_dim),
            nn.ReLU(inplace=True)
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embedding_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, self.embedding_dim))
        self.dropout = nn.Dropout(0.3)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim, nhead=8, dim_feedforward=2048,
            dropout=0.3, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Dropout(0.3),
            nn.Linear(self.embedding_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        features = self.backbone(x)
        features = self.feature_projection(features)
        features = features.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        features = torch.cat([cls_tokens, features], dim=1)
        features = features + self.pos_embed
        features = self.dropout(features)
        encoded = self.transformer(features)
        cls_output = encoded[:, 0]
        return self.classifier(cls_output)
