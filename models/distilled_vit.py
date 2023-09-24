import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.data_utils import to_2tuple
from models.vit import VisionTransformer, Identity

class DistilledVisionTransformer(VisionTransformer):

    def __init__(self,
                 img_size=384,
                 patch_size=16,
                 class_dim=100,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4,
                 qkv_bias=False,
                 norm_layer=nn.LayerNorm,
                 epsilon=1e-5,
                 **kwargs):
        super(DistilledVisionTransformer, self).__init__(
            img_size=img_size,
            patch_size=patch_size,
            class_dim=class_dim,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            epsilon=epsilon,
            **kwargs
        )
        # 增加 distillation token, 调整相应的位置编码长度
        self.pos_embed = nn.Parameter(torch.zeros(size=(1, self.patch_embed.num_patches + 2, self.embed_dim)))
        # distillation token
        self.dist_token = nn.Parameter(torch.zeros(size=(1, 1, self.embed_dim)))

        self.register_parameter("pos_embed", self.pos_embed)
        self.register_parameter("cls_token", self.cls_token)

        # Classifier head
        self.head_dist = nn.Linear(
            self.embed_dim,
            self.class_dim
        ) if self.class_dim > 0 else Identity()

        nn.init.trunc_normal_(self.dist_token)
        nn.init.trunc_normal_(self.pos_embed)
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        B = x.shape[0]
        # 将图片分块,并调整每个块向量的维度
        x = self.patch_embed(x)
        # 将class token、distillation token与前面的分块进行拼接
        cls_token = self.cls_token.expand((B, -1, -1))
        dist_token = self.dist_token.expand((B, -1, -1))
        x = torch.cat((cls_token, dist_token, x), dim=1)
        # 将编码向量中加入位置编码
        x = x + self.pos_embed
        x = self.pos_drop(x)
        # 堆叠 transformer 结构
        for blk in self.blocks:
            x = blk(x)
        # LayerNorm
        x = self.norm(x)
        # 提取class token以及distillation token的输出
        return x[:, 0], x[:, 1]

    def forward(self, x):
        # 获取图像特征
        x, x_dist = self.forward_features(x)
        # 图像分类
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        # 取 class token以及distillation token 的平均值作为结果
        return (x + x_dist) / 2

