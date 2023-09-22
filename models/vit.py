import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.data_utils import to_2tuple

class VisionTransformer(nn.Module):

    def __init__(self,
                 img_size=384,
                 patch_size=16,
                 in_chans=3,
                 class_dim=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 epsilon=1e-5,
                 **kwargs):
        super(VisionTransformer, self).__init__()
        self.class_dim = class_dim

        self.num_features = self.embed_dim = embed_dim
        # 图片分块并对其进行降维, 块大小为patch_size, 最终块向量维度为embed_dim
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        # patch数量
        num_patchs = self.patch_embed.num_patches

        # learnable position encoding
        self.pos_embed = nn.Parameter(torch.zeros(size=(1, num_patchs + 1, embed_dim)))
        # 添加class token,并使用该向量进行分类预测
        self.cls_token = nn.Parameter(torch.zeros(size=(1, 1, embed_dim)))

        self.register_parameter('pos_embed', self.pos_embed)
        self.register_parameter('cls_token', self.cls_token)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = torch.linspace(0, drop_path_rate, depth)
        # transformer
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                epsilon=epsilon
            )
            for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim, eps=epsilon)

        # Classifier head
        self.head = nn.Linear(embed_dim, class_dim) if class_dim > 0 else Identity()

        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward_features(self, x):
        B = x.shape[0]
        # 将图片分块, 并调整每个块向量的维度
        x = self.patch_embed(x)
        # 将class token与前面的分块进行拼接
        cls_tokens = self.cls_token.expand((B, -1, -1))
        x = torch.cat((cls_tokens, x), dim=1)
        # 将pos encoding加入其中
        x = x + self.pos_embed
        x = self.pos_drop(x)
        # 堆叠transformer结构
        for blk in self.blocks:
            x = blk(x)
        # LayerNorm
        x = self.norm(x)
        # 提取分类 tokens 的输出
        return x[:, 0]

    def forward(self, x):
        # 获取图像特征
        x = self.forward_features(x)
        # 图像分类
        x = self.head(x)

        return x


class PatchEmbed(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        """
        图像分块
        :param img_size: 图像的尺寸
        :param patch_size:  图像块的尺寸
        :param in_chans:    传入通道数
        :param embed_dim:   embedding特征维度
        """
        super(PatchEmbed, self).__init__()
        # 原始大小为int, 转为tuple, 即: img_size原始输入为224, 变换后为[224, 224]
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        # 图像块的个数
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        # kernel_size=块大小,即每个块输出一个值,类似每个块展平后使用相同的全连接层进行处理
        # 输入: 输入维度为3, 输出维度为块向量长度
        #   分块、展平、全连接降维保持一致
        # 输出: [B, C, H, W]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})"
        # [B, C, H, W] -> [B, C, H*W] -> [B, H*W, C]
        x = self.proj(x).flatten(2).permute(0, 2, 1)
        return x


class Attention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        """
        注意力机制计算
        :param dim: 传入的embedding特征维度
        :param num_heads: 多头注意力数量
        :param qkv_bias: 是否使用qkv矩阵bias
        :param qk_scale: qk放缩scale值
        :param attn_drop: 注意力的dropout概率
        :param proj_drop: 映射proj的dropout概率
        """
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** (-0.5)

        # 计算 q， k，v的转移矩阵
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        # 最终的线性层
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        N, C = x.shape[1:]  # x.shape: [B, N, C]
        # 线性变换
        qkv = self.qkv(x).reshape((-1, N, 3, self.num_heads,
                                   C // self.num_heads)).permute(2, 0, 3, 1, 4)
        # 分割q,k,v
        q, k, v = qkv[0], qkv[1], qkv[2]
        # Scaled Dot-Product Attention
        # Matmul + Scale
        attn = (q.matmul(k.permute(0, 1, 3, 2))) * self.scale
        # SoftMax
        # TODO: softmax需要进行更改
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        # Matmul
        x = (attn.matmul(v)).permute(0, 2, 1, 3).reshape((-1, N, C))
        # 线性变换
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MLP(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        """
        多层感知机
        :param in_features: 输入特征维度
        :param hidden_features: 隐藏层
        :param out_features:    输出层
        :param act_layer:   激活函数
        :param drop:    dropout
        """
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act()(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


def drop_path(x, drop_prob=0., training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = torch.tensor(1 - drop_prob)
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype)
    random_tensor = torch.floor(random_tensor)
    output = x.divide(keep_prob) * random_tensor

    return output


class DropPath(nn.Module):

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4,
                 qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, epsilon=1e-5):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim, eps=epsilon)
        # Multi-head Self-attention
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        # DropPath
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.norm2 = norm_layer(dim, eps=epsilon)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)

    def forward(self, x):
        # Multi-head Self-attention, Add, LayerNorm
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # Feed Forwards, Add, LayerNorm
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input

if __name__ == '__main__':
    model = VisionTransformer()
    test_in = torch.rand(size=(2, 3, 384, 384))
    print(model)

    test_out = model(test_in)
    print(test_out)