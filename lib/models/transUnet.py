import os

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torchvision.utils as vutils
import torch.nn.functional as F

from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .layers import CBAM
from einops.layers.torch import Rearrange
from .unet_parts import *

import math


# feed forward
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):

        qkv_bias = True
        qk_scale = None
        attn_drop = 0.
        proj_drop = 0.

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):

        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]  # make torch script happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # add position embedding
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# swin transformer block
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dropout=0.):
        super(TransformerBlock, self).__init__()
        self.mlp_ratio = 4.
        self.norm = nn.LayerNorm(dim)
        self.att = Attention(dim, heads=heads, dim_head=dim // heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim, int(self.mlp_ratio * dim)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(self.mlp_ratio * dim), dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.att(self.norm(x)) + x
        x = self.ffn(self.norm(x)) + x
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=2, shift_size=False, drop_path=0.):
        super().__init__()

        mlp_ratio = 4.
        qkv_bias = True
        qk_scale = None
        drop = 0.
        attn_drop = 0.
        drop_path = 0.
        act_layer = nn.GELU
        norm_layer = nn.LayerNorm

        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        if shift_size:
            self.shift_size = window_size // 2
        else:
            self.shift_size = 0
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(self.dim)

        self.attn = WindowAttention(dim, window_size=to_2tuple(self.window_size), num_heads=num_heads)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # calculate attention mask for SW-MSA
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):

        H, W = self.input_resolution

        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)

        x = x.view(B, H, W, C)
        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    def __init__(self, dim, input_resolution, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)
        # todo
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C
        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchExpand(nn.Module):
    def __init__(self, dim, input_resolution, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = norm_layer(dim // 2)

    def forward(self, x):
        """
        x: B, H*W, C
        """

        H, W = self.input_resolution
        x = self.expand(x)

        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)

        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=64, norm_layer=None):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]

        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x


class AttBlock(nn.Module):
    def __init__(self, in_channels, opt, input_resolution):
        super().__init__()
        if opt.attention == "swintf":
            self.att = SwinTransformerBlock(in_channels, input_resolution, num_heads=4, drop_path=opt.drop_rate)
            self.att2 = SwinTransformerBlock(in_channels, input_resolution, num_heads=4, drop_path=opt.drop_rate,
                                             shift_size=True)
        elif opt.attention == "tf":
            self.att = TransformerBlock(in_channels, 4, dropout=opt.drop_rate)
            self.att2 = TransformerBlock(in_channels, 4, dropout=opt.drop_rate)
        elif opt.attention == "fca":
            c2wh = dict([(64, 56), (128, 28), (256, 14), (512, 7), (1024, 1)])
            self.att = nn.Sequential(
                Rearrange('b (h d) c -> b c h d', h=input_resolution[0]),
                MultiSpectralAttentionLayer(in_channels, c2wh[in_channels], c2wh[in_channels])
            )
            self.att2 = nn.Sequential(
                MultiSpectralAttentionLayer(in_channels, c2wh[in_channels], c2wh[in_channels]),
                Rearrange('b c h d -> b (h d) c', h=input_resolution[0])
            )

        self.ln = nn.LayerNorm(in_channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.att(x)
        # x = self.ln(x)
        x = self.att2(x)
        return self.act(self.ln(x))


class AttDown(nn.Module):
    def __init__(self, in_channels, input_resolution, opt, last=False):
        super().__init__()

        self.att = AttBlock(in_channels, opt, input_resolution=(input_resolution[0], input_resolution[1]))
        # self.proj = nn.Linear(in_channels, out_channels)
        if not last:
            self.down = PatchMerging(in_channels, input_resolution)
        else:
            self.down = nn.Identity()

    def forward(self, x):
        x = self.att(x)
        return self.down(x)


class AttUp(nn.Module):
    def __init__(self, in_channels, input_resolution, opt, last=False):
        super().__init__()
        self.contact = nn.Linear(in_channels * 2, in_channels)
        self.att = AttBlock(in_channels, opt, input_resolution=(input_resolution[0], input_resolution[1]))
        # self.proj = nn.Linear(in_channels, out_channels)
        if not last:
            self.up = PatchExpand(in_channels, input_resolution)
        else:
            self.up = nn.Identity()

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], -1)

        x = self.contact(x)
        x = self.att(x)
        return self.up(x)


class PositionEmbedding(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super(PositionEmbedding, self).__init__()
        self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        trunc_normal_(self.absolute_pos_embed, std=.02)

    def forward(self, x):
        return x + self.absolute_pos_embed


class Adjust(nn.Module):
    def __init__(self, scale):
        super(Adjust, self).__init__()
        self.scale = scale

    def forward(self, x):
        B, L, C = x.shape
        size = math.sqrt(L)
        new_size = int(size * self.scale)
        new_x = x.view(B, new_size * new_size, -1)
        return new_x

class Embedding(nn.Module):
    def __init__(self, opt, scale=1):
        super(Embedding, self).__init__()
        size = opt.isize // scale
        patch_resolution = size // opt.patchsize
        patch_resolution = patch_resolution * patch_resolution
        self.bn = nn.BatchNorm2d(3)
        self.Emb = nn.Sequential(PatchEmbed(img_size=size, patch_size=opt.patchsize, in_chans=opt.nc,
                                            embed_dim=opt.dim * (2 ** (scale // 2)),
                                            norm_layer=nn.LayerNorm),
                                 PositionEmbedding(patch_resolution, opt.dim * (2 ** (scale // 2))))

    def forward(self, x):
        return self.Emb(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()

        # 输入的维度
        inner_dim = dim_head * heads

        # 只有一个head就不投影
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads

        self.scale = dim_head ** -0.5

        # 对某一维度的行进行softmax计算
        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class SwinUNet(nn.Module):
    def __init__(self, input_nc, opt):
        super(SwinUNet, self).__init__()

        self.n_channels = input_nc
        self.n_classes = 3
        self.input_resolution = (opt.isize // opt.patchsize, opt.isize // opt.patchsize)
        self.dim = opt.dim
        self.patch_size = opt.patchsize
        self.depth = 3

        self.inc = Embedding(opt)
        self.inc2 = Embedding(opt, 2)
        self.inc4 = Embedding(opt, 4)

        self.pos_drop = nn.Dropout(p=0)

        self.down = nn.ModuleList()
        self.skip = nn.ModuleList()

        for i in range(self.depth):  # 0-4
            if i == self.depth - 1:
                self.skip.append(nn.Identity())
            else:
                self.skip.append(nn.Sequential(
                    AttBlock(self.dim * (2 ** i), opt,
                             (self.input_resolution[0] // (2 ** i), self.input_resolution[1] // (2 ** i)))
                ))
            self.down.append(AttDown(self.dim * (2 ** i),
                                     (self.input_resolution[0] // (2 ** i), self.input_resolution[1] // (2 ** i)), opt))

        self.bottom = nn.Sequential(
            AttDown(self.dim * (2 ** self.depth), (
                self.input_resolution[0] // (2 ** self.depth), self.input_resolution[1] // (2 ** self.depth)), opt,
                    last=True),
            PatchExpand(self.dim * (2 ** self.depth), (
                self.input_resolution[0] // (2 ** self.depth), self.input_resolution[1] // (2 ** self.depth)))
        )

        self.up = nn.ModuleList()
        for j in range(self.depth):
            i = self.depth - j - 1
            self.up.append(AttUp(self.dim * (2 ** i),
                                 (self.input_resolution[0] // (2 ** i),
                                  self.input_resolution[1] // (2 ** i)), opt))

        self.expand = nn.Linear(self.dim // 2, input_nc * (self.patch_size // 2) ** 2)
        self.adjust_up = Adjust(2)
        self.adjust_down = Adjust(0.5)
        self.out = nn.Sequential(
            nn.Conv2d(in_channels=input_nc, out_channels=3, kernel_size=1, bias=False)
        )

    def _get_initial_point(self):
        patch_size_w = self.feat_size / self.num_point_w
        patch_size_h = self.feat_size / self.num_point_h
        coord_w = torch.Tensor(
            [i * patch_size_w for i in range(self.num_point_w)])
        coord_w += patch_size_w / 2
        coord_h = torch.Tensor(
            [i * patch_size_h for i in range(self.num_point_h)])
        coord_h += patch_size_h / 2

        grid_x, grid_y = torch.meshgrid(coord_w, coord_h)
        grid_x = grid_x.unsqueeze(0)
        grid_y = grid_y.unsqueeze(0)
        point_coord = torch.cat([grid_y, grid_x], dim=0)
        point_coord = point_coord.view(2, -1)
        point_coord = point_coord.permute(1, 0).contiguous().unsqueeze(0)

        return point_coord

    def output(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        S = self.patch_size
        x = self.expand(x)
        x = x.view(B, S * H, S * W, -1)
        x = x.permute(0, 3, 1, 2)  # B,C,H,W

        return self.out(x)

    def forward(self, x):

        # B * C * H * W
        x = self.inc(x)  # B W/2*H/2 2C
        down_sample = []
        for inx in range(len(self.down)):
            down_sample.append(self.skip[inx](x))
            x = self.down[inx](x)

        x = self.bottom(x)

        for inx in range(len(self.up)):
            x = self.up[inx](x, down_sample[self.depth - inx - 1])

        x = self.output(x)
        return x
