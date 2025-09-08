# Module.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'ConvBN', 'Block', 'StarNet', 'DropPath', 'Conv', 'autopad', 'LayerNorm',
    'SpatialOperation', 'ChannelOperation', 'AdditiveTokenMixer', 'CAFE',
    'AFFM', 'EUCB', 'MSDC', 'MSCB', 'CMFF', 'AMFB', 'RTDETRDecoder'
]

# ----------------------------
# StarNet Components
# ----------------------------
class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', nn.BatchNorm2d(out_planes))
            nn.init.constant_(self.bn.weight, 1)
            nn.init.constant_(self.bn.bias, 0)


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, 1, 3, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, 3, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        identity = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        return identity + self.drop_path(x)


class StarNet(nn.Module):
    def __init__(self, base_dim=16, depths=(1, 1, 3, 1), mlp_ratio=3):
        super().__init__()
        self.stem = nn.Sequential(ConvBN(3, base_dim, 3, 2, 1), nn.ReLU6())
        in_ch = base_dim
        self.stages = nn.ModuleList()
        for i, d in enumerate(depths):
            embed_dim = base_dim * (2 ** i)
            down = ConvBN(in_ch, embed_dim, 3, 2, 1)
            blocks = [Block(embed_dim, mlp_ratio) for _ in range(d)]
            in_ch = embed_dim
            self.stages.append(nn.Sequential(down, *blocks))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.trunc_normal_(m.weight, std=.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        feats = []
        x = self.stem(x)
        feats.append(x)  # P2
        for stage in self.stages:
            x = stage(x)
            feats.append(x)  # P3, P4, P5
        return feats


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


# ----------------------------
# Basic Conv Module
# ----------------------------
class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


def autopad(k, p=None, d=1):
    """Calculate padding automatically."""
    if p is None:
        p = (k - 1) // 2 * d
    return p


# ----------------------------
# LayerNorm (channels_first version)
# ----------------------------
class LayerNorm(nn.Module):
    """channels_first version of LayerNorm"""
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


# ----------------------------
# Spatial Operation
# ----------------------------
class SpatialOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, 1, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)


# ----------------------------
# Channel Operation
# ----------------------------
class ChannelOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)


# ----------------------------
# Additive Token Mixer
# ----------------------------
class AdditiveTokenMixer(nn.Module):
    def __init__(self, dim=512, attn_bias=False, proj_drop=0.):
        super().__init__()
        self.qkv = nn.Conv2d(dim, 3 * dim, 1, bias=attn_bias)
        self.oper_q = nn.Sequential(SpatialOperation(dim), ChannelOperation(dim))
        self.oper_k = nn.Sequential(SpatialOperation(dim), ChannelOperation(dim))
        self.dwc = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.proj = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=1)
        q = self.oper_q(q)
        k = self.oper_k(k)
        out = self.proj(self.dwc(q + k) * v)
        return self.proj_drop(out)


# ----------------------------
# CAFE (formerly TransformerEncoderLayer_AdditiveTokenMixer)
# ----------------------------
class CAFE(nn.Module):
    def __init__(self, c1, cm=2048, num_heads=8, dropout=0.0):
        super().__init__()
        self.additivetoken = AdditiveTokenMixer(c1)
        self.fc1 = nn.Conv2d(c1, cm, 1)
        self.fc2 = nn.Conv2d(cm, c1, 1)
        self.norm1 = LayerNorm(c1)
        self.norm2 = LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, src):
        src2 = self.additivetoken(src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src))))
        src = src + self.dropout2(src2)
        return self.norm2(src)


# ----------------------------
# AFFM (formerly Fusion)
# ----------------------------
class AFFM(nn.Module):
    def __init__(self, inc_list):
        super().__init__()
        self.fusion_weight = nn.Parameter(torch.ones(len(inc_list), dtype=torch.float32), requires_grad=True)
        self.relu = nn.ReLU()
        self.epsilon = 1e-4

    def forward(self, x):
        fusion_weight = self.relu(self.fusion_weight.clone())
        fusion_weight = fusion_weight / (torch.sum(fusion_weight, dim=0) + self.epsilon)
        return torch.sum(torch.stack([fusion_weight[i] * x[i] for i in range(len(x))], dim=0), dim=0)


# ----------------------------
# Efficient Up-Convolution Block (EUCB)
# ----------------------------
class EUCB(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv(self.in_channels, self.in_channels, kernel_size, g=self.in_channels, s=stride)
        )
        self.pwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x):
        x = self.up_dwc(x)
        x = self.channel_shuffle(x, self.in_channels)
        x = self.pwc(x)
        return x

    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x


# ----------------------------
# Multi-Scale Depthwise Convolution (MSDC)
# ----------------------------
class MSDC(nn.Module):
    def __init__(self, in_channels, kernel_sizes, stride, dw_parallel=True):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_sizes = kernel_sizes
        self.dw_parallel = dw_parallel
        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                Conv(self.in_channels, self.in_channels, kernel_size, s=stride, g=self.in_channels)
            )
            for kernel_size in self.kernel_sizes
        ])

    def forward(self, x):
        outputs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            outputs.append(dw_out)
            if not self.dw_parallel:
                x = x + dw_out
        return outputs


# ----------------------------
# Multi-Scale Convolution Block (MSCB)
# ----------------------------
class MSCB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[1, 3, 5], stride=1, expansion_factor=2,
                 dw_parallel=True, add=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_sizes = kernel_sizes
        self.expansion_factor = expansion_factor
        self.dw_parallel = dw_parallel
        self.add = add
        self.n_scales = len(self.kernel_sizes)
        assert self.stride in [1, 2]
        self.use_skip_connection = True if self.stride == 1 else False

        self.ex_channels = int(self.in_channels * self.expansion_factor)
        self.pconv1 = nn.Sequential(Conv(self.in_channels, self.ex_channels, 1))
        self.msdc = MSDC(self.ex_channels, self.kernel_sizes, self.stride, dw_parallel=self.dw_parallel)
        self.combined_channels = self.ex_channels if self.add else self.ex_channels * self.n_scales
        self.pconv2 = nn.Sequential(Conv(self.combined_channels, self.out_channels, 1, act=False))
        if self.use_skip_connection and (self.in_channels != self.out_channels):
            self.conv1x1 = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0, bias=False)

    def forward(self, x):
        pout1 = self.pconv1(x)
        msdc_outs = self.msdc(pout1)
        dout = sum(msdc_outs) if self.add else torch.cat(msdc_outs, dim=1)
        dout = self.channel_shuffle(dout, math.gcd(self.combined_channels, self.out_channels))
        out = self.pconv2(dout)
        if self.use_skip_connection:
            if self.in_channels != self.out_channels:
                x = self.conv1x1(x)
            return x + out
        else:
            return out

    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x


# ----------------------------
# CMFF (formerly CSP_MSCB)
# ----------------------------
class CMFF(nn.Module):
    def __init__(self, c1, c2, n=1, kernel_sizes=[1, 3, 5], shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(MSCB(self.c, self.c, kernel_sizes=kernel_sizes) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ----------------------------
# AMFB (AFFM + 3x CMFF)
# ----------------------------
class AMFB(nn.Module):
    def __init__(self, channels, kernel_sizes):
        super().__init__()
        self.affm = AFFM([channels, channels])
        self.cmffs = nn.Sequential(
            CMFF(channels, channels, kernel_sizes=kernel_sizes),
            CMFF(channels, channels, kernel_sizes=kernel_sizes),
            CMFF(channels, channels, kernel_sizes=kernel_sizes)
        )

    def forward(self, x):
        # x should be a list of two tensors
        x = self.affm(x)
        return self.cmffs(x)


# ----------------------------
# RTDETRDecoder
# ----------------------------
class RTDETRDecoder(nn.Module):
    def __init__(self, nc=80, hidden_dim=256, num_queries=300, num_encoder_layers=4, num_decoder_layers=8):
        super().__init__()
        self.nc = nc
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.class_embed = nn.Linear(hidden_dim, nc + 1)
        self.bbox_embed = nn.Linear(hidden_dim, 4)
        
    def forward(self, x):
        if isinstance(x, list):
            x = x[-1]
        
        bs = x.shape[0]
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
        outputs_class = self.class_embed(query_embed)
        outputs_coord = self.bbox_embed(query_embed).sigmoid()
        
        return {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}
