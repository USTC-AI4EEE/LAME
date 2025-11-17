import math
import warnings
import torch
import torch.nn as nn

# Minimal local utils (avoid importing basicsr package)
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            'mean is more than 2 std from [a, b] in trunc_normal_. The distribution may be incorrect.',
            stacklevel=2,
        )

    with torch.no_grad():
        low = norm_cdf((a - mean) / std)
        up = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * low - 1, 2 * up - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def _ntuple(n):
    def parse(x):
        if isinstance(x, (tuple, list)):
            return x
        return (x,) * n

    return parse


to_2tuple = _ntuple(2)


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
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
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}"

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class CrossWindowAttention(nn.Module):
    """基于窗口的交叉注意力模块
    
    Args:
        dim (int): 输入通道数
        mem_dim (int): 记忆特征通道数
        window_size (tuple[int]): 窗口大小
        num_heads (int): 注意力头数
        qkv_bias (bool): 是否使用偏置，默认True
        qk_scale (float): 缩放因子，默认None
        attn_drop (float): 注意力dropout率，默认0.0
        proj_drop (float): 输出dropout率，默认0.0
    """

    def __init__(
        self,
        dim,
        mem_dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.mem_dim = mem_dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # 定义Q,K,V的线性变换
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(mem_dim, dim * 2, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mem=None, mask=None, pre_k=None, pre_v=None):
        """
        Args:
            x: 输入特征，形状为 (num_windows*B, N, C)
            mem: 记忆特征，形状为 (num_windows*B, N, C_mem)。当提供 pre_k/pre_v 时可为 None
            mask: 注意力掩码，形状为 (num_windows, Wh*Ww, Wh*Ww) 或 None
            pre_k, pre_v: 预计算好的 K/V，形状为 (num_windows*B, num_heads, N, head_dim)
        """
        B_, N, C = x.shape
        
        # 生成q,k,v
        q = self.q(x).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if pre_k is not None and pre_v is not None:
            k, v = pre_k, pre_v
        else:
            assert mem is not None, "mem must be provided when precomputed K/V are not given"
            kv = self.kv(mem).reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

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


class GatedFusion(nn.Module):
    """门控融合模块，用于融合自注意力输出和交叉注意力输出
    
    Args:
        dim (int): 输入特征维度
    """
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(2 * dim, dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, self_attn, cross_attn):
        # 拼接两个输入: [B, N, 2*C]
        fused = torch.cat([self_attn, cross_attn], dim=-1)
        # 计算门控因子: [B, N, C]
        gate = self.sigmoid(self.fc(fused))
        # 融合
        output = self_attn + gate * cross_attn
        return output


class RDG(nn.Module):
    def __init__(
        self,
        dim,
        mem_dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        shift_size,
        mlp_ratio,
        qkv_bias,
        qk_scale,
        drop,
        attn_drop,
        drop_path,
        norm_layer,
        gc,
        patch_size,
        img_size,
        use_gating=False,
    ):
        super(RDG, self).__init__()

        self.swin1 = SwinTransformerBlock(
            dim=dim,
            mem_dim=mem_dim,
            input_resolution=input_resolution,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=0,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
            norm_layer=norm_layer,
            use_gating=use_gating,
        )
        self.adjust1 = nn.Conv2d(dim, gc, 1)

        self.swin2 = SwinTransformerBlock(
            dim + gc,
            mem_dim=mem_dim,
            input_resolution=input_resolution,
            num_heads=num_heads - ((dim + gc) % num_heads),
            window_size=window_size,
            shift_size=window_size // 2,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
            norm_layer=norm_layer,
            use_gating=use_gating,
        )
        self.adjust2 = nn.Conv2d(dim + gc, gc, 1)

        self.swin3 = SwinTransformerBlock(
            dim + 2 * gc,
            mem_dim=mem_dim,
            input_resolution=input_resolution,
            num_heads=num_heads - ((dim + 2 * gc) % num_heads),
            window_size=window_size,
            shift_size=0,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
            norm_layer=norm_layer,
            use_gating=use_gating,
        )
        self.adjust3 = nn.Conv2d(dim + gc * 2, gc, 1)

        self.swin4 = SwinTransformerBlock(
            dim + 3 * gc,
            mem_dim=mem_dim,
            input_resolution=input_resolution,
            num_heads=num_heads - ((dim + 3 * gc) % num_heads),
            window_size=window_size,
            shift_size=window_size // 2,
            mlp_ratio=1,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
            norm_layer=norm_layer,
            use_gating=use_gating,
        )
        self.adjust4 = nn.Conv2d(dim + gc * 3, gc, 1)

        self.swin5 = SwinTransformerBlock(
            dim + 4 * gc,
            mem_dim=mem_dim,
            input_resolution=input_resolution,
            num_heads=num_heads - ((dim + 4 * gc) % num_heads),
            window_size=window_size,
            shift_size=0,
            mlp_ratio=1,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
            norm_layer=norm_layer,
            use_gating=use_gating,
        )
        self.adjust5 = nn.Conv2d(dim + gc * 4, dim, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.pe = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=0,
            embed_dim=dim,
            norm_layer=None,
        )

        self.pue = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=0,
            embed_dim=dim,
            norm_layer=None,
        )

    def forward(self, x, mem, xsize):
        # 将mem分成5份,对应5个swin block
        if mem is not None:
            B, L, C = mem.shape
            mem_splits = torch.chunk(mem, 5, dim=-1)
        else:
            mem_splits = [None] * 5

        x1 = self.pe(self.lrelu(self.adjust1(self.pue(self.swin1(x, mem_splits[0], xsize), xsize))))
        x2 = self.pe(
            self.lrelu(
                self.adjust2(self.pue(self.swin2(torch.cat((x, x1), -1), mem_splits[1], xsize), xsize))
            )
        )
        x3 = self.pe(
            self.lrelu(
                self.adjust3(
                    self.pue(self.swin3(torch.cat((x, x1, x2), -1), mem_splits[2], xsize), xsize)
                )
            )
        )
        x4 = self.pe(
            self.lrelu(
                self.adjust4(
                    self.pue(self.swin4(torch.cat((x, x1, x2, x3), -1), mem_splits[3], xsize), xsize)
                )
            )
        )
        x5 = self.pe(
            self.adjust5(
                self.pue(self.swin5(torch.cat((x, x1, x2, x3, x4), -1), mem_splits[4], xsize), xsize)
            )
        )

        return x5 * 0.2 + x

    @torch.no_grad()
    def prepare_static_kv(self, mem, x_size):
        """为本RDG内各个block预计算K/V缓存。

        参数:
            mem: (B, L, C_mem_total_for_this_RDG) 对应本RDG的mem切片
            x_size: (H, W)
        """
        if mem is None:
            return
        B, L, C_total = mem.shape
        mem_splits = torch.chunk(mem, 5, dim=-1)
        self.swin1.prepare_static_kv(mem_splits[0], x_size)
        self.swin2.prepare_static_kv(mem_splits[1], x_size)
        self.swin3.prepare_static_kv(mem_splits[2], x_size)
        self.swin4.prepare_static_kv(mem_splits[3], x_size)
        self.swin5.prepare_static_kv(mem_splits[4], x_size)


class SwinTransformerBlock(nn.Module):
    r"""Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        mem_dim (int): Number of memory channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self,
        dim,
        mem_dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        use_gating=False,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert (
            0 <= self.shift_size < self.window_size
        ), "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

        # 只在mem_dim > 0时初始化交叉注意力相关模块
        if mem_dim > 0:
            self.cross_attn = CrossWindowAttention(
                dim=dim,
                mem_dim=mem_dim,
                window_size=to_2tuple(self.window_size),
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
            self.norm_cross = norm_layer(mem_dim)
            # 静态缓存：用于固定的ref在推理阶段复用K/V
            self._static_k = None
            self._static_v = None
            self._static_x_size = None  # (H, W)
            self._static_nW = None
        # 只在mem_dim > 0且use_gating=True时初始化门控融合模块
        if mem_dim > 0 and use_gating:
            self.gated_fusion = GatedFusion(dim)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(
            img_mask, self.window_size
        )  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
            attn_mask == 0, float(0.0)
        )

        return attn_mask

    def forward(self, x, mem, x_size):
        H, W = x_size
        B, L, C = x.shape
        if mem is not None:
            _, _, C_mem = mem.shape
        
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # 只在有mem或有静态缓存时进行cross attention
        cross_enabled = hasattr(self, 'cross_attn') and (mem is not None or self._static_k is not None)
        if cross_enabled and mem is not None:
            mem = self.norm_cross(mem)
            mem = mem.contiguous().view(B, H, W, C_mem)
        
        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            if hasattr(self, 'cross_attn') and mem is not None:
                shifted_mem = torch.roll(mem, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            if hasattr(self, 'cross_attn') and mem is not None:
                shifted_mem = mem

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        
        if cross_enabled and mem is not None:
            mem_windows = window_partition(shifted_mem, self.window_size)
            mem_windows = mem_windows.view(-1, self.window_size * self.window_size, C_mem)

        # W-MSA/SW-MSA
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)
            if cross_enabled:
                if self._static_k is not None and self._static_x_size == x_size:
                    # 复制缓存到当前batch
                    k = self._static_k.repeat(B, 1, 1, 1)
                    v = self._static_v.repeat(B, 1, 1, 1)
                    cross_attn_windows = self.cross_attn(x_windows, mem=None, mask=self.attn_mask, pre_k=k, pre_v=v)
                else:
                    assert mem is not None, "No static K/V prepared and mem is None"
                    cross_attn_windows = self.cross_attn(x_windows, mem_windows, mask=self.attn_mask)
                
                # 使用门控融合模块
                if hasattr(self, 'gated_fusion'):
                    attn_windows = self.gated_fusion(attn_windows, cross_attn_windows)
                else:
                    attn_windows = attn_windows + cross_attn_windows
        else:
            attn_mask = self.calculate_mask(x_size).to(x.device)
            attn_windows = self.attn(x_windows, mask=attn_mask)
            if cross_enabled:
                if self._static_k is not None and self._static_x_size == x_size:
                    k = self._static_k.repeat(B, 1, 1, 1)
                    v = self._static_v.repeat(B, 1, 1, 1)
                    cross_attn_windows = self.cross_attn(x_windows, mem=None, mask=attn_mask, pre_k=k, pre_v=v)
                else:
                    assert mem is not None, "No static K/V prepared and mem is None"
                    cross_attn_windows = self.cross_attn(x_windows, mem_windows, mask=attn_mask)
                
                # 使用门控融合模块
                if hasattr(self, 'gated_fusion'):
                    attn_windows = self.gated_fusion(attn_windows, cross_attn_windows)
                else:
                    attn_windows = attn_windows + cross_attn_windows

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

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

    @torch.no_grad()
    def prepare_static_kv(self, mem, x_size):
        """预计算并缓存当前block在给定 x_size 下的 K/V。

        参数:
            mem: (B, L, C_mem) 的记忆特征（已经过 PatchEmbed，且对应本block的mem切片）
            x_size: (H, W)
        说明:
            - 仅在存在 cross_attn 时有效；
            - 建议 B=1 进行缓存，运行时会按 batch 复制；
            - 当窗口大小/shift或 x_size 变化时需重新 prepare。
        """
        if not hasattr(self, 'cross_attn'):
            return
        H, W = x_size
        B, L, C_mem = mem.shape
        device = mem.device
        mem = self.norm_cross(mem)
        mem = mem.contiguous().view(B, H, W, C_mem)
        # cyclic shift (与forward保持一致)
        if self.shift_size > 0:
            shifted_mem = torch.roll(mem, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_mem = mem
        # partition windows
        mem_windows = window_partition(shifted_mem, self.window_size)
        N = self.window_size * self.window_size
        mem_windows = mem_windows.view(-1, N, C_mem)
        # 线性映射得到 K/V
        C = self.dim
        kv = self.cross_attn.kv(mem_windows).reshape(-1, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        self._static_k = kv[0].contiguous().to(device)
        self._static_v = kv[1].contiguous().to(device)
        self._static_x_size = x_size
        self._static_nW = int(H * W / (self.window_size * self.window_size))

    def clear_static_kv(self):
        self._static_k = None
        self._static_v = None
        self._static_x_size = None
        self._static_nW = None

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, "
            f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"
        )

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r"""Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class PatchEmbed(nn.Module):
    r"""Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # 结构为 [B, num_patches, C]
        if self.norm is not None:
            x = self.norm(x)  # 归一化
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = (
            Ho
            * Wo
            * self.embed_dim
            * self.in_chans
            * (self.patch_size[0] * self.patch_size[1])
        )
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r"""Image to Patch Unembedding

    输入:
        img_size (int): 图像的大小，默认为 224*224.
        patch_size (int): Patch token 的大小，默认为 4*4.
        in_chans (int): 输入图像的通道数，默认为 3.
        embed_dim (int): 线性 projection 输出的通道数，默认为 96.
        norm_layer (nn.Module, optional): 归一化层， 默认为N None.
    """

    def __init__(
        self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None
    ):
        super().__init__()
        img_size = to_2tuple(img_size)  # 图像的大小，默认为 224*224
        patch_size = to_2tuple(patch_size)  # Patch token 的大小，默认为 4*4
        patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        ]  # patch 的分辨率
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = (
            patches_resolution[0] * patches_resolution[1]
        )  # patch 的个数，num_patches

        self.in_chans = in_chans  # 输入图像的通道数
        self.embed_dim = embed_dim  # 线性 projection 输出的通道数

    def forward(self, x, x_size):
        B, HW, C = x.shape  # 输入 x 的结构
        x = x.transpose(1, 2).view(
            B, -1, x_size[0], x_size[1]
        )  # 输出结构为 [B, Ph*Pw, C]
        return x


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(
                f"scale {scale} is not supported. " "Supported scales: 2^n and 3."
            )
        super(Upsample, self).__init__(*m)


class DRCT(nn.Module):
    def __init__(
        self,
        img_size=64,
        patch_size=1,
        in_chans=3,
        mem_chans=16,
        embed_dim=96,
        depths=(6, 6, 6, 6),
        num_heads=(6, 6, 6, 6),
        window_size=7,
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        overlap_ratio=0.5,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        upscale=2,
        img_range=1.0,
        upsampler="",
        resi_connection="1conv",
        gc=32,
        num_out_ch=2,
        cross_mode=True,
        use_gating=False,
        # memory regularization
        mem_dropout2d: float = 0.0,
        **kwargs,
    ):
        super(DRCT, self).__init__()

        self.window_size = window_size
        self.shift_size = window_size // 2
        self.overlap_ratio = overlap_ratio
        self.cross_mode = cross_mode
        self.use_gating = use_gating
        # memory dropout2d probability (applied on raw mem feature maps)
        self.mem_dropout2d_p = float(mem_dropout2d)
        self.mem_dropout2d = nn.Dropout2d(p=self.mem_dropout2d_p) if self.mem_dropout2d_p > 0 else nn.Identity()

        num_in_ch = in_chans
        num_out_ch = num_out_ch
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler

        # ------------------------- 1, shallow feature extraction ------------------------- #
        if cross_mode:
            conv_in_chans = num_in_ch
        else:
            conv_in_chans = num_in_ch + mem_chans
            
        self.conv_first = nn.Conv2d(conv_in_chans, embed_dim, 3, 1, 1)

        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # 计算每个RDG中需要的mem通道数
        blocks_per_rdg = 5  # 每个RDG中有5个swin block
        total_mem_channels = mem_chans * blocks_per_rdg * len(depths)

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        
        # add memory patch embedding with correct mem_chans
        if mem_chans > 0:
            self.mem_patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=total_mem_channels,  # 使用计算得到的总mem通道数
                embed_dim=total_mem_channels,  # embed_dim也使用相同的通道数
                norm_layer=norm_layer if self.patch_norm else None,
            )

        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim)
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RDG(
                dim=embed_dim,
                mem_dim=mem_chans,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=0,
                num_heads=num_heads[i_layer],
                window_size=window_size,
                shift_size=window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                gc=gc,
                img_size=img_size,
                patch_size=patch_size,
                use_gating=self.use_gating,
            )

            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        if resi_connection == "1conv":
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == "identity":
            self.conv_after_body = nn.Identity()

        # ------------------------- 3, high quality image reconstruction ------------------------- #
        if self.upsampler == "pixelshuffle":
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True)
            )
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def forward_features(self, x, mem):
        x_size = (x.shape[2], x.shape[3])
        
        x = self.patch_embed(x)
        if mem is not None:
            # Apply spatial/channel-wise dropout to memory maps before embedding
            mem = self.mem_dropout2d(mem)
            mem = self.mem_patch_embed(mem)
            # 将mem分成与layers数量相同的份数
            B, L, C = mem.shape
            mem_splits = torch.chunk(mem, len(self.layers), dim=-1)
        else:
            mem_splits = [None] * len(self.layers)
        
        if self.ape:
            x = x + self.absolute_pos_embed
            
        x = self.pos_drop(x)
        if mem is not None:
            mem = self.pos_drop(mem)

        # 每个layer使用不同的mem部分
        for i, layer in enumerate(self.layers):
            x = layer(x, mem_splits[i], x_size)

        x = self.norm(x)
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x, mem=None):
        self.mean = self.mean.type_as(x)
        
        if self.upsampler == "pixelshuffle":
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x, mem)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))

        return x

    @torch.no_grad()
    def prepare_static_kv(self, ref, x_size):
        """预计算所有层的静态K/V，以在推理阶段复用，降低在线FLOPs。

        参数:
            ref: (B, C_ref_total, H, W) 的记忆图（cross_mode=True时应为 total_mem_channels）
            x_size: (H, W) 输入分辨率
        """
        if not hasattr(self, 'mem_patch_embed'):
            return
        mem = self.mem_patch_embed(ref)
        B, L, C = mem.shape
        mem_splits = torch.chunk(mem, len(self.layers), dim=-1)
        for i, layer in enumerate(self.layers):
            layer.prepare_static_kv(mem_splits[i], x_size)

    def clear_static_kv(self):
        for layer in self.layers:
            if hasattr(layer, 'swin1'):
                layer.swin1.clear_static_kv()
                layer.swin2.clear_static_kv()
                layer.swin3.clear_static_kv()
                layer.swin4.clear_static_kv()
                layer.swin5.clear_static_kv()
