import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmdet.registry import MODELS
from mmengine.runner import load_checkpoint
from mmengine.logging import MMLogger


class EvaAttention(BaseModule):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        qkv_fused=True,
        attn_drop=0.,
        proj_drop=0.,
        attn_head_dim=None,
        norm_layer=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias) if qkv_fused else None
        if not qkv_fused:
            self.q_proj = nn.Linear(dim, all_head_dim, bias=qkv_bias)
            self.k_proj = nn.Linear(dim, all_head_dim, bias=False)
            self.v_proj = nn.Linear(dim, all_head_dim, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm = norm_layer(all_head_dim) if norm_layer is not None else nn.Identity()

    def forward(self, x, rope=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if rope is not None:
            q_t = q.permute(2, 0, 1, 3)
            k_t = k.permute(2, 0, 1, 3)
            cos, sin = rope(q_t, seq_len=N)
            q, k = apply_rotary_pos_emb(q_t, k_t, cos, sin)
            q = q.permute(1, 2, 0, 3)
            k = k.permute(1, 2, 0, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class EvaBlock(BaseModule):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=False,
        qkv_fused=True,
        drop=0.,
        attn_drop=0.,
        init_values=None,
        drop_path=0.,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_head_dim=None,
        use_post_norm=False,
    ):
        super().__init__()
        self.use_post_norm = use_post_norm
        self.norm1 = norm_layer(dim)
        self.attn = EvaAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qkv_fused=qkv_fused,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values is not None:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(dim))
            self.gamma_2 = nn.Parameter(init_values * torch.ones(dim))
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, rope=None):
        if self.use_post_norm:
            x = x + self.drop_path(self.attn(self.norm1(x), rope=rope))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            if self.gamma_1 is None:
                x = x + self.drop_path(self.attn(self.norm1(x), rope=rope))
                x = x + self.drop_path(self.mlp(self.norm2(x)))
            else:
                x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), rope=rope))
                x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class Mlp(BaseModule):
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

@MODELS.register_module()
class EVA(BaseModule):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        qkv_bias=True,
        qkv_fused=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        norm_layer=nn.LayerNorm,
        init_values=None,
        use_abs_pos_emb=True,
        use_rot_pos_emb=False,
        use_post_norm=False,
        rope_args=None,
        out_indices=[3, 5, 7, 11],
        init_cfg=None
    ):
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.use_post_norm = use_post_norm
        
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_rot_pos_emb:
            if rope_args:
                self.rope = Rope(dim=embed_dim // num_heads, **rope_args)
            else:
                self.rope = Rope(dim=embed_dim // num_heads)
        else:
            self.rope = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            EvaBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qkv_fused=qkv_fused,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, use_post_norm=use_post_norm)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)
        self.out_indices = out_indices

        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self):
        logger = MMLogger.get_current_instance()
        if self.init_cfg is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            if self.pos_embed is not None:
                nn.init.trunc_normal_(self.pos_embed, std=.02)
            nn.init.trunc_normal_(self.cls_token, std=.02)
            self.apply(self._init_weights)
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            ckpt = load_checkpoint(self, self.init_cfg.checkpoint, map_location='cpu', logger=logger, strict=False)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        outs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x, rope=self.rope)
            if i in self.out_indices:
                out = self.norm(x) if self.use_post_norm else x
                B, N, C = out.shape
                H = W = int((N - 1) ** 0.5)  # cls 토큰 제외
                out = out[:, 1:, :].reshape(B, H, W, C).permute(0, 3, 1, 2)
                outs.append(out)
        return tuple(outs)

    def forward(self, x):
        if isinstance(x, list):
            x = torch.stack(x)
        elif not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected input to be torch.Tensor or list of tensors, but got {type(x)}")
        
        x = self.forward_features(x)
        return x

class PatchEmbed(BaseModule):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class RotaryEmbedding(BaseModule):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, max_len):
        t = torch.arange(max_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb[None, None, :, :]
    
class Rope(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = max_position_embeddings

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]
        
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)
        
        return (
            self.cos_cached[:, :, :seq_len, ...].expand(x.shape[0], -1, -1, -1).to(x.device),
            self.sin_cached[:, :, :seq_len, ...].expand(x.shape[0], -1, -1, -1).to(x.device)
        )

    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).to(self.inv_freq.device)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

def apply_rotary_pos_emb(q, k, cos, sin):
    # Ensure that cos and sin have the correct shape
    cos = cos[:, :, :q.shape[2], :]
    sin = sin[:, :, :q.shape[2], :]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)