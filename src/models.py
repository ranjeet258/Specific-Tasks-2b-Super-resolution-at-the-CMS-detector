"""
src/models.py
=============
Three generator architectures for CMS Calorimeter Super-Resolution.

┌─────────────────────────────────────────────────────────────────┐
│  Model 1: CMSSRGANGenerator  — Modified SRGAN (strong baseline) │
│  Model 2: CMSTransformerSR   — Swin-Transformer generator       │
│  Model 3: CMSDiffusionSR     — Conditional diffusion U-Net      │
│  Discriminator: CMSPatchGAN  — Spectral-norm PatchGAN           │
└─────────────────────────────────────────────────────────────────┘

Upsampling strategy (ALL three models — fixes artifact issue):
  64×64 ──[PixelShuffle ×2]──► 128×128 ──[AdaptiveAvgPool2d(125)]──► 125×125

  Why PixelShuffle over transposed conv:
    - Rearranges learned sub-pixel features → sharper, less blurry edges
    - No checkerboard artifacts (common with stride-2 transposed conv)

  Why AdaptiveAvgPool2d(125) over F.interpolate to 125:
    - Produces EXACT 125×125 — no floating-point rounding
    - 128→125 removes 3 border pixels via local averaging, zero ringing
    - Deterministic output size regardless of input padding settings

Key architectural features:
- Residual Dense Blocks + SE attention in SRGAN generator for better detail recovery
- Shifted-window attention in Transformer generator for long-range spatial correlations
- Diffusion model conditioned on upsampled LR image for stable training on sparse data
- Spectral normalization in PatchGAN discriminator for stable adversarial training
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED BUILDING BLOCKS
# ═══════════════════════════════════════════════════════════════════════════════

class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid), nn.ReLU(inplace=True),
            nn.Linear(mid, channels), nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.se(x).view(x.size(0), x.size(1), 1, 1)


class ResidualDenseBlock(nn.Module):
    """
    RRDB-lite: dense connections within a residual block + SE attention.
    Denser feature reuse than plain ResBlock → better detail recovery on
    sparse calorimeter data.
    """
    def __init__(self, ch: int, growth: int = 32):
        super().__init__()
        self.c1 = nn.Conv2d(ch,          growth,  3, 1, 1)
        self.c2 = nn.Conv2d(ch+growth,   growth,  3, 1, 1)
        self.c3 = nn.Conv2d(ch+growth*2, growth,  3, 1, 1)
        self.c4 = nn.Conv2d(ch+growth*3, ch,      3, 1, 1)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.se  = SEBlock(ch)
        self.beta = 0.2

    def forward(self, x):
        x1 = self.act(self.c1(x))
        x2 = self.act(self.c2(torch.cat([x, x1], 1)))
        x3 = self.act(self.c3(torch.cat([x, x1, x2], 1)))
        x4 = self.c4(torch.cat([x, x1, x2, x3], 1))
        return self.se(x4) * self.beta + x


def _exact_upsample_64_to_125(in_ch: int) -> nn.Sequential:
    """
    Shared upsampling module: 64×64 → EXACTLY 125×125.

        Conv2d(C → C×4)  +  PixelShuffle(2)  →  (C, 128, 128)
        AdaptiveAvgPool2d(125)                 →  (C, 125, 125)  ← EXACT
        Conv2d refinement                      →  (C, 125, 125)
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch * 4, kernel_size=3, padding=1),
        nn.PixelShuffle(2),               # (in_ch, 128, 128)
        nn.PReLU(),
        nn.AdaptiveAvgPool2d(125),        # (in_ch, 125, 125)  EXACT OUTPUT SIZE
        nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
        nn.PReLU(),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 1 — SRGAN GENERATOR  (baseline)
# ═══════════════════════════════════════════════════════════════════════════════

class CMSSRGANGenerator(nn.Module):
    """
    Modified SRGAN with Residual Dense Blocks + SE attention.
    Input  (B,3,64,64) → Output (B,3,125,125)
    """
    def __init__(self, in_ch=3, out_ch=3, features=64, n_blocks=12):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, features, 9, padding=4), nn.PReLU()
        )
        self.body = nn.Sequential(
            *[ResidualDenseBlock(features) for _ in range(n_blocks)],
            SEBlock(features),
        )
        self.post = nn.Sequential(
            nn.Conv2d(features, features, 3, padding=1, bias=False),
            nn.BatchNorm2d(features),
        )
        self.up   = _exact_upsample_64_to_125(features)
        self.tail = nn.Sequential(
            nn.Conv2d(features, features // 2, 3, padding=1), nn.PReLU(),
            nn.Conv2d(features // 2, out_ch, 9, padding=4),
            nn.ReLU(),       # ✅ no energy suppression
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.head(x)
        r = self.post(self.body(h)) + h
        out = self.tail(self.up(r))
        # ✅ KEY FIX: add bicubic-upsampled LR as residual
        # Model now learns enhancement on top of LR, not full reconstruction
        # This prevents the all-zeros collapse on sparse calorimeter data
        lr_up = F.interpolate(x, size=(125, 125), mode='bicubic', align_corners=False)
        return F.relu(out + lr_up)        # (B,3,125,125) always non-negative


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 2 — TRANSFORMER GENERATOR  (modern technique — required by project brief)
# ═══════════════════════════════════════════════════════════════════════════════

class WindowAttention(nn.Module):
    """Shifted-Window Multi-Head Self-Attention (SwinIR, Liang et al. 2021)."""
    def __init__(self, dim: int, win: int = 8, n_heads: int = 8):
        super().__init__()
        self.win = win; self.n_heads = n_heads
        self.scale = (dim // n_heads) ** -0.5
        self.qkv   = nn.Linear(dim, dim * 3)
        self.proj  = nn.Linear(dim, dim)
        self.bias  = nn.Parameter(torch.zeros((2*win-1)**2, n_heads))
        nn.init.trunc_normal_(self.bias, std=0.02)
        # build relative position index
        c = torch.arange(win)
        g = torch.stack(torch.meshgrid(c, c, indexing='ij')).flatten(1)
        r = (g[:, :, None] - g[:, None, :]).permute(1, 2, 0).contiguous()
        r += win - 1; r[:,:,0] *= 2*win - 1
        self.register_buffer('rel_idx', r.sum(-1))

    def forward(self, x):
        Bw, N, C = x.shape
        qkv = self.qkv(x).reshape(Bw, N, 3, self.n_heads, C//self.n_heads).permute(2,0,3,1,4)
        q, k, v = qkv.unbind(0)
        attn = q @ k.transpose(-2, -1) * self.scale
        bias = self.bias[self.rel_idx.view(-1)].view(N, N, -1).permute(2,0,1)
        attn = F.softmax(attn + bias.unsqueeze(0), dim=-1)
        return self.proj((attn @ v).transpose(1,2).reshape(Bw, N, C))


class SwinBlock(nn.Module):
    """One Swin-Transformer block with optional cyclic shift."""
    def __init__(self, dim: int, win: int = 8, n_heads: int = 8, shift: bool = False):
        super().__init__()
        self.win = win; self.shift = shift; self.s = win // 2
        self.norm1 = nn.LayerNorm(dim); self.norm2 = nn.LayerNorm(dim)
        self.attn  = WindowAttention(dim, win, n_heads)
        mlp_dim    = dim * 4
        self.ff    = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, dim)
        )

    def _partition(self, x, W):
        B, H, Ww, C = x.shape
        x = x.view(B, H//W, W, Ww//W, W, C)
        return x.permute(0,1,3,2,4,5).contiguous().view(-1, W*W, C)

    def _reverse(self, x, W, H, Ww):
        B = int(x.shape[0] / ((H//W)*(Ww//W)))
        return x.view(B, H//W, Ww//W, W, W, -1).permute(0,1,3,2,4,5).contiguous().view(B,H,Ww,-1)

    def forward(self, x):
        B, C, H, W = x.shape
        xt = x.permute(0,2,3,1)
        if self.shift: xt = torch.roll(xt, (-self.s, -self.s), (1, 2))
        wins = self._partition(xt, self.win)
        xt   = self._reverse(self.attn(self.norm1(wins)) + wins, self.win, H, W)
        if self.shift: xt = torch.roll(xt, (self.s, self.s), (1, 2))
        xt = xt + self.ff(self.norm2(xt))
        return xt.permute(0,3,1,2)


class CMSTransformerSR(nn.Module):
    """
    Swin-Transformer SR generator.
    Long-range attention captures full-jet spatial correlations — not possible
    with convolutions alone. Alternating W-MSA and SW-MSA blocks.

    Input  (B,3,64,64) → Output (B,3,125,125)
    """
    def __init__(self, in_ch=3, out_ch=3, features=64, n_stages=6, win=8, n_heads=8):
        super().__init__()
        self.shallow = nn.Conv2d(in_ch, features, 3, 1, 1)
        self.deep    = nn.Sequential(
            *[SwinBlock(features, win, n_heads, shift=(i % 2 == 1))
              for i in range(n_stages)]
        )
        self.aggreg  = nn.Conv2d(features, features, 3, 1, 1)
        self.up      = _exact_upsample_64_to_125(features)
        self.recon   = nn.Sequential(
            nn.Conv2d(features, features, 3, 1, 1), nn.PReLU(),
            nn.Conv2d(features, out_ch, 3, 1, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        f = self.shallow(x)
        r = self.aggreg(self.deep(f)) + f   # residual skip
        return self.recon(self.up(r))        # (B,3,125,125) EXACT


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 3 — CONDITIONAL DIFFUSION SR  (DiffLense-inspired)
# ═══════════════════════════════════════════════════════════════════════════════

class SinusoidalEmb(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        freq = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half-1))
        args = t[:,None].float() * freq[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class DiffResBlock(nn.Module):
    """Residual block conditioned on timestep via AdaGroupNorm scale+shift."""
    def __init__(self, in_ch, out_ch, t_dim):
        super().__init__()
        self.n1 = nn.GroupNorm(min(8, in_ch), in_ch)
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.n2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.tp = nn.Linear(t_dim, out_ch * 2)
        self.rs = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.c1(F.silu(self.n1(x)))
        sc, sh = self.tp(F.silu(t_emb)).chunk(2, dim=-1)
        h = F.silu(self.n2(h) * (1 + sc[:,:,None,None]) + sh[:,:,None,None])
        return self.c2(h) + self.rs(x)


class CMSDiffusionDenoiser(nn.Module):
    """
    Conditional denoising U-Net for CMS SR.
    Conditioned on: timestep t and pre-processed upsampled LR image.
    Architecture: U-Net with attention bottleneck (DiffLense style).
    """
    def __init__(self, in_ch=3, base_ch=64, t_dim=256):
        super().__init__()
        C = base_ch
        self.t_emb = nn.Sequential(
            SinusoidalEmb(C), nn.Linear(C, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim)
        )
        self.enc_in = nn.Conv2d(in_ch*2, C, 3, padding=1)
        self.e1     = DiffResBlock(C,   C*2, t_dim)
        self.d1     = nn.Conv2d(C*2, C*2, 4, 2, 1)
        self.e2     = DiffResBlock(C*2, C*4, t_dim)
        self.d2     = nn.Conv2d(C*4, C*4, 4, 2, 1)
        self.m1     = DiffResBlock(C*4, C*4, t_dim)
        self.m_att  = nn.MultiheadAttention(C*4, num_heads=8, batch_first=True)
        self.m2     = DiffResBlock(C*4, C*4, t_dim)
        self.u2     = nn.ConvTranspose2d(C*4, C*4, 4, 2, 1)
        self.r2     = DiffResBlock(C*8, C*2, t_dim)
        self.u1     = nn.ConvTranspose2d(C*2, C*2, 4, 2, 1)
        self.r1     = DiffResBlock(C*4, C,   t_dim)
        self.out    = nn.Sequential(
            nn.GroupNorm(min(8,C), C), nn.SiLU(),
            nn.Conv2d(C, in_ch, 3, padding=1),
        )

    def forward(self, x_t, t, cond):
        te = self.t_emb(t)
        x  = self.enc_in(torch.cat([x_t, cond], 1))
        s1 = self.e1(x,  te)
        x  = self.d1(s1)
        s2 = self.e2(x,  te)
        x  = self.d2(s2)
        x  = self.m1(x,  te)
        B, C, H, W = x.shape
        xf = x.flatten(2).transpose(1,2)
        xa, _ = self.m_att(xf, xf, xf)
        x  = xa.transpose(1,2).view(B, C, H, W) + x
        x  = self.m2(x,  te)
        x  = F.interpolate(self.u2(x), size=s2.shape[-2:], mode='nearest')
        x  = self.r2(torch.cat([x, s2], 1), te)
        x  = F.interpolate(self.u1(x), size=s1.shape[-2:], mode='nearest')
        x  = self.r1(torch.cat([x, s1], 1), te)
        x  = F.interpolate(x, size=(125, 125), mode='nearest')  # guarantee exact size
        return self.out(x)


class CMSDiffusionSR(nn.Module):
    """
    Full diffusion SR model.
    Training  : forward(lr, hr)   → noise-prediction L1 loss
    Inference : sample(lr, steps) → SR image (B,3,125,125)

    Condition preprocessing (DiffLense pipeline):
        LR (64×64) → PixelShuffle+Pool → 125×125 → concat with x_t
    """
    def __init__(self, in_ch=3, base_ch=64, n_steps=1000):
        super().__init__()
        self.n_steps  = n_steps
        self.denoiser = CMSDiffusionDenoiser(in_ch, base_ch)
        self.cond_up  = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1), nn.SiLU(),
            _exact_upsample_64_to_125(base_ch),
            nn.Conv2d(base_ch, in_ch, 3, padding=1),
        )
        betas = self._cosine_schedule(n_steps)
        ab    = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer('betas',     betas)
        self.register_buffer('sqrt_ab',   ab.sqrt())
        self.register_buffer('sqrt_1mab', (1 - ab).sqrt())
        self.register_buffer('alpha_bar', ab)

    @staticmethod
    def _cosine_schedule(T):
        s = 0.008; steps = torch.arange(T+1)
        f = torch.cos(((steps/T + s)/(1+s)) * math.pi/2) ** 2
        return (1 - f[1:]/f[:-1]).clamp(1e-4, 0.9999)

    def forward(self, lr, hr):
        B = lr.size(0)
        t = torch.randint(0, self.n_steps, (B,), device=lr.device)
        noise = torch.randn_like(hr)
        x_t   = self.sqrt_ab[t].view(B,1,1,1)*hr + self.sqrt_1mab[t].view(B,1,1,1)*noise
        cond  = self.cond_up(lr)
        return F.l1_loss(self.denoiser(x_t, t, cond), noise)

    @torch.no_grad()
    def sample(self, lr, n_steps=None):
        B, C, device = lr.size(0), lr.size(1), lr.device
        T    = n_steps or self.n_steps
        cond = self.cond_up(lr)
        x    = torch.randn(B, C, 125, 125, device=device)
        for i in reversed(range(T)):
            t      = torch.full((B,), i, device=device, dtype=torch.long)
            eps    = self.denoiser(x, t, cond)
            ab     = self.alpha_bar[i]
            ab_m1  = self.alpha_bar[i-1] if i > 0 else torch.tensor(1.0, device=device)
            beta   = self.betas[i]
            x0     = ((x - (1-ab).sqrt()*eps) / ab.sqrt()).clamp(-1, 1)
            mean   = (ab_m1.sqrt()*beta/(1-ab))*x0 + ((1-ab_m1)*(1-beta)/(1-ab)).sqrt()*x
            x      = mean + (beta.sqrt()*torch.randn_like(x) if i > 0 else 0)
        return x.clamp(-1, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# DISCRIMINATOR — spectral-norm PatchGAN
# ═══════════════════════════════════════════════════════════════════════════════

class CMSPatchGAN(nn.Module):
    """
    PatchGAN with spectral normalisation (Miyato et al. 2018).
    Spectral norm constrains Lipschitz constant → stable training on sparse data.
    Input (B,3,125,125) → patch-level real/fake logits.
    """
    def __init__(self, in_ch=3, base_ch=64, n_layers=4):
        super().__init__()
        sn = nn.utils.spectral_norm
        nf = base_ch
        layers = [sn(nn.Conv2d(in_ch, nf, 4, 2, 1)), nn.LeakyReLU(0.2, True)]
        for i in range(1, n_layers):
            nf_prev, nf = nf, min(nf*2, 512)
            s = 2 if i < n_layers-1 else 1
            layers += [sn(nn.Conv2d(nf_prev, nf, 4, s, 1)),
                       nn.BatchNorm2d(nf), nn.LeakyReLU(0.2, True)]
        layers.append(sn(nn.Conv2d(nf, 1, 4, 1, 1)))
        self.model = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, x): return self.model(x)


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

def build_generator(cfg: dict) -> nn.Module:
    mtype = cfg.get("model", {}).get("type", "srgan").lower()
    g     = cfg.get("model", {}).get("generator", {})
    if mtype == "srgan":
        return CMSSRGANGenerator(
            in_ch=g.get("in_channels",3), out_ch=g.get("out_channels",3),
            features=g.get("base_features",64), n_blocks=g.get("num_res_blocks",12))
    elif mtype == "transformer":
        return CMSTransformerSR(
            in_ch=g.get("in_channels",3), out_ch=g.get("out_channels",3),
            features=g.get("base_features",64), n_stages=g.get("n_stages",6))
    elif mtype == "diffusion":
        return CMSDiffusionSR(
            in_ch=g.get("in_channels",3), base_ch=g.get("base_features",64),
            n_steps=g.get("n_diffusion_steps",1000))
    else:
        raise ValueError(f"Unknown model type '{mtype}'. Choose: srgan | transformer | diffusion")


def build_discriminator(cfg: dict) -> CMSPatchGAN:
    d = cfg.get("model", {}).get("discriminator", {})
    return CMSPatchGAN(
        in_ch=d.get("in_channels",3), base_ch=d.get("base_features",64),
        n_layers=d.get("num_layers",4))


def count_parameters(model: nn.Module) -> str:
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"{n/1e6:.2f}M" if n >= 1_000_000 else f"{n/1e3:.1f}K"
