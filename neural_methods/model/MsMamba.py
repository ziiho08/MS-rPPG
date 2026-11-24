import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import trunc_normal_, lecun_normal_, DropPath
from mamba_ssm.modules.mamba_simple import Mamba

def _init_weights(module, n_layer, initializer_range=0.02,
                  rescale_prenorm_residual=True, n_residuals_per_layer=1):
    if isinstance(module, nn.Linear):
        if module.bias is not None and not getattr(module.bias, "_no_reinit", False):
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)
    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)

def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None: nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        lecun_normal_(m.weight)
        if m.bias is not None: nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.GroupNorm)):
        nn.init.zeros_(m.bias); nn.init.ones_(m.weight)

class StandardFFN(nn.Module):
    """Plain MLP FFN — ablation baseline for Frequencydomain_FFN."""
    def __init__(self, dim, mlp_ratio):
        super().__init__()
        hidden = dim * mlp_ratio
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = nn.GELU()
    def forward(self, x):   # [B, N, C]
        return self.fc2(self.act(self.fc1(x)))

class Frequencydomain_FFN(nn.Module):
    def __init__(self, dim, mlp_ratio):
        super().__init__()
        self.hidden = dim * mlp_ratio
        self.scale = 0.02

        self.r  = nn.Parameter(self.scale * torch.randn(self.hidden, self.hidden))
        self.i  = nn.Parameter(self.scale * torch.randn(self.hidden, self.hidden))
        self.rb = nn.Parameter(self.scale * torch.randn(self.hidden))
        self.ib = nn.Parameter(self.scale * torch.randn(self.hidden))

        self.fc1 = nn.Sequential(
            nn.Conv1d(dim, self.hidden, kernel_size=1, bias=False),
            nn.GroupNorm(1, self.hidden),
            nn.GELU(),
        )
        self.fc2 = nn.Sequential(
            nn.Conv1d(self.hidden, dim, kernel_size=1, bias=False),
            nn.GroupNorm(1, dim),
        )

    def forward(self, x):
        # x: [B, N, C]
        B, N, C = x.shape
        # channel expand
        x = self.fc1(x.transpose(1, 2)).transpose(1, 2)  # [B, N, hidden]

        # rFFT on time axis (half-spectrum, complex)
        X = torch.fft.rfft(x, dim=1, norm='ortho')       # [B, N_r, hidden]
        Xr, Xi = X.real, X.imag                          # both [B, N_r, hidden]

        # [B, N_r, hidden] x [hidden, hidden] -> [B, N_r, hidden]
        Yr = F.gelu(torch.einsum('bnc,cd->bnd', Xr, self.r) -
                    torch.einsum('bnc,cd->bnd', Xi, self.i) + self.rb)
        Yi = F.gelu(torch.einsum('bnc,cd->bnd', Xi, self.r) +
                    torch.einsum('bnc,cd->bnd', Xr, self.i) + self.ib)

        Y = torch.complex(Yr, Yi)                        # [B, N_r, hidden]
        y = torch.fft.irfft(Y, n=N, dim=1, norm='ortho') # [B, N, hidden] (float)
        y = self.fc2(y.transpose(1, 2)).transpose(1, 2)  # [B, N, C]
        return y

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__(); self.eps=eps; self.gamma=nn.Parameter(torch.ones(dim))
    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.gamma

class ChannelBiMambaLite(nn.Module):
    def __init__(self,
                 n_channels: int,
                 seq_len_t: int,
                 e_dim: int = 64,
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand: int = 2,
                 dropout: float = 0.1,
                 use_rmsnorm: bool = True,
                 direction: str = 'bi'):   # 'bi' or 'fwd'
        super().__init__()
        self.C = n_channels
        self.T = seq_len_t
        self.E = e_dim
        self.direction = direction

        self.time_to_embed = nn.Linear(self.T, self.E, bias=False)
        self.embed_to_time = nn.Linear(self.E, self.T, bias=False)

        self.mamba_fwd = Mamba(d_model=self.E, d_state=d_state, d_conv=d_conv, expand=expand)
        if direction == 'bi':
            self.mamba_rev = Mamba(d_model=self.E, d_state=d_state, d_conv=d_conv, expand=expand)

        Norm = RMSNorm if use_rmsnorm else nn.LayerNorm
        self.ln_in   = Norm(self.E)
        self.ln_mout = Norm(self.E)
        self.drop    = nn.Dropout(dropout)
        self.res_scale = nn.Parameter(torch.tensor(1.0))

        self._init_like_isometry()

    def _init_like_isometry(self):
        nn.init.orthogonal_(self.time_to_embed.weight)
        with torch.no_grad():
            self.embed_to_time.weight.copy_(self.time_to_embed.weight.t())

    def forward(self, x):                 # x: (B, T, C)
        B, T, C = x.shape
        assert T == self.T and C == self.C, "ChannelBiMambaLite: (T,C) mismatch."

        xc = x.transpose(1, 2)                 # (B, C, T)
        z  = self.time_to_embed(xc)            # (B, C, E)
        z  = self.ln_in(z)

        y_fwd = self.mamba_fwd(z)              # (B, C, E)
        if self.direction == 'bi':
            y_rev = torch.flip(self.mamba_rev(torch.flip(z, dims=[1])), dims=[1])
            y_mid = (y_fwd + y_rev) * 0.5
        else:
            y_mid = y_fwd

        y_mid = self.ln_mout(y_mid)
        y_mid = self.drop(y_mid)

        y_time = self.embed_to_time(y_mid)     # (B, C, T)

        out = xc + self.res_scale * y_time     # (B, C, T)
        return out.transpose(1, 2)             # (B, T, C)


class MSMamba(nn.Module):
    def __init__(self, dim, mlp_ratio, drop_path=0., norm_layer=nn.LayerNorm, share_mamba=True,
                 use_channel_bimamba: bool = True,
                 frames: int = 160,
                 chan_e_dim: int = 64,
                 chan_dropout: float = 0.1,
                 ffn_type: str = 'freq',    # 'freq' or 'mlp'
                 chan_dir: str = 'bi'):      # 'bi' or 'fwd'
        super().__init__()
        self.share_mamba = share_mamba
        self.use_channel_bimamba = use_channel_bimamba

        self.norm1_rgb = norm_layer(dim)
        self.norm1_nir = norm_layer(dim)
        self.norm2     = norm_layer(dim)
        self.norm3 = norm_layer(dim)

        if share_mamba:
            self.attn = MambaLayer(dim)
        else:
            self.attn_rgb = MambaLayer(dim)
            self.attn_nir = MambaLayer(dim)

        self.mlp = Frequencydomain_FFN(dim, mlp_ratio) if ffn_type == 'freq' else StandardFFN(dim, mlp_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.res_scale_mamba = nn.Parameter(torch.tensor(1.0))
        self.res_scale_mlp   = nn.Parameter(torch.tensor(1.0))
        self.res_scale_chan  = nn.Parameter(torch.tensor(1.0))

        if self.use_channel_bimamba:
            self.chan_scan = ChannelBiMambaLite(
                n_channels=dim, seq_len_t=frames, e_dim=chan_e_dim,
                d_state=16, d_conv=4, expand=2, dropout=chan_dropout,
                direction=chan_dir,
            )

    def forward(self, rgb, nir):
        # rgb / nir can be None when that modality is absent.
        has_rgb = rgb is not None
        has_nir = nir is not None

        if has_rgb:
            if self.share_mamba:
                rgb_out = rgb + self.drop_path(self.attn(self.norm1_rgb(rgb)))
            else:
                rgb_out = rgb + self.drop_path(self.attn_rgb(self.norm1_rgb(rgb)))
        else:
            rgb_out = None

        if has_nir:
            if self.share_mamba:
                nir_out = nir + self.drop_path(self.attn(self.norm1_nir(nir)))
            else:
                nir_out = nir + self.drop_path(self.attn_nir(self.norm1_nir(nir)))
        else:
            nir_out = None

        # Fuse only the streams that are present.
        if has_rgb and has_nir:
            fused = rgb_out + nir_out
        elif has_rgb:
            fused = rgb_out
        else:
            fused = nir_out

        x_fused = self.norm2(fused)
        x_fused = x_fused + self.drop_path(self.res_scale_chan * self.mlp(x_fused))

        if self.use_channel_bimamba:
            x_fused = self.norm3(x_fused)
            x_fused = x_fused + self.res_scale_chan * self.drop_path(self.chan_scan(x_fused))

        rgb_final = (rgb_out + x_fused) if has_rgb else None
        nir_final = (nir_out + x_fused) if has_nir else None

        return rgb_final, nir_final

class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=48, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  
            d_state=d_state,  
            d_conv=d_conv, 
            expand=expand  
        )
    def forward(self, x):
        B, N, C = x.shape
        x_mamba = self.mamba(x)    
        return x_mamba

# ------------------------- Cross-FiLM (원본 유지) -------------------------
class CrossFiLMGatingHP(nn.Module):
    """
    x_self <- FiLM( HP(x_other) ) + residual
    - depthwise HP: 조명 DC/플리커 누출 감소
    - 대칭 사용을 위해 같은 모듈을 RGB<-NIR, NIR<-RGB 모두에 재사용(share=True) 가능
    """
    def __init__(self, c, k_hp=3, k_mixer=5, dil=1,
                 gamma_lim=0.25, beta_lim=0.5, gating_drop_p=0.0):
        super().__init__()
        pad_hp = (k_hp-1)//2
        self.hp = nn.Conv1d(c, c, k_hp, padding=pad_hp, groups=c, bias=True)
        with torch.no_grad():                 # [1,-1,0] 초기화
            w = torch.zeros(c,1,k_hp); w[:,:,0]=1; w[:,:,1]=-1
            self.hp.weight.copy_(w)
        pad = (k_mixer-1)//2
        self.mix = nn.Conv1d(c, 2*c, kernel_size=k_mixer, padding=pad*dil,
                             dilation=dil, groups=c, bias=True)  # depthwise
        self.proj = nn.Conv1d(2*c, 2*c, kernel_size=1, bias=True)
        self.act  = nn.SiLU()
        self.gamma_lim, self.beta_lim = gamma_lim, beta_lim
        self.gate_drop = gating_drop_p

    def forward(self, x_self, x_other):       # both [B,T,C]
        z = x_other.transpose(1,2)       # [B,C,T]
        z = self.hp(z)
        z = self.act(self.mix(z))
        gb = self.proj(z)
        g, b = gb.chunk(2, dim=1)
        g = torch.tanh(g) * self.gamma_lim
        b = torch.tanh(b) * self.beta_lim
        if self.training and self.gate_drop>0:
            if torch.rand(1, device=x_self.device) < self.gate_drop:
                g = torch.zeros_like(g); b = torch.zeros_like(b)
        y = x_self.transpose(1,2) * (1+g) + b
        return y.transpose(1,2)
    
class FrequencyFiLMGating(nn.Module):
    """
    FiLM gating using physiological band information.
    use_fft=True (default): FFT → band-pass → IFFT before mix/proj (CSLM).
    use_fft=False: x_other fed directly to mix/proj (1D conv ablation).
    """
    def __init__(self, c, frames, fs=30, low_bpm=40, high_bpm=200,
                 k_mixer=5, dil=1, gamma_lim=0.25, beta_lim=0.5, gating_drop_p=0.0,
                 use_fft=True):
        super().__init__()
        self.frames = frames
        self.use_fft = use_fft

        if use_fft:
            freqs = torch.fft.rfftfreq(frames, d=1.0/fs)
            low_hz, high_hz = low_bpm / 60.0, high_bpm / 60.0
            self.register_buffer("band_mask", (freqs >= low_hz) & (freqs <= high_hz))

        pad = (k_mixer-1)//2 * dil
        self.mix = nn.Conv1d(c, 2*c, kernel_size=k_mixer, padding=pad,
                             dilation=dil, groups=c, bias=True)
        self.proj = nn.Conv1d(2*c, 2*c, kernel_size=1, bias=True)
        self.act = nn.SiLU()
        self.gamma_lim, self.beta_lim = gamma_lim, beta_lim
        self.gate_drop = gating_drop_p

    def forward(self, x_self, x_other): # both [B, T, C]
        if self.use_fft:
            z_freq = torch.fft.rfft(x_other, dim=1, norm='ortho')
            z_time = torch.fft.irfft(z_freq * self.band_mask.view(1, -1, 1),
                                     n=self.frames, dim=1, norm='ortho')
        else:
            z_time = x_other

        z = z_time.transpose(1, 2)  # [B, C, T]
        z = self.act(self.mix(z))
        gb = self.proj(z)
        g, b = gb.chunk(2, dim=1)
        g = torch.tanh(g) * self.gamma_lim
        b = torch.tanh(b) * self.beta_lim

        if self.training and self.gate_drop > 0:
            if torch.rand(1, device=x_self.device) < self.gate_drop:
                g = torch.zeros_like(g)
                b = torch.zeros_like(b)

        y = x_self.transpose(1, 2) * (1 + g) + b
        return y.transpose(1, 2)
# ------------------------- Heads & NIR attention (원본 유지) -------------------------
class rPPGHead(nn.Module):
    def __init__(self, dim, k=5):
        super().__init__()
        pad = (k - 1) // 2
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=k, padding=pad, groups=dim, bias=False)
        self.conv2 = nn.Conv1d(dim, 1, 1, bias=False)

        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='linear') 
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='linear')
    def forward(self, x):  # x: [B, T, C]
        x = self.conv1(x.transpose(1,2))          # [B, C, T]
        x = self.conv2(x).squeeze(1)    # [B, T]
        return x

# ------------------------- Model -------------------------
class MsMamba(nn.Module):
    """
    Multi-spectral rPPG model (RGB + NIR).
    All design choices are controlled by keyword arguments so ablation variants
    can be created by changing a single parameter without editing this class.
    """
    def __init__(self, frames=160, depth=4, embed_dim=96, mlp_ratio=2,
                 drop_path_rate=0.1,
                 share_crossfilm=True, gating_drop_p=0.0,
                 share_mamba=True,
                 use_channel_bimamba=True,
                 chan_dir='bi',
                 ffn_type='freq',
                 crossfilm_type='freq',
                 trunk_type='full',
                 trunk_channels=64,
                 conv_type='std'):
        super().__init__()

        self.embed_dim = embed_dim
        self.crossfilm_type = crossfilm_type
        self.trunk_type = trunk_type
        assert trunk_type in ('full', 'slim', 'lite'), \
            f"trunk_type must be 'full', 'slim', or 'lite', got '{trunk_type}'"
        assert conv_type in ('std', 'dw'), \
            f"conv_type must be 'std' or 'dw', got '{conv_type}'"
        tc = trunk_channels   # shorthand
        # GroupNorm: keep 2 channels per group (same ratio as original 32 groups / 64 ch)
        gn = max(1, tc // 2)

        def CB(c_in):
            if conv_type == 'dw':
                # Depthwise-separable 3D: DW(3×3×3) + PW(1×1×1)
                return nn.Sequential(
                    nn.Conv3d(c_in, c_in, 3, 1, 1, groups=c_in),
                    nn.Conv3d(c_in, tc, 1),
                    nn.GroupNorm(gn, tc), nn.ReLU(inplace=True))
            else:
                return nn.Sequential(
                    nn.Conv3d(c_in, tc, 3, 1, 1),
                    nn.GroupNorm(gn, tc), nn.ReLU(inplace=True))

        # 3D trunk
        # CB1: fixed at 16 output channels (tiny cost, scales with input resolution only)
        self.ConvBlock1_rgb = nn.Sequential(
            nn.Conv3d(3, 16, [1,5,5], padding=[0,2,2]),
            nn.BatchNorm3d(16), nn.ReLU(inplace=True))
        self.ConvBlock1_nir = nn.Sequential(
            nn.Conv3d(1, 16, [1,5,5], padding=[0,2,2]),
            nn.BatchNorm3d(16), nn.ReLU(inplace=True))
        self.MaxpoolSpa    = nn.MaxPool3d((1,2,2), stride=(1,2,2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2,2,2), stride=2)
        # stage 1 – always present
        self.ConvBlock2 = CB(16); self.ConvBlock3 = CB(tc)
        # stage 2 – always present
        self.ConvBlock4 = CB(tc); self.ConvBlock5 = CB(tc)
        # stage 3 – full / slim only
        if trunk_type in ('full', 'slim'):
            self.ConvBlock6 = CB(tc); self.ConvBlock7 = CB(tc)
        # stage 4 – full only
        if trunk_type == 'full':
            self.ConvBlock8 = CB(tc); self.ConvBlock9 = CB(tc)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose3d(tc, tc, [4,1,1], stride=[2,1,1], padding=[1,0,0]),
            nn.GroupNorm(gn, tc), nn.ELU())
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose3d(tc, tc, [4,1,1], stride=[2,1,1], padding=[1,0,0]),
            nn.GroupNorm(gn, tc), nn.ELU())

        self.poolspa = nn.AdaptiveAvgPool3d((frames,1,1))
        self.to_embed = nn.Conv3d(tc, embed_dim, 1)

        # Cross-FiLM (only instantiated when crossfilm_type != 'none')
        if crossfilm_type in ('freq', 'conv'):
            film_fn = partial(FrequencyFiLMGating, c=embed_dim, frames=frames, fs=30,
                              gating_drop_p=gating_drop_p,
                              use_fft=(crossfilm_type == 'freq'))
            if share_crossfilm:
                _film = film_fn()
                self.film_r_from_n = _film
                self.film_n_from_r = _film
            else:
                self.film_r_from_n = film_fn()
                self.film_n_from_r = film_fn()
        elif crossfilm_type == 'hp':
            _film = CrossFiLMGatingHP(embed_dim, gating_drop_p=gating_drop_p)
            self.film_r_from_n = _film if share_crossfilm else CrossFiLMGatingHP(embed_dim, gating_drop_p=gating_drop_p)
            self.film_n_from_r = _film if share_crossfilm else CrossFiLMGatingHP(embed_dim, gating_drop_p=gating_drop_p)
        # crossfilm_type == 'none': no film modules

        self.ln_rgb = nn.LayerNorm(embed_dim)
        self.ln_nir = nn.LayerNorm(embed_dim)

        # MSMamba blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        inter_dpr = [0.0] + dpr
        self.blocks = nn.ModuleList([MSMamba(
                dim=embed_dim,
                mlp_ratio=mlp_ratio,
                drop_path=inter_dpr[i],
                norm_layer=nn.LayerNorm,
                share_mamba=share_mamba,
                use_channel_bimamba=use_channel_bimamba,
                frames=frames,
                chan_e_dim=embed_dim,
                chan_dropout=0.1,
                ffn_type=ffn_type,
                chan_dir=chan_dir,
            )
            for i in range(depth)
        ])

        # Head
        self.rPPGHead = rPPGHead(embed_dim, k=5)

        # init
        self.apply(segm_init_weights)
        self.apply(partial(_init_weights, n_layer=depth))

    def _trunk_prepool(self, x: torch.Tensor, first: nn.Module) -> torch.Tensor:
        x = x.permute(0,2,1,3,4)                              # [B,C,T,H,W]
        x = first(x); x = self.MaxpoolSpa(x)                  # [B,16,T,H/2,W/2]
        x = self.ConvBlock2(x); x = self.ConvBlock3(x)        # [B,64,T,H/2,W/2]
        x = self.MaxpoolSpaTem(x)                              # [B,64,T/2,H/4,W/4]
        x = self.ConvBlock4(x); x = self.ConvBlock5(x)        # [B,64,T/2,H/4,W/4]
        x = self.MaxpoolSpaTem(x)                              # [B,64,T/4,H/8,W/8]
        if self.trunk_type in ('full', 'slim'):
            x = self.ConvBlock6(x); x = self.ConvBlock7(x)    # [B,64,T/4,H/8,W/8]
            x = self.MaxpoolSpa(x)                             # [B,64,T/4,H/16,W/16]
        if self.trunk_type == 'full':
            x = self.ConvBlock8(x); x = self.ConvBlock9(x)    # [B,64,T/4,H/16,W/16]
        x = self.deconv1(x); x = self.deconv2(x)              # [B,64,T,H',W']
        return x

    def _to_seq(self, x_pre: torch.Tensor) -> torch.Tensor:  # -> [B,T,C]
        x = self.poolspa(x_pre)               # [B,64,T,1,1]
        x = self.to_embed(x).squeeze(-1).squeeze(-1)  # [B,C,T]
        return x.transpose(1, 2)              # [B,T,C]

    def forward(self, rgb: torch.Tensor = None, nir: torch.Tensor = None) -> torch.Tensor:
        has_rgb = rgb is not None
        has_nir = nir is not None

        if not has_rgb and not has_nir:
            raise ValueError("MsMamba.forward: at least one of rgb or nir must be provided.")

        # Channel-slice guards (DiffNormalized may have extra channels).
        if has_rgb and rgb.shape[2] > 3:
            rgb = rgb[:, :, 3:, :, :]
        if has_nir and nir.shape[2] > 1:
            nir = nir[:, :, :1, :, :]

        # ── Trunk (only for available modalities) ────────────────────────────
        feat_rgb = self._to_seq(self._trunk_prepool(rgb, self.ConvBlock1_rgb)) if has_rgb else None
        feat_nir = self._to_seq(self._trunk_prepool(nir, self.ConvBlock1_nir)) if has_nir else None

        # ── CrossFiLM (only when both streams are present) ───────────────────
        if has_rgb and has_nir and self.crossfilm_type != 'none':
            feat_nir = self.ln_nir(self.film_n_from_r(feat_nir, feat_rgb))
            feat_rgb = self.ln_rgb(self.film_r_from_n(feat_rgb, feat_nir))
        else:
            if has_rgb:
                feat_rgb = self.ln_rgb(feat_rgb)
            if has_nir:
                feat_nir = self.ln_nir(feat_nir)

        # ── MSMamba blocks ────────────────────────────────────────────────────
        x_rgb, x_nir = feat_rgb, feat_nir
        for blk in self.blocks:
            x_rgb, x_nir = blk(x_rgb, x_nir)

        # ── Final fusion ──────────────────────────────────────────────────────
        if has_rgb and has_nir:
            x_final = (x_rgb + x_nir) / 2
        elif has_rgb:
            x_final = x_rgb
        else:
            x_final = x_nir

        return self.rPPGHead(x_final)
