'''
For calculating Computational Cost (+ fvcore cross-check)
measure_model()로 여러 모델 비교 가능
'''
import sys, os, math
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import subprocess
from thop import profile, clever_format

# pip install fvcore
from fvcore.nn import FlopCountAnalysis, parameter_count_table

from mamba_ssm.modules.mamba_simple import Mamba

from neural_methods.model.DeepPhys import DeepPhys
from neural_methods.model.EfficientPhys import EfficientPhys
from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
from neural_methods.model.PhysFormer import ViT_ST_ST_Compact3_TDC_gra_sharp
from neural_methods.model.RhythmFormer import RhythmFormer
from neural_methods.model.RhythmMamba import RhythmMamba
from neural_methods.model.MsMamba import MsMamba
from neural_methods.model.PhysMamba import PhysMamba

device = torch.device("cuda:0")

########################################################################
# Mamba 커스텀 FLOPs 핸들러 (thop용)
# selective_scan은 커스텀 CUDA 커널이라 thop/fvcore가 추적 불가 → 이론값으로 보정
#
# Mamba SSM (d_model=D, d_state=N, d_conv=k, expand=E, seq_len=L):
#   d_inner   = D * E
#   dt_rank   = ceil(D / 16)
#   in_proj   : 2·L·D·(2·d_inner)         Linear, no bias
#   conv1d    : 2·L·d_inner·k             depthwise causal conv
#   x_proj    : 2·L·d_inner·(dt_rank+2N)  Linear
#   dt_proj   : 2·L·dt_rank·d_inner       Linear
#   SSM scan  : 9·L·d_inner·N             selective scan 핵심 연산
#   out_proj  : 2·L·d_inner·D             Linear
########################################################################
def _mamba_flops_counter(m: Mamba, x, y):
    """thop custom_ops 핸들러 — Mamba 모듈의 이론적 FLOPs를 계산.

    RhythmMamba의 Multi-temporal Parallelization처럼 내부적으로 배치를
    repeat하는 경우도 정확히 반영하기 위해 실제 입력 B를 사용.
    """
    inp = x[0]                          # [B, L, D]
    B, L, D = inp.shape

    d_inner  = m.d_inner                # D * expand
    N        = m.d_state
    k        = m.d_conv
    dt_rank  = m.dt_rank                # math.ceil(D/16) by default

    flops  = 2 * L * D * (2 * d_inner)          # in_proj
    flops += 2 * L * d_inner * k                 # conv1d (depthwise)
    flops += 2 * L * d_inner * (dt_rank + 2 * N) # x_proj
    flops += 2 * L * dt_rank * d_inner            # dt_proj
    flops += 9 * L * d_inner * N                  # SSM selective scan
    flops += 2 * L * d_inner * D                  # out_proj

    m.total_ops += torch.DoubleTensor([int(B * flops)])

def get_gpu_info():
    result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
    return result.stdout.decode('utf-8')


########################################################################
# 공통 측정 함수 (단일/멀티모달 모두 지원)
########################################################################
def measure_model(model_name, model, inputs, iterations=1000, frames=None):
    """
    Args:
        model_name : 출력에 표시될 이름
        model      : nn.Module (이미 .to(device)된 것 권장)
        inputs     : tuple of tensors — forward(*inputs) 형태로 호출
        iterations : 지연시간 측정 반복 횟수
        frames     : (optional) 프레임 수 직접 지정; None이면 자동 추론
    """
    model = model.to(device).eval()

    # ── Warm-up ──────────────────────────────────────────────
    with torch.no_grad():
        for _ in range(50):
            model(*inputs)

    # ── Inference latency / FPS ───────────────────────────────
    starter = torch.cuda.Event(enable_timing=True)
    ender   = torch.cuda.Event(enable_timing=True)
    times   = torch.zeros(iterations, device=device)
    with torch.no_grad():
        for i in range(iterations):
            starter.record()
            model(*inputs)
            ender.record()
            torch.cuda.synchronize()
            times[i] = starter.elapsed_time(ender)  # ms

    mean_ms  = times.mean().item()
    fps      = 1000.0 / mean_ms

    # T: 명시 지정 > 자동 추론 (batch=1 NDCHW → dim1, 나머지 → dim0)
    if frames is not None:
        T = frames
    elif inputs[0].dim() == 5:
        T = inputs[0].shape[1]   # [B, T, C, H, W]
    else:
        T = inputs[0].shape[0]   # [T, C, H, W]
    kfps = (T * fps) / 1000.0

    # ── THOP (Mamba 커스텀 핸들러 포함) ─────────────────────────
    custom_ops = {Mamba: _mamba_flops_counter}
    flops_thop, params_thop = profile(model, inputs=inputs, verbose=False,
                                      custom_ops=custom_ops)
    macs_thop = flops_thop / 2.0

    # ── fvcore ───────────────────────────────────────────────
    fca          = FlopCountAnalysis(model, inputs)
    fca.unsupported_ops_warnings(False)
    flops_fvcore = fca.total()
    macs_fvcore  = flops_fvcore / 2.0
    params_count = sum(p.numel() for p in model.parameters())

    # fvcore는 Mamba SSM FLOPs를 0으로 잡으므로, thop 기준값을 신뢰
    rel_gap = abs(flops_fvcore - flops_thop) / max(flops_fvcore, flops_thop, 1.0)

    # ── 출력 ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Model : {model_name}  (T={T})")
    print(f"  Latency : {mean_ms:.3f} ms | FPS : {fps:.2f} | Kfps : {kfps:.3f}")
    print(f"  Params        (M) : {params_count/1e6:.3f}")
    print(f"  THOP  FLOPs   (G) : {flops_thop/1e9:.3f}  [Mamba SSM 이론값 포함]")
    print(f"  THOP  MACs    (G) : {macs_thop/1e9:.3f}")
    print(f"  fvcore FLOPs  (G) : {flops_fvcore/1e9:.3f}  [Mamba SSM = 0, 과소추정]")
    print(f"  fvcore MACs   (G) : {macs_fvcore/1e9:.3f}")
    print(f"  FLOPs gap (|thop-fvcore|/max) : {100*rel_gap:.1f}%")
    print(f"{'='*60}")

    return dict(
        model=model_name, T=T,
        mean_ms=mean_ms, fps=fps, kfps=kfps,
        params_M=params_count/1e6,
        flops_thop_G=flops_thop/1e9, macs_thop_G=macs_thop/1e9,
        flops_fvcore_G=flops_fvcore/1e9, macs_fvcore_G=macs_fvcore/1e9,
    )


########################################################################
# PhysFormer wrapper: gra_sharp(float)를 내부 고정해 단일 텐서 입력으로 통일
########################################################################
class PhysFormerWrapper(nn.Module):
    def __init__(self, frames=160, gra_sharp=2.0):
        super().__init__()
        self.model = ViT_ST_ST_Compact3_TDC_gra_sharp(
            image_size=(frames, 128, 128), patches=(4, 4, 4), dim=96, ff_dim=144,
            num_heads=4, num_layers=12, dropout_rate=0.1, theta=0.7)
        self.gra_sharp = gra_sharp

    def forward(self, x):
        rppg, _, _, _ = self.model(x, self.gra_sharp)
        return rppg


########################################################################
# GPU 정보
########################################################################
print(get_gpu_info())

########################################################################
# 측정 대상 정의
# ── 공통 해상도 : 128×128  (EfficientPhys만 96×96 — 모델 제약)
# ── 공통 프레임 수 : T=160 (실험 CHUNK_LENGTH)
########################################################################
T = 160

results = []

# ── [내 모델] MsMamba – RGB only ─────────────────────────────────────
rgb = torch.randn(1, T, 3, 128, 128).to(device)
results.append(measure_model(
    "MsMamba (RGB)",
    MsMamba(frames=T, depth=4, embed_dim=96),  # crossfilm은 RGB 단독 시 자동 비활성
    (rgb,),
    frames=T,
))

# ── [내 모델] MsMamba – RGB + NIR ────────────────────────────────────
rgb = torch.randn(1, T, 3, 128, 128).to(device)
nir = torch.randn(1, T, 1, 128, 128).to(device)
results.append(measure_model(
    "MsMamba (RGB+NIR)",
    MsMamba(frames=T, depth=4, embed_dim=96),
    (rgb, nir),
    frames=T,
))

# ── DeepPhys ─────────────────────────────────────────────────────────
# 입력: [T, 6, H, W]  (6ch = diff 3ch + raw 3ch); img_size=96 (96 max)
inp_dp = torch.randn(T, 6, 96, 96).to(device)
results.append(measure_model(
    "DeepPhys (96²)",
    DeepPhys(img_size=96),
    (inp_dp,),
    frames=T,
))

# ── EfficientPhys ─────────────────────────────────────────────────────
# 모델 내부에서 torch.diff → T+1 프레임 필요; img_size=96 (128 미지원)
inp_ep = torch.randn(T + 1, 3, 96, 96).to(device)
results.append(measure_model(
    "EfficientPhys (96²)",
    EfficientPhys(frame_depth=10, img_size=96),
    (inp_ep,),
    frames=T,
))

# ── PhysFormer ────────────────────────────────────────────────────────
# gra_sharp 인수를 wrapper로 고정
inp_pf = torch.randn(1, T, 3, 128, 128).to(device)
results.append(measure_model(
    "PhysFormer",
    PhysFormerWrapper(frames=T),
    (inp_pf,),
    frames=T,
))

# ── PhysMamba ─────────────────────────────────────────────────────────
# 입력: [B, 3, T, H, W] (NCDHW)
inp_pm = torch.randn(1, 3, T, 128, 128).to(device)
results.append(measure_model(
    "PhysMamba",
    PhysMamba(frames=T),
    (inp_pm,),
    frames=T,
))

# ── PhysNet ───────────────────────────────────────────────────────────
# 입력: [B, 3, T, H, W] (NCDHW)
inp_pn = torch.randn(1, 3, T, 128, 128).to(device)
results.append(measure_model(
    "PhysNet",
    PhysNet_padding_Encoder_Decoder_MAX(frames=T),
    (inp_pn,),
    frames=T,
))

# ── RhythmFormer ──────────────────────────────────────────────────────
# 입력: [B, T, C, H, W] (NDCHW)
inp_rfor = torch.randn(1, T, 3, 128, 128).to(device)
results.append(measure_model(
    "RhythmFormer",
    RhythmFormer(frame=T, image_size=(T, 128, 128)),
    (inp_rfor,),
    frames=T,
))

# ── RhythmMamba ───────────────────────────────────────────────────────
# 입력: [B, T, C, H, W] (NDCHW)
inp_rmam = torch.randn(1, T, 3, 128, 128).to(device)
results.append(measure_model(
    "RhythmMamba",
    RhythmMamba(),
    (inp_rmam,),
    frames=T,
))

########################################################################
# 요약 테이블
########################################################################
print("\n" + "="*88)
print("{:<22} {:>5} {:>10} {:>12} {:>14} {:>16}".format(
    "Model", "T", "Params(M)", "Latency(ms)", "THOP MACs(G)", "fvcore MACs(G)"))
print("-"*88)
for r in results:
    print("{:<22} {:>5} {:>10.3f} {:>12.3f} {:>14.3f} {:>16.3f}".format(
        r['model'], r['T'], r['params_M'],
        r['mean_ms'], r['macs_thop_G'], r['macs_fvcore_G']))
print("="*88)
print("* DeepPhys / EfficientPhys: 모델 구조상 96×96 입력 사용 (train 설정과 동일)")
print("* THOP MACs: Mamba SSM 이론값 포함 (신뢰 기준)")
print("* fvcore MACs: Mamba SSM FLOPs = 0으로 계산 (과소추정)")
