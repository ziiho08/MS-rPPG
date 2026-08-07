"""
PhysDrive illumination breakdown from saved pickle files.
Computes MAE/RMSE/MAPE/Pearson per illumination group.
Skips MACC (O(N^2)) to keep computation fast.
"""
import os, pickle, sys, torch, numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from collections import defaultdict
import scipy.signal
from scipy.signal import butter
from evaluation.post_process import _detrend, _calculate_fft_hr

ILLUM_MAP   = {'Z': 'Noon', 'H': 'Dawn_Dusk', 'W': 'Night', 'Y': 'Rainy'}
ILLUM_ORDER = ['Noon', 'Dawn_Dusk', 'Night', 'Rainy']
BASE = 'runs/exp'

MODELS = {
    'DeepPhys':     (f'{BASE}/diff_stand/saved_test_outputs/PhysDrive_deepphys_Epoch29_Seed50_PhysDrive_outputs.pickle',     True),
    'PhysNet':      (f'{BASE}/Diff_128/saved_test_outputs/PhysDrive_physnet_Epoch29_Seed50_PhysDrive_outputs.pickle',        True),
    'PhysMamba':    (f'{BASE}/Diff_128/saved_test_outputs/PhysDrive_physmamba_Epoch29_Seed50_PhysDrive_outputs.pickle',      True),
    'MSMamba':      (f'{BASE}/Diff_160/saved_test_outputs/PhysDrive_MSMamba_Epoch29_Seed50_PhysDrive_outputs.pickle',        True),
    'RhythmFormer': (f'{BASE}/Standardized_160/saved_test_outputs/PhysDrive_RhythmFormer_Epoch29_Seed50_PhysDrive_outputs.pickle', False),
    'RhythmMamba':  (f'{BASE}/Standardized_160/saved_test_outputs/PhysDrive_RhythmMamba_Epoch29_Seed0_PhysDrive_outputs.pickle',   False),
}


def parse_illum(fn):
    try:    return ILLUM_MAP.get(fn.split('_')[0][2])
    except: return None


def reform(data):
    sd = sorted(data.items(), key=lambda x: x[0])
    return torch.cat([v for _, v in sd], dim=0).cpu().numpy().reshape(-1)


def get_hr_fft(signal, diff_flag, fs=30):
    sig = _detrend(np.cumsum(signal), 100) if diff_flag else _detrend(signal, 100)
    b, a = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
    sig = scipy.signal.filtfilt(b, a, np.double(sig))
    return _calculate_fft_hr(sig, fs=fs)


def compute(path, diff_flag, fs=30):
    print(f'  Loading {path.split("/")[-1]} ...', flush=True)
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    recs = defaultdict(list)
    for fn in obj['predictions']:
        il = parse_illum(fn)
        if not il:
            continue
        p = reform(obj['predictions'][fn])
        l = reform(obj['labels'][fn])
        pred_hr  = get_hr_fft(p, diff_flag, fs)
        label_hr = get_hr_fft(l, diff_flag, fs)
        recs[il].append((label_hr, pred_hr))
    out = {}
    for il in ILLUM_ORDER:
        if il not in recs:
            continue
        gts = np.array([r[0] for r in recs[il]])
        ps  = np.array([r[1] for r in recs[il]])
        e   = ps - gts
        n   = len(e)
        out[il] = dict(
            n       = n,
            MAE     = float(np.mean(np.abs(e))),
            RMSE    = float(np.sqrt(np.mean(e**2))),
            MAPE    = float(np.mean(np.abs(e / (gts + 1e-9))) * 100),
            Pearson = float(np.corrcoef(ps, gts)[0, 1]) if n >= 2 else float('nan'),
        )
    return out


if __name__ == '__main__':
    AR = {}
    for m, (p, d) in MODELS.items():
        print(f'\n[{m}]', flush=True)
        AR[m] = compute(p, d)

    for metric in ['MAE', 'RMSE', 'MAPE', 'Pearson']:
        print(f'\n{"="*66}')
        print(f'  {metric}')
        print(f'{"="*66}')
        print(f"  {'Model':<14}" + ''.join(f'{il:>12}' for il in ILLUM_ORDER))
        print('  ' + '-' * 62)
        for model, res in AR.items():
            row = f"  {model:<14}"
            for il in ILLUM_ORDER:
                row += f"{res[il][metric]:>12.2f}" if il in res else f"{'—':>12}"
            print(row)

    print(f'\n{"="*66}')
    print(f'  Sessions per illumination group')
    print(f'{"="*66}')
    print(f"  {'Model':<14}" + ''.join(f'{il:>12}' for il in ILLUM_ORDER) + f"{'Total':>8}")
    print('  ' + '-' * 70)
    for model, res in AR.items():
        row = f"  {model:<14}"
        t = 0
        for il in ILLUM_ORDER:
            n = res[il]['n'] if il in res else 0
            row += f"{n:>12}"
            t += n
        print(row + f"{t:>8}")

    print('\n* RhythmFormer/RhythmMamba: Dawn_Dusk 없음 (Std_160 test set에 미포함)')
    print('* MSMamba: Night 5, Rainy 11 (MULTI_SPECTRAL 전처리 시 일부 세션 누락)')
    print('* 나머지 모델 (DeepPhys/PhysNet/PhysMamba): Night 12, Rainy 12 (35세션)')
