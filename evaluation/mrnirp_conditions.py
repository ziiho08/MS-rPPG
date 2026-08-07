"""
MR-NIRP condition-based metric breakdown.

Parses MR-NIRP-CAR filenames (e.g. ``subject1_driving_still_motion_940``)
into condition dimensions and reports HR metrics grouped by motion level
and NIR wavelength.  Only ``driving`` sessions are included; ``garage``
sessions are skipped.

Condition dimensions
--------------------
  motion     : still / small / large
  wavelength : 940nm / 975nm
"""

import os
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

from evaluation.post_process import calculate_metric_per_video

# ── Naming-convention maps ────────────────────────────────────────────────────
MOTION_ORDER     = ['still', 'small', 'large']
WAVELENGTH_ORDER = ['940nm', '975nm']

DIM_ORDER = {
    'motion':     MOTION_ORDER,
    'wavelength': WAVELENGTH_ORDER,
}


def parse_mrnirp_filename(filename: str):
    """Parse ``subject1_driving_still_motion_940`` → condition dict.

    Expected token layout:
        subject{N} _ driving _ {still|small|large} _ motion _ {940|975}

    Returns a dict with keys ``motion`` and ``wavelength``,
    or ``None`` for garage sessions or unrecognised filenames.
    """
    try:
        parts = filename.split('_')
        # Need at least: subject, location, motion_level, 'motion', wavelength
        if len(parts) < 5:
            return None
        location       = parts[1]           # 'driving' or 'garage'
        if location != 'driving':
            return None                     # skip garage
        motion_level   = parts[2]           # still / small / large
        wavelength_str = parts[4]           # '940' or '975'
        if motion_level not in ('still', 'small', 'large'):
            return None
        return {
            'motion':     motion_level,
            'wavelength': wavelength_str + 'nm',
        }
    except Exception:
        return None


# ── Per-video HR collection ───────────────────────────────────────────────────

def _collect_per_video_hr(predictions, labels, config):
    """Collect per-window HR estimates with condition tags.

    Returns
    -------
    records : list of dict
        One entry per evaluation window, each with keys:
        ``filename``, ``gt_hr``, ``pred_hr``, ``snr``, ``macc``,
        ``motion``, ``wavelength``.
        Garage sessions are omitted.
    """
    import torch

    def _reform(data):
        sort_data = sorted(data.items(), key=lambda x: x[0])
        sort_data = torch.cat([v for _, v in sort_data], dim=0)
        return np.reshape(sort_data.cpu().numpy(), (-1))

    diff_flag   = config.TEST.DATA.PREPROCESS.LABEL_TYPE == 'DiffNormalized'
    eval_method = config.INFERENCE.EVALUATION_METHOD
    fs          = config.TEST.DATA.FS
    use_window  = config.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW
    win_sec     = config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE

    records = []
    for filename in predictions:
        cond = parse_mrnirp_filename(filename)
        if cond is None:
            continue                        # skip garage / unparseable

        prediction = _reform(predictions[filename])
        label      = _reform(labels[filename])

        video_len  = prediction.shape[0]
        win_frames = int(win_sec * fs) if use_window else video_len
        win_frames = min(win_frames, video_len)

        for i in range(0, video_len, win_frames):
            pred_w  = prediction[i:i + win_frames]
            label_w = label[i:i + win_frames]
            if len(pred_w) < 9:
                continue

            gt_hr, pred_hr, snr, macc = calculate_metric_per_video(
                pred_w, label_w,
                diff_flag=diff_flag,
                fs=fs,
                hr_method='FFT' if eval_method == 'FFT' else 'Peak',
            )
            rec = {
                'filename':  filename,
                'gt_hr':     gt_hr,
                'pred_hr':   pred_hr,
                'snr':       snr,
                'macc':      macc,
            }
            rec.update(cond)
            records.append(rec)

    return records


# ── Stats helpers ─────────────────────────────────────────────────────────────

def _stats(arr):
    arr = np.asarray(arr, dtype=float)
    n   = len(arr)
    if n == 0:
        return float('nan'), float('nan')
    return float(np.mean(arr)), float(np.std(arr) / np.sqrt(n))


def _group_stats(records, dim, config):
    requested = set(config.TEST.METRICS)

    groups         = defaultdict(list)
    session_counts = defaultdict(set)
    subject_counts = defaultdict(set)
    for rec in records:
        groups[rec[dim]].append(rec)
        session_counts[rec[dim]].add(rec['filename'])
        subject_counts[rec[dim]].add(rec['filename'].split('_')[0])

    ordered_keys = DIM_ORDER.get(dim, sorted(groups.keys()))
    rows = []
    for key in ordered_keys:
        if key not in groups:
            continue
        g      = groups[key]
        n_win  = len(g)
        n_sess = len(session_counts[key])
        n_subj = len(subject_counts[key])

        gt_hrs   = np.array([r['gt_hr']   for r in g])
        pred_hrs = np.array([r['pred_hr'] for r in g])
        snrs     = np.array([r['snr']     for r in g])
        maccs    = np.array([r['macc']    for r in g])

        m = {}
        if 'MAE'     in requested:
            m['MAE']     = _stats(np.abs(pred_hrs - gt_hrs))
        if 'RMSE'    in requested:
            errors_sq = np.square(pred_hrs - gt_hrs)
            rmse_val  = float(np.sqrt(np.mean(errors_sq)))
            rmse_se   = float(np.sqrt(np.std(errors_sq) / np.sqrt(n_win))) if n_win > 0 else float('nan')
            m['RMSE'] = (rmse_val, rmse_se)
        if 'MAPE'    in requested:
            m['MAPE']    = _stats(np.abs((pred_hrs - gt_hrs) / (gt_hrs + 1e-9)) * 100)
        if 'Pearson' in requested:
            if n_win >= 2:
                r_val = float(np.corrcoef(pred_hrs, gt_hrs)[0, 1])
                se    = float(np.sqrt(max(0, 1 - r_val**2) / max(1, n_win - 2)))
            else:
                r_val, se = float('nan'), float('nan')
            m['Pearson'] = (r_val, se)
        if 'SNR'     in requested:
            m['SNR']     = _stats(snrs)
        if 'MACC'    in requested:
            m['MACC']    = _stats(maccs)

        rows.append((key, n_subj, n_sess, n_win, m))
    return rows


# ── Printing ──────────────────────────────────────────────────────────────────

def _print_condition_table(dim_label, rows, requested_metrics):
    active = [m for m in ['MAE', 'RMSE', 'MAPE', 'Pearson', 'SNR', 'MACC']
              if m in requested_metrics]

    col_w  = 12
    met_w  = 9
    header = f"  {'Group':<{col_w}}  {'Subjects':>8}  {'Sessions':>8}  {'Windows':>8}"
    for m in active:
        header += f"  {m:>{met_w}}"
    sep = '─' * len(header)

    print(f"\n[{dim_label}]")
    print(sep)
    print(header)
    print(sep)
    for key, n_subj, n_sess, n_win, m in rows:
        line = f"  {key:<{col_w}}  {n_subj:>8}  {n_sess:>8}  {n_win:>8}"
        for metric in active:
            if metric in m:
                mean, se = m[metric]
                line += f"  {mean:>{met_w - 4}.2f}±{se:.2f}"
            else:
                line += f"  {'—':>{met_w}}"
        print(line)
    print(sep)


# ── Excel saving ──────────────────────────────────────────────────────────────

def _save_condition_excel(all_dim_rows, filename_id, config, eval_method):
    save_dir = config.TEST.EXCEL_SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)
    xlsx_path = os.path.join(save_dir, f'{filename_id}_mrnirp_conditions.xlsx')

    requested = set(config.TEST.METRICS)
    active = [m for m in ['MAE', 'RMSE', 'MAPE', 'Pearson', 'SNR', 'MACC']
              if m in requested]

    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        for dim_label, rows in all_dim_rows.items():
            table_rows = []
            for key, n_subj, n_sess, n_win, m in rows:
                row = {
                    'Group':       key,
                    'Subjects':    n_subj,
                    'Sessions':    n_sess,
                    'Windows':     n_win,
                    'seed':        config.SEED,
                    'timestamp':   datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'eval_method': eval_method,
                }
                for metric in active:
                    if metric in m:
                        mean, se = m[metric]
                        row[f'{metric}_mean'] = round(mean, 4)
                        row[f'{metric}_se']   = round(se,   4)
                    else:
                        row[f'{metric}_mean'] = None
                        row[f'{metric}_se']   = None
                table_rows.append(row)
            df = pd.DataFrame(table_rows)
            df.to_excel(writer, sheet_name=dim_label[:31], index=False)

    print(f"[MR-NIRP] Condition breakdown saved → {xlsx_path}")


# ── Public entry point ────────────────────────────────────────────────────────

def calculate_metrics_by_condition(predictions, labels, config):
    """Compute and print MR-NIRP condition-based metric breakdown.

    Dimensions: Motion (still/small/large) and Wavelength (940nm/975nm).
    Garage sessions are automatically excluded.

    Args:
        predictions : dict  {filename → {chunk_id → tensor}}
        labels      : dict  {filename → {chunk_id → tensor}}
        config      : yacs CfgNode (frozen)
    """
    eval_method = config.INFERENCE.EVALUATION_METHOD
    print('\n' + '═' * 60)
    print(f'  MR-NIRP Condition Breakdown  [{eval_method}]')
    print('═' * 60)

    records = _collect_per_video_hr(predictions, labels, config)
    if not records:
        print('  No driving-session records to evaluate.')
        return

    DIMS = {
        'Motion':     'motion',
        'Wavelength': 'wavelength',
    }

    requested     = set(config.TEST.METRICS)
    all_dim_rows  = {}
    for dim_label, dim_key in DIMS.items():
        rows = _group_stats(records, dim_key, config)
        _print_condition_table(dim_label, rows, requested)
        all_dim_rows[dim_label] = rows

    if config.TOOLBOX_MODE == 'train_and_test':
        filename_id = config.TRAIN.MODEL_FILE_NAME
    else:
        model_root  = config.INFERENCE.MODEL_PATH.split('/')[-1].split('.pth')[0]
        filename_id = model_root + '_' + config.TEST.DATA.DATASET

    _save_condition_excel(all_dim_rows, filename_id, config, eval_method)
