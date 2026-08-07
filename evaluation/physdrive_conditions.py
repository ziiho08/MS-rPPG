"""
PhysDrive condition-based metric breakdown.

Parses PhysDrive filenames (e.g. ``AFH1_AS``) into condition dimensions and
reports HR metrics grouped by each dimension.  Called after the standard
``calculate_metrics()`` so the overall result is always printed first.

Condition dimensions
--------------------
Subject ID  ``[Vehicle][Sex][Illumination][Index]``
Segment     ``[Road][Motion]``

Output columns per group
------------------------
  Sessions  – number of unique session keys in predictions
  Windows   – number of evaluation windows (same unit as calculate_metrics N)
  MAE / RMSE / MAPE / Pearson / SNR / MACC  (whichever are in TEST.METRICS)
"""

import os
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

from evaluation.post_process import calculate_metric_per_video

# ── Naming-convention maps ────────────────────────────────────────────────────
VEHICLE_MAP     = {'A': 'A0-Seg',  'B': 'B-Seg',  'C': 'SUV'}
SEX_MAP         = {'M': 'Male',    'F': 'Female'}
ILLUMINATION_MAP= {'Z': 'Noon',    'H': 'Dawn_Dusk', 'W': 'Night', 'Y': 'Rainy'}
ROAD_MAP        = {'A': 'Flat_Open', 'B': 'Flat_Congested', 'C': 'Bumpy'}
MOTION_MAP      = {'S': 'Stationary', 'T': 'Talking'}

# Ordered display for each dimension (preserves logical order in output)
DIM_ORDER = {
    'illumination': ['Noon', 'Dawn_Dusk', 'Night', 'Rainy'],
    'motion':       ['Stationary', 'Talking'],
    'road':         ['Flat_Open', 'Flat_Congested', 'Bumpy'],
    'sex':          ['Male', 'Female'],
    'vehicle':      ['A0-Seg', 'B-Seg', 'SUV'],
}


def parse_physdrive_filename(filename: str) -> dict:
    """Parse ``AFH1_AS`` → condition dict.

    Returns a dict with keys ``vehicle``, ``sex``, ``illumination``, ``index``,
    ``road``, ``motion``.  Returns ``None`` if the filename cannot be parsed.
    """
    try:
        parts = filename.split('_')
        subj, seg = parts[0], parts[1]
        return {
            'vehicle':      VEHICLE_MAP[subj[0]],
            'sex':          SEX_MAP[subj[1]],
            'illumination': ILLUMINATION_MAP[subj[2]],
            'index':        int(subj[3]),
            'road':         ROAD_MAP[seg[0]],
            'motion':       MOTION_MAP[seg[1]],
        }
    except Exception:
        return None


# ── Per-video HR collection ───────────────────────────────────────────────────

def _collect_per_video_hr(predictions, labels, config):
    """Replicate the inner loop of calculate_metrics() per session.

    Returns
    -------
    records : list of dict
        One entry per evaluation window, each with keys:
        ``filename``, ``gt_hr``, ``pred_hr``, ``snr``, ``macc``,
        plus all condition keys from ``parse_physdrive_filename``.
    """
    import torch

    def _reform(data):
        sort_data = sorted(data.items(), key=lambda x: x[0])
        sort_data = torch.cat([v for _, v in sort_data], dim=0)
        return np.reshape(sort_data.cpu().numpy(), (-1))

    diff_flag = config.TEST.DATA.PREPROCESS.LABEL_TYPE == 'DiffNormalized'
    eval_method = config.INFERENCE.EVALUATION_METHOD
    fs = config.TEST.DATA.FS

    use_window = config.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW
    win_sec    = config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE

    records = []
    for filename in predictions:
        prediction = _reform(predictions[filename])
        label      = _reform(labels[filename])

        video_len = prediction.shape[0]
        win_frames = int(win_sec * fs) if use_window else video_len
        win_frames = min(win_frames, video_len)

        cond = parse_physdrive_filename(filename)

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
                'filename': filename,
                'gt_hr':    gt_hr,
                'pred_hr':  pred_hr,
                'snr':      snr,
                'macc':     macc,
            }
            if cond is not None:
                rec.update(cond)
            else:
                for k in ('vehicle', 'sex', 'illumination', 'road', 'motion'):
                    rec[k] = 'Unknown'
            records.append(rec)

    return records


# ── Stats helper ──────────────────────────────────────────────────────────────

def _stats(arr):
    """Return (mean, se) for a 1-D array."""
    arr = np.asarray(arr, dtype=float)
    n   = len(arr)
    if n == 0:
        return float('nan'), float('nan')
    return float(np.mean(arr)), float(np.std(arr) / np.sqrt(n))


def _group_stats(records, dim, config):
    """
    Group records by ``dim`` and compute metric stats.

    Returns list of (group_label, n_sessions, n_windows, metrics_dict)
    where metrics_dict maps metric_name → (mean, se).
    """
    requested = set(config.TEST.METRICS)
    eval_method = config.INFERENCE.EVALUATION_METHOD

    # Gather windows per group
    groups = defaultdict(list)
    for rec in records:
        groups[rec[dim]].append(rec)

    # Unique sessions and subjects per group
    session_counts = defaultdict(set)
    subject_counts = defaultdict(set)
    for rec in records:
        session_counts[rec[dim]].add(rec['filename'])
        subject_counts[rec[dim]].add(rec['filename'].split('_')[0])

    ordered_keys = DIM_ORDER.get(dim, sorted(groups.keys()))
    rows = []
    for key in ordered_keys:
        if key not in groups:
            continue
        g = groups[key]
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

    col_w   = 18
    met_w   = 9
    header  = f"  {'Group':<{col_w}}  {'Subjects':>8}  {'Sessions':>8}  {'Windows':>8}"
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
    """Append condition breakdown sheets to the existing CSV or write a new file."""
    save_dir = config.TEST.EXCEL_SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)
    xlsx_path = os.path.join(save_dir, f'{filename_id}_conditions.xlsx')

    requested = set(config.TEST.METRICS)
    active = [m for m in ['MAE', 'RMSE', 'MAPE', 'Pearson', 'SNR', 'MACC']
              if m in requested]

    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        for dim_label, rows in all_dim_rows.items():
            table_rows = []
            for key, n_subj, n_sess, n_win, m in rows:
                row = {
                    'Group':    key,
                    'Subjects': n_subj,
                    'Sessions': n_sess,
                    'Windows':  n_win,
                    'seed':     config.SEED,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
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
            sheet = dim_label[:31]   # Excel sheet name max 31 chars
            df.to_excel(writer, sheet_name=sheet, index=False)

    print(f"[PhysDrive] Condition breakdown saved → {xlsx_path}")


# ── Public entry point ────────────────────────────────────────────────────────

def calculate_metrics_by_condition(predictions, labels, config):
    """Compute and print PhysDrive condition-based metric breakdown.

    Called after ``calculate_metrics()`` in the trainer's ``test()`` method.

    Args:
        predictions : dict  {filename → {chunk_id → tensor}}
        labels      : dict  {filename → {chunk_id → tensor}}
        config      : yacs CfgNode (frozen)
    """
    eval_method = config.INFERENCE.EVALUATION_METHOD
    print('\n' + '═' * 66)
    print(f'  PhysDrive Condition Breakdown  [{eval_method}]')
    print('═' * 66)

    records = _collect_per_video_hr(predictions, labels, config)
    if not records:
        print('  No records to evaluate.')
        return

    DIMS = {
        'Illumination': 'illumination',
        'Motion':       'motion',
        'Road':         'road',
        'Sex':          'sex',
        'Vehicle':      'vehicle',
    }

    requested = set(config.TEST.METRICS)
    all_dim_rows = {}
    for dim_label, dim_key in DIMS.items():
        rows = _group_stats(records, dim_key, config)
        _print_condition_table(dim_label, rows, requested)
        all_dim_rows[dim_label] = rows

    # Build filename_id (same logic as calculate_metrics)
    if config.TOOLBOX_MODE == 'train_and_test':
        filename_id = config.TRAIN.MODEL_FILE_NAME
    else:
        model_root  = config.INFERENCE.MODEL_PATH.split('/')[-1].split('.pth')[0]
        filename_id = model_root + '_' + config.TEST.DATA.DATASET

    _save_condition_excel(all_dim_rows, filename_id, config, eval_method)
