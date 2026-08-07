import os
import re
import glob
import json
import csv
import cv2
import numpy as np
import pandas as pd
from dataset.data_loader.BaseLoader import BaseLoader

# Constants
VIDEO_PAT = "*.[mM][pP]4"
CSV_PATS = ["*.[cC][sS][vV]", "*.[tT][sS][vV]"]


# Utility Functions
def _to_int(s):
    """Extract integer from string for sorting purposes."""
    m = re.search(r'(\d+)', str(s))
    return int(m.group(1)) if m else 10**9


def _norm_task(val):
    """Normalize task value to 'driving', 'still', or None."""
    if val is None:
        return None
    s = str(val).strip().lower()
    return s if s in ("driving", "still") else None


def _extract_sid(name: str):
    """Extract session ID (Sxx format) from filename."""
    m = re.search(r'(S\d+)', name, flags=re.IGNORECASE)
    return m.group(1).upper() if m else None


def _glob_many(root, pat_list):
    """Apply multiple glob patterns and return sorted results."""
    out = []
    for p in pat_list:
        out += glob.glob(os.path.join(root, p))
    return sorted(out)


def _find_videos(root, sid):
    """Find video files for a given segment ID under a root directory."""
    subdir = os.path.join(root, sid)
    if os.path.isdir(subdir):
        return sorted(glob.glob(os.path.join(subdir, VIDEO_PAT)))
    return (sorted(glob.glob(os.path.join(root, f"{sid}{VIDEO_PAT[1:]}"))) or
            sorted(glob.glob(os.path.join(root, f"*{sid}*{VIDEO_PAT[1:]}"))))


def _signal_candidates(sig_root: str, sid: str, mode: str):
    """Find signal files based on mode (ECG for driving, PPG for still)."""
    want = "ECG" if mode.lower() == "driving" else "PPG"

    # Try folder structure first (Signal/Sxx/)
    folder = os.path.join(sig_root, sid)
    files = []

    if os.path.isdir(folder):
        files += _glob_many(folder, [f"{want}*.[cC][sS][vV]", f"{want}*.[tT][sS][vV]"])
        if not files:
            files += _glob_many(folder, [f"*{sid}*.[cC][sS][vV]", f"*{sid}*.[tT][sS][vV]"])

    # Try file structure (Signal/)
    if not files:
        files += _glob_many(sig_root, [
            f"{want}_{sid}*.[cC][sS][vV]", f"{sid}_{want}*.[cC][sS][vV]",
            f"{want}_{sid}*.[tT][sS][vV]", f"{sid}_{want}*.[tT][sS][vV]"
        ])

    if not files:
        files += _glob_many(sig_root, [
            f"*{sid}*{want}*.[cC][sS][vV]", f"*{sid}*{want}*.[tT][sS][vV]"
        ])

    if not files:
        files += _glob_many(sig_root, [
            f"{sid}*.[cC][sS][vV]", f"*{sid}*.[cC][sS][vV]",
            f"{sid}*.[tT][sS][vV]", f"*{sid}*.[tT][sS][vV]"
        ])

    # HR_Sxx.csv sits beside the waveform and matches the loose patterns above,
    # but it holds beats per minute, not a waveform.
    files = [f for f in files
             if not os.path.basename(f).lower().startswith(("hr_", "gt_hr"))]

    # Remove duplicates while preserving order
    ordered = list(dict.fromkeys(files))

    # Prioritize files containing the wanted signal type
    ordered.sort(key=lambda p: (0 if want.lower() in os.path.basename(p).lower() else 1, p))
    return ordered


def _load_fixed_affine(path_):
    """Load fixed affine transformation matrix from file."""
    ext = os.path.splitext(path_)[1].lower()

    if ext == ".npy":
        M = np.load(path_)
    elif ext == ".npz":
        with np.load(path_) as z:
            M = z["M"] if "M" in z else z[z.files[0]]
    elif ext == ".json":
        with open(path_, "r") as f:
            obj = json.load(f)
            M = obj["M"] if isinstance(obj, dict) and "M" in obj else obj
    else:
        raise ValueError(f"Unsupported fixed matrix file: {path_}")

    M = np.asarray(M, dtype=np.float32)
    if M.shape != (2, 3):
        raise ValueError(f"Fixed affine must be 2x3, got {M.shape}")

    return M


class MSDriveLoader(BaseLoader):
    """
    MMDrive dataset loader with support for RGB/NIR video pairs and physiological signals.

    Expected structure:
    <root> or <root>/RawData/
      Subjectxx/ or s01_M3/ (any Subject*/subject*/s[0-9]* prefix)/
        driving|still/
          RGB/(Sxx/*.mp4 or *Sxx*.mp4)
          NIR/(Sxx/*.mp4 or *Sxx*.mp4)
          Signal/(Sxx/*.csv|tsv or ECG/PPG prefix files)
    """

    def __init__(self, name, data_path, config_data, device=None):
        self.filtering = getattr(config_data, "FILTERING", None)

        # ── modality ──────────────────────────────────────────────────────────
        self.modality = getattr(config_data, 'MODALITY', 'MULTI_SPECTRAL').upper()
        if self.modality not in ('RGB', 'NIR', 'MULTI_SPECTRAL'):
            raise ValueError(
                f"MSDriveLoader: MODALITY must be 'RGB', 'NIR', or 'MULTI_SPECTRAL', "
                f"got '{self.modality}'."
            )

        # ── modality-specific cache paths ─────────────────────────────────────
        suffix = {'RGB': '_rgb', 'NIR': '_nir', 'MULTI_SPECTRAL': '_ms'}[self.modality]
        cfg = config_data.clone()
        cfg.defrost()
        cfg.CACHED_PATH = cfg.CACHED_PATH + suffix
        base_path, ext = os.path.splitext(cfg.FILE_LIST_PATH)
        cfg.FILE_LIST_PATH = base_path + suffix + (ext if ext else '')
        cfg.freeze()

        super().__init__(name, data_path, cfg)

    def get_raw_data(self, data_path):
        """Discover and collect all valid data records."""
        print(f"[INFO] Discovering data in: {data_path}")

        # Determine root directory (<path> or <path>/RawData)
        _subj_patterns = ["p[0-9]*", "Subject*", "subject*", "s[0-9]*"]
        cand_roots = [data_path, os.path.join(data_path, "RawData")]
        root = next((r for r in cand_roots if
                     any(glob.glob(os.path.join(r, p)) for p in _subj_patterns)), None)

        if root is None:
            raise ValueError(f"No Subject* directories found under {cand_roots}")

        print(f"[INFO] Using root directory: {root}")

        # Find all subject directories
        subjects = sorted(set(
            d for p in _subj_patterns
            for d in glob.glob(os.path.join(root, p))
            if os.path.isdir(d)
        ), key=lambda d: (_to_int(os.path.basename(d)), os.path.basename(d)))

        if not subjects:
            raise ValueError(f"No subject directories found under root={root}")

        print(f"[INFO] Found {len(subjects)} subject directories")

        # Get filtering parameters
        task = _norm_task(getattr(self.filtering, "TASK", None))
        modes = [task] if task else ["driving", "still"]

        seg_wl = set(getattr(self.filtering, "SEGMENT_LIST", []) or
                     getattr(self.filtering, "SEGMENT_WHITELIST", []))
        excl_on = bool(getattr(self.filtering, "USE_EXCLUSION_LIST", False))
        excl_set = set(x.lower() for x in getattr(self.filtering, "EXCLUSION_LIST", [])) if excl_on else set()

        print(f"[INFO] Processing modes: {modes}")
        if seg_wl:
            print(f"[INFO] Filtering segments: {sorted(seg_wl)}")
        if excl_on:
            print(f"[INFO] Excluding: {sorted(excl_set)}")

        recs = []

        for subj_dir in subjects:
            subj = os.path.basename(subj_dir)
            print(f"\n[INFO] Processing subject: {subj}")

            # The release drops the session level: a subject folder holds RGB/
            # NIR/Signal directly.  Fall back to that when no session dir exists.
            session_dirs = [(m, os.path.join(subj_dir, m)) for m in modes
                            if os.path.isdir(os.path.join(subj_dir, m))]
            if not session_dirs and os.path.isdir(os.path.join(subj_dir, "Signal")):
                session_dirs = [(modes[0], subj_dir)]

            for mode, mode_dir in session_dirs:
                print(f"  [INFO] Processing mode: {mode}")

                # Check required directories
                rgb_root = os.path.join(mode_dir, "RGB")
                nir_root = os.path.join(mode_dir, "NIR")
                sig_root = os.path.join(mode_dir, "Signal")

                if self.modality == 'RGB':
                    required = [("RGB", rgb_root), ("Signal", sig_root)]
                elif self.modality == 'NIR':
                    required = [("NIR", nir_root), ("Signal", sig_root)]
                else:  # MULTI_SPECTRAL
                    required = [("RGB", rgb_root), ("NIR", nir_root), ("Signal", sig_root)]

                missing_dirs = [n for n, p in required if not os.path.isdir(p)]
                if missing_dirs:
                    print(f"    [WARN] Missing directories: {missing_dirs}")
                    continue

                # Discover segment IDs from the primary video root
                primary_root = nir_root if self.modality == 'NIR' else rgb_root
                seg_ids = set()

                for d in sorted(glob.glob(os.path.join(primary_root, "S*"))):
                    if os.path.isdir(d):
                        sid = _extract_sid(os.path.basename(d))
                        if sid:
                            seg_ids.add(sid)

                for f in sorted(glob.glob(os.path.join(primary_root, VIDEO_PAT))):
                    sid = _extract_sid(os.path.splitext(os.path.basename(f))[0])
                    if sid:
                        seg_ids.add(sid)

                seg_ids = sorted(s for s in seg_ids if s)
                print(f"    [INFO] Found segments: {seg_ids}")

                for sid in seg_ids:
                    # Apply segment filtering
                    if seg_wl and sid not in seg_wl:
                        print(f"      [SKIP] Segment {sid} not in whitelist")
                        continue

                    print(f"    [INFO] Processing segment: {sid}")

                    # Find RGB files
                    rgb_files = []
                    if self.modality != 'NIR':
                        rgb_files = _find_videos(rgb_root, sid)
                        if not rgb_files:
                            print(f"      [WARN] No RGB files found for {sid}")
                            continue

                    # Find NIR files
                    nir_files = []
                    if self.modality != 'RGB':
                        nir_files = _find_videos(nir_root, sid)
                        if not nir_files:
                            print(f"      [WARN] No NIR files found for {sid}")
                            continue

                    # Find signal files (mode-specific ECG/PPG)
                    sig_files = _signal_candidates(sig_root, sid, mode)
                    if not sig_files:
                        print(f"      [WARN] No signal files found for {sid}")
                        continue

                    # Apply exclusion filtering
                    if excl_on:
                        keys = [subj, f"{subj}_{mode}", f"{subj}_{mode}_{sid}", sid]
                        if any(k and k.lower() in excl_set for k in keys):
                            print(f"      [SKIP] Segment {sid} in exclusion list")
                            continue

                    if self.modality != 'NIR':
                        print(f"      [SUCCESS] RGB: {os.path.basename(rgb_files[0])}")
                    if self.modality != 'RGB':
                        print(f"      [SUCCESS] NIR: {os.path.basename(nir_files[0])}")
                    print(f"      [SUCCESS] Signal: {os.path.basename(sig_files[0])}")

                    rec = {"subject": subj, "mode": mode, "session": sid, "signal": sig_files[0]}
                    if self.modality != 'NIR':
                        rec["rgb"] = rgb_files[0]
                    if self.modality != 'RGB':
                        rec["nir"] = nir_files[0]
                    recs.append(rec)

        if not recs:
            raise ValueError(f"No valid records found! Check TASK/SEGMENT_LIST/EXCLUSION_LIST and directory structure.")

        # Sort records
        recs.sort(key=lambda r: (_to_int(r["subject"]), r["mode"], _to_int(r["session"])))

        print(f"\n[INFO] Total records collected: {len(recs)}")
        return recs

    def split_raw_data(self, data_dirs, begin, end):
        """Split data for train/val/test."""
        if begin == 0 and end == 1:
            return data_dirs
        n = len(data_dirs)
        return [data_dirs[i] for i in range(int(begin * n), int(end * n))]

    # I/O utility methods
    @staticmethod
    def read_video(video_file, to_gray=False):
        """Read video file and return frames as numpy array."""
        cap = cv2.VideoCapture(video_file)
        frames = []

        ret, frame = cap.read()
        while ret:
            if to_gray:
                # BGR -> GRAY -> (H,W,1)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[..., None]
            else:
                # BGR -> RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frames.append(frame)
            ret, frame = cap.read()

        cap.release()
        return np.asarray(frames)

    @staticmethod
    def read_wave_csv(signal_file):
        """Read physiological signal from CSV/TSV file."""
        try:
            df = pd.read_csv(signal_file)
            if df.empty:
                return np.zeros((0,), dtype=np.float32)

            # Preferred column names based on file type
            prefer = {"ecg", "ppg", "bvp", "value", "signal"}
            base = os.path.basename(signal_file).lower()

            if "ecg" in base:
                prefer = {"ecg"} | prefer
            if "ppg" in base:
                prefer = {"ppg"} | prefer

            # Map columns to lowercase for matching
            lower = {c: str(c).strip().lower() for c in df.columns}
            prefer_cols = [c for c in df.columns if lower[c] in prefer]
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

            # Exclude time-related columns
            time_like = {"time", "timestamp", "ts", "t", "ms", "sec", "seconds"}
            prefer_cols = [c for c in prefer_cols if lower[c] not in time_like]
            num_cols = [c for c in num_cols if lower[c] not in time_like]

            # Select best column
            if prefer_cols:
                col = prefer_cols[0]
            elif len(num_cols) == 1:
                col = num_cols[0]
            elif num_cols:
                col = num_cols[-1]
            else:
                col = df.columns[0]

            vals = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=np.float32)
            return vals

        except Exception:
            # Fallback: try reading as simple CSV
            vals = []
            with open(signal_file, "r") as f:
                for row in csv.reader(f):
                    try:
                        vals.append(float(row[0]))
                    except:
                        pass
            return np.asarray(vals, dtype=np.float32)

    def _custom_align_nir_to_rgb(self, rgb_frames, nir_frames, backend, fixed_M=None):
        """
        Custom alignment method to fix BaseLoader bug.
        Aligns NIR frames to RGB frames using affine transformation.
        """
        N, H, W = nir_frames.shape
        out = np.empty_like(nir_frames)  # Preserve original dtype/shape

        # Common warp options for reproducibility and stability
        dsize = (W, H)
        interp = cv2.INTER_LINEAR
        border = cv2.BORDER_CONSTANT
        border_val = 0

        if fixed_M is not None:
            M = np.asarray(fixed_M, dtype=np.float32)
        else:
            # Fallback: identity transformation
            print("[WARN] No fixed affine matrix, using identity transformation")
            M = np.eye(2, 3, dtype=np.float32)

        for t in range(N):
            warped = cv2.warpAffine(
                nir_frames[t], M, dsize,
                flags=interp, borderMode=border, borderValue=border_val
            )
            # Ensure 2D output
            if warped.ndim == 3 and warped.shape[-1] == 1:
                warped = warped[..., 0]
            out[t] = warped

        return out

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """Preprocess a single data record.

        RGB:            loads RGB + signal only; all NIR processing is skipped.
        MULTI_SPECTRAL: loads RGB + NIR + signal, aligns NIR, saves both.
        """
        rec = data_dirs[i]
        save_name = f"{rec['subject']}_{rec['mode']}_{rec['session']}"

        print(f"\n[PROCESS {i+1}/{len(data_dirs)}] {save_name}")

        # 1) Read signal (always required)
        print(f"  [LOAD] Signal: {rec['signal']}")
        sig = self.read_wave_csv(rec["signal"])
        print(f"         Length: {len(sig)} samples")

        # 2) Read RGB (RGB or MULTI_SPECTRAL)
        rgb = None
        if self.modality != 'NIR':
            print(f"  [LOAD] RGB: {rec['rgb']}")
            rgb = self.read_video(rec["rgb"], to_gray=False)  # (N,H,W,3)
            print(f"         Shape: {rgb.shape}")
            print("  [PROCESS] Rotating RGB 90° counterclockwise...")
            rgb = np.array([cv2.rotate(f, cv2.ROTATE_90_COUNTERCLOCKWISE) for f in rgb])
            print(f"            RGB after rotation: {rgb.shape}")

        # 3) Read NIR (NIR or MULTI_SPECTRAL)
        nir = None
        if self.modality != 'RGB':
            print(f"  [LOAD] NIR: {rec['nir']}")
            nir = self.read_video(rec["nir"], to_gray=True)   # (N,H,W,1)
            print(f"         Shape: {nir.shape}")
            print("  [PROCESS] Rotating NIR 90° counterclockwise...")
            nir = np.array([cv2.rotate(f, cv2.ROTATE_90_COUNTERCLOCKWISE) for f in nir])
            print(f"            NIR after rotation: {nir.shape}")

        # 4) (Optional) NIR alignment
        def _align_nir(nir, rgb_ref):
            crop_face_cfg = getattr(config_preprocess, "CROP_FACE", None)
            use_align = bool(getattr(crop_face_cfg, "FACE_ALIGNMENT", False)) if crop_face_cfg is not None else False
            if not use_align:
                return nir
            backend = getattr(crop_face_cfg, "BACKEND", "HC") if crop_face_cfg is not None else "HC"
            fixed_M_path = getattr(crop_face_cfg, "FIXED_MATRIX_PATH", None) if crop_face_cfg is not None else None
            if not fixed_M_path:
                fixed_M_path = ".logs/global_affine_MMdrive.npy"
            print(f"  [ALIGN] backend={backend}, matrix={fixed_M_path}")
            fixed_M = None
            try:
                if os.path.exists(fixed_M_path):
                    fixed_M = _load_fixed_affine(fixed_M_path)
                    print(f"          Loaded fixed affine: {fixed_M.shape}")
                else:
                    print("          Fixed affine not found, using fallback")
            except Exception as e:
                print(f"          Error loading fixed affine: {e}")
            # Flatten to 2D (N,H,W) for warpAffine
            nir_2d = nir[..., 0] if nir.ndim != 3 else nir
            # rgb_ref is used only as a spatial reference by BaseLoader; not needed here
            nir_aligned = self._custom_align_nir_to_rgb(rgb_ref, nir_2d, backend, fixed_M)
            return nir_aligned[..., None]  # (N,H,W,1)

        def _preprocess_nir(nir, sig):
            if nir.ndim == 4 and nir.shape[-1] == 1:
                nir_rgb = np.repeat(nir, 3, axis=-1)
            elif nir.ndim == 4 and nir.shape[-1] == 3:
                nir_rgb = nir
            else:
                nir_rgb = np.repeat(nir[..., None], 3, axis=-1)
            print("  [PREPROCESS] Processing NIR...")
            # NIR은 항상 Raw(원본 이미지)만 사용 (RGB는 config_preprocess.DATA_TYPE 그대로 적용)
            nir_config = config_preprocess.clone()
            nir_config.defrost()
            nir_config.DATA_TYPE = ["Raw"]
            nir_config.freeze()
            nir_clips_rgb, sig_clips = self.preprocess(nir_rgb, sig, nir_config)
            nir_clips = nir_clips_rgb[..., :1]   # keep 1ch
            print(f"               Generated {len(nir_clips)} NIR clips")
            return nir_clips, sig_clips

        if self.modality == 'RGB':
            # ── RGB-only ────────────────────────────────────────────────────
            L = min(len(rgb), len(sig))
            if L == 0:
                print("  [ERROR] No data after length matching"); file_list_dict[i] = []; return
            rgb, sig = rgb[:L], sig[:L]
            print(f"  [SYNC] Matched length: {L}")
            print("  [PREPROCESS] Processing RGB...")
            rgb_clips, sig_clips = self.preprocess(rgb, sig, config_preprocess)
            print(f"               Generated {len(rgb_clips)} RGB clips")
            print("  [SAVE] Saving...")
            inputs, labels = self.save_multi_process(rgb_clips, None, sig_clips, save_name)

        elif self.modality == 'NIR':
            # ── NIR-only ────────────────────────────────────────────────────
            # Alignment can be applied without RGB reference (fixed_M only)
            nir = _align_nir(nir, nir)   # rgb_ref unused by _custom_align_nir_to_rgb
            L = min(len(nir), len(sig))
            if L == 0:
                print("  [ERROR] No data after length matching"); file_list_dict[i] = []; return
            nir, sig = nir[:L], sig[:L]
            print(f"  [SYNC] Matched length: {L}")
            nir_clips, sig_clips = _preprocess_nir(nir, sig)
            print("  [SAVE] Saving...")
            inputs, labels = self.save_multi_process(None, nir_clips, sig_clips, save_name)

        else:
            # ── MULTI_SPECTRAL ───────────────────────────────────────────────
            nir = _align_nir(nir, rgb)
            L = min(len(rgb), len(nir), len(sig))
            if L == 0:
                print("  [ERROR] No data after length matching"); file_list_dict[i] = []; return
            rgb, nir, sig = rgb[:L], nir[:L], sig[:L]
            print(f"  [SYNC] Matched length: {L}")
            print("  [PREPROCESS] Processing RGB...")
            rgb_clips, sig_clips = self.preprocess(rgb, sig, config_preprocess)
            print(f"               Generated {len(rgb_clips)} RGB clips")
            nir_clips, _ = _preprocess_nir(nir, sig)
            print("  [SAVE] Saving...")
            inputs, labels = self.save_multi_process(rgb_clips, nir_clips, sig_clips, save_name)

        print(f"         Saved {len(inputs)} input files")
        file_list_dict[i] = inputs

    def save_multi_process(self, rgb_clips, nir_clips, bvp_clips, filename):
        """Save preprocessed clips to disk.

        Args:
            rgb_clips: array of RGB clip data.
            nir_clips: array of NIR clip data, or ``None`` for RGB-only mode.
            bvp_clips: array of BVP label data.
            filename:  base filename for saved files.
        """
        os.makedirs(self.cached_path, exist_ok=True)

        input_path_name_list, label_path_name_list = [], []
        for k, bvp_clip in enumerate(bvp_clips):
            in_path = os.path.join(self.cached_path, f"{filename}_input{k}.npz")
            lb_path = os.path.join(self.cached_path, f"{filename}_label{k}.npy")

            save_dict = {}
            if rgb_clips is not None:
                save_dict['rgb'] = rgb_clips[k]
            if nir_clips is not None:
                save_dict['nir'] = nir_clips[k]
            np.savez(in_path, **save_dict)
            np.save(lb_path, bvp_clip)

            input_path_name_list.append(in_path)
            label_path_name_list.append(lb_path)

        return input_path_name_list, label_path_name_list

    def build_file_list_retroactive(self, data_dirs, begin, end):
        """Override BaseLoader's version: MSDriveLoader saves .npz inputs and
        uses subject/mode/session keys instead of 'index'."""
        data_dirs_subset = self.split_raw_data(data_dirs, begin, end)

        file_list = []
        for rec in data_dirs_subset:
            fname = f"{rec['subject']}_{rec['mode']}_{rec['session']}"
            found = sorted(glob.glob(
                os.path.join(self.cached_path, f"{fname}_input*.npz")
            ))
            file_list.extend(found)

        if not file_list:
            raise ValueError(
                self.dataset_name,
                'File list empty. Check preprocessed data folder exists and is not empty. '
                f'Searched in: {self.cached_path}'
            )

        file_list_df = pd.DataFrame(file_list, columns=['input_files'])
        os.makedirs(os.path.dirname(self.file_list_path), exist_ok=True)
        file_list_df.to_csv(self.file_list_path)

    def load_preprocessed_data(self):
        """Load preprocessed data with filtering."""
        super().load_preprocessed_data()

        self.labels = [
            p.replace("_input", "_label").rsplit(".npz", 1)[0] + ".npy"
            for p in self.inputs
        ]

        if self.filtering is not None:
            print("[INFO] Applying filters to preprocessed data...")

            seg_wl = set(getattr(self.filtering, "SEGMENT_LIST", []) or
                        getattr(self.filtering, "SEGMENT_WHITELIST", []))
            excl_on = bool(getattr(self.filtering, "USE_EXCLUSION_LIST", False))
            excl_set = set(x.lower() for x in getattr(self.filtering, "EXCLUSION_LIST", [])) if excl_on else set()
            task = _norm_task(getattr(self.filtering, "TASK", None))

            filtered_inputs, filtered_labels = [], []

            for inp, lab in zip(self.inputs, self.labels):
                # Parse filename: supports any subject name (Subject01, s01_M3, etc.)
                m = re.match(
                    r"(.+?)_(driving|still)_(S\d+)_input\d+\.npz",
                    os.path.basename(inp),
                    re.IGNORECASE
                )

                if m:
                    subj, mode, sid = m.group(1), m.group(2).lower(), m.group(3).upper()
                else:
                    subj, mode, sid = None, None, None

                # Apply filters
                if seg_wl and sid not in seg_wl:
                    continue

                if excl_on:
                    keys = [subj, f"{subj}_{mode}", f"{subj}_{mode}_{sid}", sid]
                    if any(k and k.lower() in excl_set for k in keys):
                        continue

                if task and mode != task:
                    continue

                filtered_inputs.append(inp)
                filtered_labels.append(lab)

            self.inputs, self.labels = filtered_inputs, filtered_labels
            print(f"[INFO] After filtering: {len(self.inputs)} samples")

        self.preprocessed_data_len = len(self.inputs)

    def __getitem__(self, index):
        """Get a single data sample.

        MULTI_SPECTRAL: returns ``[rgb, nir], label, filename, chunk_id``
        RGB:            returns ``rgb, label, filename, chunk_id``
        """
        input_path = self.inputs[index]

        with np.load(input_path) as npz_file:
            if self.modality != 'NIR':
                rgb = npz_file["rgb"]
            if self.modality != 'RGB':
                nir = npz_file["nir"]

        label = np.load(self.labels[index]).astype(np.float32)

        # Format conversion
        if self.data_format == 'NDCHW':
            if self.modality != 'NIR':
                rgb = np.transpose(rgb, (0, 3, 1, 2))
            if self.modality != 'RGB':
                nir = np.transpose(nir, (0, 3, 1, 2))
        elif self.data_format == 'NCDHW':
            if self.modality != 'NIR':
                rgb = np.transpose(rgb, (3, 0, 1, 2))
            if self.modality != 'RGB':
                nir = np.transpose(nir, (3, 0, 1, 2))
        elif self.data_format == 'NDHWC':
            pass
        else:
            raise ValueError('Unsupported Data Format!')

        # Parse filename and chunk ID
        stem, _ = os.path.splitext(os.path.basename(input_path))
        filename, _, chunk_id = stem.rpartition('_input')

        if self.modality == 'MULTI_SPECTRAL':
            return [rgb.astype(np.float32), nir.astype(np.float32)], label, filename, chunk_id
        elif self.modality == 'NIR':
            return nir.astype(np.float32), label, filename, chunk_id
        else:  # RGB
            return rgb.astype(np.float32), label, filename, chunk_id
