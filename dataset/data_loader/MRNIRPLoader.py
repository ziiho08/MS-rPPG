import os
import glob
import io
import json
import zipfile

import cv2
import imageio
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.io import loadmat

from dataset.data_loader.BaseLoader import BaseLoader


class MRNIRPLoader(BaseLoader):
    """
    Data loader for the MR-NIRP processed dataset (RGB + NIR + PulseOX).

    Supports two modalities via YAML config key ``MODALITY``:
      - ``'MULTI_SPECTRAL'`` (default): loads and returns both RGB and NIR.
        ``__getitem__`` returns ``(rgb, nir), label, filename, chunk_id``.
      - ``'RGB'``: loads and returns RGB only; all NIR processing is skipped.
        ``__getitem__`` returns ``rgb, label, filename, chunk_id``.

    The cached path is automatically suffixed with ``_ms`` or ``_rgb`` so that
    datasets preprocessed in different modalities are kept separate.

    Dataset structure (example):

        RawData/
          └── subject1/
                ├── subject1_driving_small_motion_975/
                │     ├── RGB/ or RGB.zip
                │     ├── NIR/ or NIR.zip   (only used for MULTI_SPECTRAL)
                │     └── PulseOX/ or PulseOx.zip
                └── ...

    Args:
        name (str): name of the dataloader.
        data_path (str): path to "RawData".
        config_data (CfgNode): configuration (FS, PREPROCESS, FILTERING, etc.)
            Must include ``MODALITY`` key (``'RGB'`` or ``'MULTI_SPECTRAL'``).
    """
    def __init__(self, name, data_path, config_data, device=None):
        self.filtering = config_data.FILTERING

        # ── modality ──────────────────────────────────────────────────────────
        self.modality = getattr(config_data, 'MODALITY', 'MULTI_SPECTRAL').upper()
        if self.modality not in ('RGB', 'NIR', 'MULTI_SPECTRAL'):
            raise ValueError(
                f"MRNIRPLoader: MODALITY must be 'RGB', 'NIR', or 'MULTI_SPECTRAL', "
                f"got '{self.modality}'."
            )

        # ── modality-specific cache paths ─────────────────────────────────────
        # When MODALITY is RGB or NIR, reuse the MULTI_SPECTRAL (_ms) cache if it
        # already exists — __getitem__ will slice only the needed key from the npz.
        # This avoids duplicate preprocessing.  Falls back to the modality-specific
        # suffix when no _ms cache is found (or when MODALITY == MULTI_SPECTRAL).
        _ms_cached = config_data.CACHED_PATH + '_ms'
        if self.modality in ('RGB', 'NIR') and os.path.isdir(_ms_cached):
            suffix = '_ms'
            print(f"[MRNIRPLoader] MODALITY='{self.modality}': MS cache found → slicing from {_ms_cached}")
        else:
            suffix = {'RGB': '_rgb', 'NIR': '_nir', 'MULTI_SPECTRAL': '_ms'}[self.modality]
        cfg = config_data.clone()
        cfg.defrost()
        cfg.CACHED_PATH = cfg.CACHED_PATH + suffix
        base, ext = os.path.splitext(cfg.FILE_LIST_PATH)
        cfg.FILE_LIST_PATH = base + suffix + (ext if ext else '')
        cfg.freeze()

        super().__init__(name, data_path, cfg)

    # -------------------------------------------------------------------------
    # 1) Raw directory listing & splitting
    # -------------------------------------------------------------------------
    def get_raw_data(self, data_path):
        """Return a list of subject-task directories."""
        data_dirs = glob.glob(os.path.join(data_path, "subject*", "subject*"))
        if not data_dirs:
            raise ValueError("MR-NIRP dataset paths empty!")
        return [{"index": os.path.basename(p), "path": p} for p in data_dirs]

    def split_raw_data(self, data_dirs, begin, end):
        """Return subset of data dirs between [begin, end)."""
        if begin == 0 and end == 1:
            return data_dirs
        n = len(data_dirs)
        return [data_dirs[i] for i in range(int(begin * n), int(end * n))]

    # -------------------------------------------------------------------------
    # 2) Load preprocessed file list (cached .npz / .npy)
    # -------------------------------------------------------------------------
    @staticmethod
    def _filter_identifiers(inp_name, subject_name, task, task_full, wavelength):
        """Return the set of identifiers that can be matched in allow/deny lists."""
        return {inp_name, subject_name, task, task_full, wavelength, f"{task}_{wavelength}"}

    def load_preprocessed_data(self):
        """Load and filter cached preprocessed data from CSV list."""
        file_list_df = pd.read_csv(self.file_list_path)
        base_inputs = file_list_df["input_files"].tolist()

        filtered_inputs = []
        for inp in base_inputs:
            inp_name = os.path.basename(inp).split(".")[0].rsplit("_", 1)[0]
            subject_name = inp_name.rsplit("_", 1)[0].split("_")[0]  # e.g. subject1
            task_full = inp_name.rsplit("_", 1)[0]                   # e.g. subject1_driving_small_motion
            task = task_full.split("_", 1)[1] if "_" in task_full else task_full
            wavelength = inp_name.split("_")[-1]                     # '940' or '975'

            identifiers = self._filter_identifiers(inp_name, subject_name, task, task_full, wavelength)

            # allow-list
            if self.filtering.SELECT_TASKS:
                task_list = set(getattr(self.filtering, "TASK_LIST", []))
                if not (identifiers & task_list):
                    continue

            # deny-list
            if self.filtering.USE_EXCLUSION_LIST:
                excl = set(self.filtering.EXCLUSION_LIST)
                if identifiers & excl:
                    continue

            filtered_inputs.append(inp)

        if not filtered_inputs:
            raise ValueError(f"{self.dataset_name} dataset loading data error: no inputs after filtering.")

        filtered_inputs = sorted(filtered_inputs)
        labels = [
            inp.replace("_input", "_label").rsplit(".npz", 1)[0] + ".npy"
            for inp in filtered_inputs
        ]

        self.inputs = filtered_inputs
        self.labels = labels
        self.preprocessed_data_len = len(filtered_inputs)

    # -------------------------------------------------------------------------
    # 3) Low-level readers (zipped / unzipped)
    # -------------------------------------------------------------------------
    @staticmethod
    def read_video(video_zip_path, resize_dim=144):
        """
        Read frames from a ZIP of .pgm files.
        Returns: np.ndarray, shape (T, H, W, 3), dtype=uint8
        """
        frames = []
        with zipfile.ZipFile(video_zip_path, "r") as zf:
            for name in sorted(zf.namelist()):
                if not name.lower().endswith(".pgm"):
                    continue
                frame = np.array(imageio.imread(io.BytesIO(zf.read(name))), dtype=np.uint16)
                frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2RGB)
                frame = (frame >> 8).astype(np.uint8)
                if resize_dim is not None:
                    w = min(resize_dim, frame.shape[1])
                    h = int(w * frame.shape[0] / frame.shape[1])
                    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
                frames.append(frame)

        if not frames:
            raise ValueError("EMPTY VIDEO: " + video_zip_path)
        return np.asarray(frames)

    @staticmethod
    def read_video_unzipped(dir_path, to_gray=False):
        """
        Read frames from an unzipped directory containing Frame*.pgm
        Returns: (T, H, W, C)
        """
        frames = []
        for pgm_path in sorted(glob.glob(os.path.join(dir_path, "Frame*.pgm"))):
            frame = cv2.imread(pgm_path, cv2.IMREAD_UNCHANGED)
            if frame is None:
                print("Error in reading frame:", pgm_path)
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2RGB)
            frame = (frame >> 8).astype(np.uint8)

            if to_gray:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                frame = frame[..., np.newaxis]

            frames.append(frame)

        return np.asarray(frames, dtype=np.uint8)

    @staticmethod
    def read_wave_unzipped(dir_path):
        """Read PulseOX/pulseOx.mat from unzipped directory."""
        raw = loadmat(os.path.join(dir_path, "pulseOx.mat"))
        ts = raw["pulseOxTime"][0]
        ts = ts - ts[0]
        ppg = raw["pulseOxRecord"][0]
        return ppg, ts

    @staticmethod
    def read_wave(zip_path):
        """Read PulseOX/pulseOx.mat from zip file."""
        with zipfile.ZipFile(zip_path, "r") as zf:
            mat = loadmat(zf.open("PulseOX/pulseOx.mat"))
        ppg = mat["pulseOxRecord"][0]
        ts = mat["pulseOxTime"][0]
        ts = ts - ts[0]
        return ppg, ts

    # -------------------------------------------------------------------------
    # 4) Signal utilities
    # -------------------------------------------------------------------------
    @staticmethod
    def correct_irregular_sampling(ppg, timestamps, target_fs=30):
        """
        Linearly re-sample an irregularly sampled PPG to target_fs.
        Assumes timestamps are strictly increasing.
        """
        resampled = []
        duration = timestamps[-1]

        for t in np.arange(0.0, duration, 1.0 / target_fs):
            stop_idx = np.argmax(timestamps - t > 0)  # first index where ts > t

            if stop_idx == 0:
                # t is before the first timestamp, or past the last (all-False argmax → 0)
                resampled.append(ppg[0] if timestamps[0] >= t else ppg[-1])
                continue

            start_idx = stop_idx - 1
            span = timestamps[stop_idx] - timestamps[start_idx]
            w = 0.0 if span == 0 else (t - timestamps[start_idx]) / span
            resampled.append(ppg[start_idx] * (1 - w) + ppg[stop_idx] * w)

        return np.array(resampled, dtype=np.float32)

    @staticmethod
    def match_length(*arrays):
        """
        Truncate all arrays to the same (minimum) length.
        Example: ppg, rgb, nir = match_length(ppg, rgb, nir)
        """
        min_len = min(len(a) for a in arrays)
        return tuple(a[:min_len] for a in arrays)

    # -------------------------------------------------------------------------
    # 5) Fixed affine matrix loading
    # -------------------------------------------------------------------------
    @staticmethod
    def _load_fixed_affine(path):
        """Load a 2x3 affine matrix from .npy, .npz, or .json file."""
        ext = os.path.splitext(path)[1].lower()
        if ext == ".npy":
            M = np.load(path)
        elif ext == ".npz":
            with np.load(path) as z:
                M = z["M"] if "M" in z else z[z.files[0]]
        elif ext == ".json":
            with open(path, "r") as f:
                M = np.array(json.load(f)["M"], dtype=np.float32)
        else:
            raise ValueError(f"Unsupported fixed matrix file: {path}")
        M = np.asarray(M, dtype=np.float32)
        if M.shape != (2, 3):
            raise ValueError(f"Fixed affine must be 2x3, got {M.shape}")
        return M

    # -------------------------------------------------------------------------
    # 6) Preprocess (multiprocess child)
    # -------------------------------------------------------------------------
    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """
        Invoked by preprocess_dataset (multiprocessing).

        MULTI_SPECTRAL: loads RGB + NIR + PPG, aligns NIR to RGB, saves both.
        RGB:            loads RGB + PPG only; NIR is entirely skipped.
        """
        # skip known-bad sample
        if data_dirs[i]["index"] == "subject2_garage_small_motion_940":
            return

        session_path = data_dirs[i]["path"]

        # --- load RGB video ---
        rgb_frames = self.read_video_unzipped(os.path.join(session_path, "RGB"))

        # --- load PPG ---
        ppg, timestamps = self.read_wave_unzipped(os.path.join(session_path, "PulseOX"))
        ppg = self.correct_irregular_sampling(ppg, timestamps, target_fs=self.config_data.FS)

        def _load_nir_frames():
            """Load, subsample, and optionally align NIR frames (uses rgb_frames as reference)."""
            nir = self.read_video_unzipped(os.path.join(session_path, "NIR"), to_gray=True)
            nir = nir[::2]
            crop_face_cfg = getattr(config_preprocess, "CROP_FACE", None)
            use_align = bool(getattr(crop_face_cfg, "FACE_ALIGNMENT", False)) if crop_face_cfg else False
            if use_align:
                backend = getattr(crop_face_cfg, "BACKEND", "HC") if crop_face_cfg else "HC"
                fixed_M_path = getattr(crop_face_cfg, "FIXED_MATRIX_PATH", ".logs/global_affine_MRNIRP.npy")
                fixed_M = None
                if os.path.exists(fixed_M_path):
                    try:
                        fixed_M = self._load_fixed_affine(fixed_M_path)
                    except Exception as e:
                        print(f"[WARN] Failed to load fixed affine ({fixed_M_path}): {e}")
                nir = self.align_nir_to_rgb(
                    rgb_frames, nir, backend,
                    mode="fixed" if fixed_M is not None else "first_frame",
                    fixed_M=fixed_M, save_csv_path=False, return_debug=False,
                )
            return nir

        def _preprocess_nir(nir, bvps):
            """Run nir through the 3-ch preprocess pipeline; return (nir_clips_1ch, bvps_clips)."""
            nir_config = config_preprocess.clone()
            nir_config.defrost()
            nir_config.DATA_TYPE = ["Raw"]
            nir_config.freeze()
            nir_clips_3ch, bvps_clips = self.preprocess(np.repeat(nir, 3, axis=-1), bvps, nir_config)
            if nir_clips_3ch.shape[-1] == 3:
                return nir_clips_3ch[..., 0:1], bvps_clips        # (Clips, T, H, W, 1)
            elif nir_clips_3ch.shape[1] == 3:
                return nir_clips_3ch[:, 0:1, ...], bvps_clips     # (Clips, 1, T, H, W)
            else:
                raise ValueError(f"Unexpected shape from preprocess: {nir_clips_3ch.shape}")

        # ── per-modality processing ───────────────────────────────────────────

        if self.modality == 'RGB':
            bvps, frames = self.match_length(ppg, rgb_frames)
            frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
            input_name_list, _ = self.save_multi_process(
                frames_clips, None, bvps_clips, data_dirs[i]['index']
            )

        elif self.modality == 'NIR':
            nir_frames = _load_nir_frames()
            bvps, nir_frames = self.match_length(ppg, nir_frames)
            nir_clips, bvps_clips = _preprocess_nir(nir_frames, bvps)
            input_name_list, _ = self.save_multi_process(
                None, nir_clips, bvps_clips, data_dirs[i]['index']
            )

        else:  # MULTI_SPECTRAL
            nir_frames = _load_nir_frames()
            bvps, frames, nir_frames = self.match_length(ppg, rgb_frames, nir_frames)
            frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
            nir_clips, _ = _preprocess_nir(nir_frames, bvps)
            input_name_list, _ = self.save_multi_process(
                frames_clips, nir_clips, bvps_clips, data_dirs[i]['index']
            )

        file_list_dict[i] = input_name_list

    # -------------------------------------------------------------------------
    # 7) Save helpers
    # -------------------------------------------------------------------------
    def save_multi_process(self, rgb_clips, nir_clips, bvp_clips, filename):
        """Save RGB (and optionally NIR) clips along with labels.

        Args:
            rgb_clips: array of RGB clip data.
            nir_clips: array of NIR clip data, or ``None`` for RGB-only mode.
            bvp_clips: array of BVP label data.
            filename:  base filename for saved files.
        """
        os.makedirs(self.cached_path, exist_ok=True)

        if self.data_format not in ('NDCHW', 'NCDHW', 'NDHWC'):
            raise ValueError(f'Unsupported Data Format: {self.data_format}')

        def _fmt(arr: np.ndarray) -> np.ndarray:
            """Transpose clip from NDHWC to the configured data format."""
            if self.data_format == 'NDCHW':
                arr = np.transpose(arr, (0, 3, 1, 2))
            elif self.data_format == 'NCDHW':
                arr = np.transpose(arr, (3, 0, 1, 2))
            # NDHWC: unchanged
            return np.float32(arr)

        input_path_name_list = []
        label_path_name_list = []

        for count, bvp_clip in enumerate(bvp_clips):
            input_path_name = os.path.join(self.cached_path, f"{filename}_input{count}.npz")
            label_path_name = os.path.join(self.cached_path, f"{filename}_label{count}.npy")
            input_path_name_list.append(input_path_name)
            label_path_name_list.append(label_path_name)

            save_dict = {'data_format': np.array(self.data_format)}
            if rgb_clips is not None:
                save_dict['rgb'] = _fmt(rgb_clips[count])
            if nir_clips is not None:
                save_dict['nir'] = _fmt(nir_clips[count])

            np.savez(input_path_name, **save_dict)
            np.save(label_path_name, np.float32(bvp_clip))

        return input_path_name_list, label_path_name_list

    # -------------------------------------------------------------------------
    # 8) Format conversion helper
    # -------------------------------------------------------------------------
    _FORMAT_AXES = {
        'NCDHW': 'CDHW',
        'NDCHW': 'DCHW',
        'NDHWC': 'DHWC',
    }

    @classmethod
    def _convert_format(cls, arr: np.ndarray, from_fmt: str, to_fmt: str) -> np.ndarray:
        """Convert a single clip (no batch dim) between data formats."""
        if from_fmt == to_fmt:
            return arr
        from_axes = cls._FORMAT_AXES[from_fmt]
        to_axes   = cls._FORMAT_AXES[to_fmt]
        perm = tuple(from_axes.index(c) for c in to_axes)
        return np.transpose(arr, perm)

    # -------------------------------------------------------------------------
    # 9) __getitem__
    # -------------------------------------------------------------------------
    def __getitem__(self, index):
        """Return a clip and its label.

        MULTI_SPECTRAL: returns ``(rgb, nir), label, filename, chunk_id``
            where the first element is a tuple so trainers can unpack via
            ``rgb, nir = batch[0]``.
        RGB / NIR: returns ``data, label, filename, chunk_id``.

        If the cache was saved in a different data format than the current
        config (e.g. _ms saved as NCDHW but config requests NDCHW), the
        clip is converted on the fly.  Old cache files without a
        ``data_format`` key are assumed to be NDCHW.
        """
        input_path = self.inputs[index]
        with np.load(input_path) as npz_file:
            keys = list(npz_file.keys())
            saved_fmt = str(npz_file['data_format']) if 'data_format' in keys else 'NDCHW'
            if self.modality != 'NIR':
                rgb = npz_file['rgb']
            if self.modality != 'RGB':
                if 'nir' not in keys:
                    raise KeyError(
                        f"Cache file is missing 'nir' key (found: {keys}). "
                        f"Re-preprocess with DO_PREPROCESS: True to regenerate: {input_path}"
                    )
                nir = npz_file['nir']

        if saved_fmt != self.data_format:
            if self.modality != 'NIR':
                rgb = self._convert_format(rgb, saved_fmt, self.data_format)
            if self.modality != 'RGB':
                nir = self._convert_format(nir, saved_fmt, self.data_format)

        label = np.load(self.labels[index])

        stem, _ = os.path.splitext(os.path.basename(input_path))
        filename, _, chunk_id = stem.rpartition('_input')

        if self.modality == 'MULTI_SPECTRAL':
            return (rgb, nir), label, filename, chunk_id
        elif self.modality == 'NIR':
            return nir, label, filename, chunk_id
        else:  # RGB
            return rgb, label, filename, chunk_id
