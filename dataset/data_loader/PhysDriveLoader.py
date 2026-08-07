"""The dataloader for PhysDrive datasets.
PhysDrive: https://github.com/WJULYW/PhysDrive-Dataset
Jiyao Wang, Xiao Yang, Qingyong Hu, Jiankai Tang, Can Liu, Dengbo He, Yuntao Wang, Ying-Cong Chen, Kaishun Wu. (2025) PhysDrive: A Multimodal Remote Physiological Measurement Dataset for In-vehicle Driver Monitoring
"""

import glob
import json
import os
import re
import scipy.io as sio
import cv2
import numpy as np
import pandas as pd
from dataset.data_loader.BaseLoader import BaseLoader
import threading
from neurokit2 import ppg_peaks, ppg_quality, NeuroKitWarning
import warnings

warnings.filterwarnings("ignore", category=NeuroKitWarning)

# Subjects excluded from all splits (warping failure or poor warping quality).
# These subjects will not be preprocessed, loaded, or evaluated.
# Additional exclusions (e.g. B-prefix subjects) are configured via
# DATA.FILTERING.USE_EXCLUSION_LIST / EXCLUSION_LIST in YAML.
_EXCLUDE_EXPLICIT = frozenset({'AFZ3'})

EXCLUDE_SUBJECTS = _EXCLUDE_EXPLICIT  # kept for backward-compat references


def _load_fixed_matrix(path_):
    """Load fixed transformation matrix (2×3 affine or 3×3 homography) from file."""
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
    if M.shape not in ((2, 3), (3, 3)):
        raise ValueError(f"Fixed transform must be 2x3 (affine) or 3x3 (homography), got {M.shape}")
    return M


# 하위 호환용 alias
_load_fixed_affine = _load_fixed_matrix


def _load_subject_matrix(path_or_dir: str, sequence_index: str):
    """Subject별 또는 global matrix 로드.

    path_or_dir가 디렉토리 → {subject_id}.npy 우선, 없으면 __GLOBAL__.npy fallback.
    path_or_dir가 파일     → 해당 파일 직접 로드.
    """
    def _read(p):
        M = _load_fixed_matrix(p)
        return M

    try:
        if os.path.isdir(path_or_dir):
            subject_id = sequence_index.split("_")[0]
            candidate = os.path.join(path_or_dir, f"{subject_id}.npy")
            if os.path.exists(candidate):
                M = _read(candidate)
                print(f"  [ALIGN] subject matrix: {candidate}  shape={M.shape}")
                return M
            print(f"  [ALIGN] no subject matrix found for {subject_id} in {path_or_dir} "
                  f"(global fallback disabled)")
            return None
        elif os.path.isfile(path_or_dir):
            M = _read(path_or_dir)
            print(f"  [ALIGN] fixed matrix: {path_or_dir}  shape={M.shape}")
            return M
        else:
            print(f"  [ALIGN] path not found: {path_or_dir}")
            return None
    except Exception as e:
        print(f"  [ALIGN] load error ({path_or_dir}): {e}")
        return None


class PhysDriveLoader(BaseLoader):
    """The data loader for the PhysDrive dataset.

    Supports three modalities via YAML config key ``MODALITY``:
      - ``'MULTI_SPECTRAL'``: preprocesses both RGB.mp4 and IR.mp4; saves as
        ``.npz`` with ``'rgb'`` (3-ch, DiffNormalized) and ``'nir'`` (1-ch, Raw)
        keys.  ``__getitem__`` returns ``(rgb, nir), label, filename, chunk_id``.
      - ``'RGB'``: preprocesses RGB.mp4 only.
        ``__getitem__`` returns ``rgb, label, filename, chunk_id``.
      - ``'NIR'``: preprocesses IR.mp4 only (grayscale → Raw).
        ``__getitem__`` returns ``nir, label, filename, chunk_id``.

    When MODALITY is ``'RGB'`` or ``'NIR'`` and a MULTI_SPECTRAL (``_ms``) cache
    already exists, the loader reads from that cache and slices the required key
    — avoiding redundant preprocessing.

    Cache-path suffixes: ``_ms`` / ``_rgb`` / ``_nir``.
    All input clips are stored as ``.npz``; labels as ``.npy``.
    """

    def __init__(self, name, data_path, config_data, device=None):
        """Initializes a PhysDrive dataloader.
            Args:
                data_path (str): Path to a folder containing raw video and BVP data.
                For example, data_path should be "On-Road-rPPG" for the following structure:
                -----------------
                     On-Road-rPPG/
                     |-- AFH1/
                         |-- A1/
                             |-- Video/
                                 |-- RGB.mp4
                                 |-- IR.mp4
                             |-- Label/
                                 |-- BVP.mat
                         |-- A2/...
                     |-- AFH2/
                     ...
                -----------------
                name (str): Name of the dataloader.
                config_data (CfgNode): Data settings (ref: config.py).
        """
        self.modality = getattr(config_data, 'MODALITY', 'MULTI_SPECTRAL').upper()
        if self.modality not in ('RGB', 'NIR', 'MULTI_SPECTRAL'):
            raise ValueError(
                f"PhysDriveLoader: MODALITY must be 'RGB', 'NIR', or 'MULTI_SPECTRAL', "
                f"got '{self.modality}'."
            )

        # If RGB/NIR is requested but an _ms cache exists and no modality-specific cache exists, reuse _ms.
        _ms_cached = config_data.CACHED_PATH + '_ms'
        _mod_suffix = {'RGB': '_rgb', 'NIR': '_nir', 'MULTI_SPECTRAL': '_ms'}[self.modality]
        _mod_cached = config_data.CACHED_PATH + _mod_suffix
        if self.modality in ('RGB', 'NIR') and os.path.isdir(_ms_cached) and not os.path.isdir(_mod_cached):
            suffix = '_ms'
            print(f"[PhysDriveLoader] MODALITY='{self.modality}': no {_mod_suffix} cache found, using MS cache → slicing from {_ms_cached}")
        else:
            suffix = {'RGB': '_rgb', 'NIR': '_nir', 'MULTI_SPECTRAL': '_ms'}[self.modality]

        cfg = config_data.clone()
        cfg.defrost()
        cfg.CACHED_PATH = cfg.CACHED_PATH + suffix
        base, ext = os.path.splitext(cfg.FILE_LIST_PATH)
        cfg.FILE_LIST_PATH = base + suffix + (ext if ext else '')
        cfg.freeze()

        # Build exclusion pattern list from FILTERING config.
        # Supports exact IDs (e.g. 'AFZ3') and prefix wildcards (e.g. 'B*').
        filtering = getattr(config_data, 'FILTERING', None)
        if filtering and getattr(filtering, 'USE_EXCLUSION_LIST', False):
            self._exclusion_patterns = [p for p in getattr(filtering, 'EXCLUSION_LIST', []) if p]
        else:
            self._exclusion_patterns = []

        super().__init__(name, data_path, cfg, device)

    def _is_subject_excluded(self, subject_id: str) -> bool:
        """Returns True if subject_id matches any hardcoded or config-based exclusion.

        Pattern rules:
          - 'B*'   → prefix match (subject_id.startswith('B'))
          - 'AFZ3' → exact match
        """
        if subject_id in _EXCLUDE_EXPLICIT:
            return True
        for pattern in self._exclusion_patterns:
            if pattern.endswith('*'):
                if subject_id.startswith(pattern[:-1]):
                    return True
            elif subject_id == pattern:
                return True
        return False

    def get_raw_data(self, data_path):
        """Returns data directories under the given path (for PhysDrive dataset)."""
        subject_dirs = glob.glob(os.path.join(data_path, "*"))
        print(f"Found {len(subject_dirs)} subject directories")
        data_dirs = []
        subject_id_map = {}

        for subject_dir in subject_dirs:
            subject_name = os.path.basename(subject_dir)  # e.g., "AFH1"
            if subject_name == "processed":
                continue  # Skip the 'processed' directory
            if self._is_subject_excluded(subject_name):
                print(f"[PhysDriveLoader] Skipping excluded subject: {subject_name}")
                continue
            if subject_name not in subject_id_map:
                subject_id_map[subject_name] = len(subject_id_map) + 1
            # Retrieve all session directories (e.g., A1, A2) under the subject
            session_dirs = glob.glob(os.path.join(subject_dir, "*"))
            for session_dir in session_dirs:
                session_name = os.path.basename(session_dir)  # e.g., "A1"
                unique_id = f"{subject_name}_{session_name}"
                data_dirs.append({
                    "index": unique_id,
                    "path": session_dir,
                    "subject": subject_id_map[subject_name]
                })
        return data_dirs

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data_dirs based on begin and end values,
        ensuring no overlapping subjects between splits."""

        if begin == 0 and end == 1:
            return data_dirs

        data_info = dict()
        for data in data_dirs:
            subject = data['subject']
            data_dir = data['path']
            index = data['index']
            if subject not in data_info:
                data_info[subject] = []
            data_info[subject].append({"index": index, "path": data_dir, "subject": subject})

        subj_list = sorted(list(data_info.keys()))
        num_subjs = len(subj_list)

        subj_range = list(range(0, num_subjs))
        if begin != 0 or end != 1:
            subj_range = list(range(int(begin * num_subjs), int(end * num_subjs)))

        data_dirs_new = []
        for i in subj_range:
            subj_num = subj_list[i]
            subj_files = data_info[subj_num]
            data_dirs_new += subj_files

        return data_dirs_new

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """Called by preprocess_dataset for multiprocessing."""
        try:
            session_info = data_dirs[i]
            session_path = session_info['path']
            saved_filename = session_info['index']

            subject_id = saved_filename.split('_')[0]
            if self._is_subject_excluded(subject_id):
                print(f"[Skip] {saved_filename}: subject in EXCLUDE_SUBJECTS")
                return

            # existing = glob.glob(os.path.join(self.cached_path, f"{saved_filename}_input*.npz"))  #260417_B process
            # if existing:  #260417_B process
            #     print(f"[Skip] {saved_filename}: already cached ({len(existing)} clips)")  #260417_B process
            #     with threading.Lock():  #260417_B process
            #         file_list_dict[i] = existing  #260417_B process
            #     return  #260417_B process

            # is_b_subject = subject_id.startswith('B')  #260417_B process

            video_dir = os.path.join(session_path, "Video")
            label_dir = os.path.join(session_path, "Label")
            if not os.path.isdir(video_dir):
                raise NotADirectoryError(f"video dir missing: {video_dir}")

            # ── Read RGB frames ───────────────────────────────────────────────
            has_rgb = self.modality in ('RGB', 'MULTI_SPECTRAL')
            has_nir = self.modality in ('NIR', 'MULTI_SPECTRAL')

            if has_rgb:
                if 'None' in config_preprocess.DATA_AUG:
                    rgb_frames = self.read_video(os.path.join(video_dir, "RGB.mp4"))
                elif 'Motion' in config_preprocess.DATA_AUG:
                    npy_files = glob.glob(os.path.join(session_path, '*.npy'))
                    if not npy_files:
                        raise FileNotFoundError(f"No .npy files in {session_path}")
                    rgb_frames = self.read_npy_video(npy_files)
                else:
                    raise ValueError(f'Unsupported DATA_AUG: {config_preprocess.DATA_AUG}')
                if rgb_frames.size == 0:
                    raise ValueError(f"Empty RGB frames: {video_dir}")

            # ── Read NIR (IR) frames ──────────────────────────────────────────
            if has_nir:
                ir_path = os.path.join(video_dir, "IR.mp4")
                if not os.path.isfile(ir_path):
                    raise FileNotFoundError(f"IR.mp4 missing: {ir_path}")
                nir_frames = self.read_nir_video(ir_path)   # (T, H, W, 1)
                if nir_frames.size == 0:
                    raise ValueError(f"Empty NIR frames: {ir_path}")

            # ── (Optional) RGB → NIR alignment (NIR이 anchor) ───────────────
            crop_face_cfg = getattr(config_preprocess, "CROP_FACE", None)
            use_align = bool(getattr(crop_face_cfg, "FACE_ALIGNMENT", False)) if crop_face_cfg else False

            if use_align and has_rgb and has_nir:
                matrix_path = (
                    getattr(crop_face_cfg, "FIXED_MATRIX_PATH", None) if crop_face_cfg else None
                ) or "/path/to/affine_matrices/per_subject_affine_physdrive"
                fixed_M = _load_subject_matrix(matrix_path, saved_filename)
                # RGB 프레임을 NIR 좌표계로 warp
                rgb_frames = self._align_rgb_to_nir(rgb_frames, nir_frames, fixed_M)

            # ── Read BVP label ────────────────────────────────────────────────
            ref_frames = rgb_frames if has_rgb else nir_frames
            if config_preprocess.USE_PSUEDO_PPG_LABEL:
                bvps = self.generate_pos_psuedo_labels(ref_frames, fs=self.config_data.FS)
            else:
                bvp_path = os.path.join(label_dir, "BVP.mat")
                if not os.path.isfile(bvp_path):
                    raise FileNotFoundError(f"BVP.mat missing: {bvp_path}")
                bvps = self.read_wave(bvp_path)
                if bvps.size == 0:
                    raise ValueError(f"Empty BVP: {bvp_path}")

            # ── Align lengths ─────────────────────────────────────────────────
            target_length = ref_frames.shape[0]
            bvps = BaseLoader.resample_ppg(bvps, target_length)

            # If both streams exist, truncate to the shorter one.
            if has_rgb and has_nir:
                min_len = min(rgb_frames.shape[0], nir_frames.shape[0])
                rgb_frames = rgb_frames[:min_len]
                nir_frames = nir_frames[:min_len]
                bvps      = bvps[:min_len]

            # ── Preprocess clips ──────────────────────────────────────────────
            # NIR이 anchor: NIR에서 face box를 구한 뒤 RGB(warp 완료)에도 공유
            rgb_clips = nir_clips = None
            if has_nir:
                nir_clips, bvps_clips, nir_face_regions = self._preprocess_nir(
                    nir_frames, bvps, config_preprocess, return_face_regions=True)
            if has_rgb:
                # 정렬 완료된 RGB는 NIR과 동일 좌표계 → NIR face box 재사용
                # face_regions = nir_face_regions if (use_align and has_nir and not is_b_subject) else None  #260417_B process
                face_regions = nir_face_regions if (use_align and has_nir) else None
                rgb_clips, bvps_clips = self.preprocess(
                    rgb_frames, bvps, config_preprocess,
                    precomputed_face_regions=face_regions)
            if has_nir and not has_rgb:
                pass  # bvps_clips already set above

            # ── Quality filter (on BVP signal) ────────────────────────────────
            q_rgb, q_nir, q_bvps = [], [], []
            skipped = 0
            for idx, b_clip in enumerate(bvps_clips):
                quality = self.single_signal_quality_assessment(b_clip, fs=self.config_data.FS)
                if quality < 0.5:
                    print(f"[Warning] Skipping low-quality clip "
                          f"{saved_filename}/{idx+1}/{len(bvps_clips)}: quality={quality:.3f}")
                    skipped += 1
                    continue
                if has_rgb:  q_rgb.append(rgb_clips[idx])
                if has_nir:  q_nir.append(nir_clips[idx])
                q_bvps.append(b_clip)

            if not q_bvps:
                print(f"[Warning] All clips in {saved_filename} are low quality. Skipping.")
                return
            print(f"{skipped}/{len(bvps_clips)} clips skipped.")

            # ── Save ──────────────────────────────────────────────────────────
            input_name_list, _ = self.save_multi_process(
                q_rgb if has_rgb else None,
                q_nir if has_nir else None,
                q_bvps,
                saved_filename,
            )
            with threading.Lock():
                file_list_dict[i] = input_name_list

        except Exception as e:
            print(f"[Error] Failed to process {session_path}: {str(e)}")

    @staticmethod
    def read_video(video_file):
        """Reads RGB.mp4, returns frames (T, H, W, 3) uint8 in RGB order."""
        VidObj = cv2.VideoCapture(video_file)
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, frame = VidObj.read()
        frames = []
        while success:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            success, frame = VidObj.read()
        VidObj.release()
        return np.asarray(frames)

    @staticmethod
    def read_nir_video(video_file):
        """Reads IR.mp4 (stored as 3-ch BGR), converts to 1-ch grayscale.
        Returns frames (T, H, W, 1) uint8.
        """
        VidObj = cv2.VideoCapture(video_file)
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, frame = VidObj.read()
        frames = []
        while success:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray[:, :, np.newaxis])   # (H, W, 1)
            success, frame = VidObj.read()
        VidObj.release()
        return np.asarray(frames, dtype=np.uint8)   # (T, H, W, 1)

    def _align_rgb_to_nir(self, rgb_frames, nir_frames, fixed_M=None):
        """RGB 프레임을 NIR 좌표계로 warp합니다 (NIR이 anchor).

        compute_global_affine_matrix_physdrive.py 가 출력하는 행렬 H는
        NIR → RGB 방향입니다.  역행렬 H^{-1} 을 이용해 RGB → NIR 로 변환합니다.

        Args:
            rgb_frames : (N, H_rgb, W_rgb, 3)
            nir_frames : (N, H_nir, W_nir, 1)  — 해상도 참조용
            fixed_M    : 3×3 homography 또는 2×3 affine (NIR→RGB 방향)

        Returns:
            np.ndarray (N, H_nir, W_nir, 3)  — NIR 좌표계로 warp된 RGB
        """
        N, H_nir, W_nir = nir_frames.shape[:3]
        out = np.empty((N, H_nir, W_nir, 3), dtype=rgb_frames.dtype)

        if fixed_M is not None:
            M = np.asarray(fixed_M, dtype=np.float32)
            if M.shape == (2, 3):
                # affine 2×3 → 3×3 으로 확장 후 역행렬
                M33 = np.vstack([M, [0, 0, 1]]).astype(np.float64)
                M_inv = np.linalg.inv(M33).astype(np.float32)
            else:
                # homography 3×3
                M_inv = np.linalg.inv(M.astype(np.float64)).astype(np.float32)
        else:
            print("[WARN] No fixed matrix provided; using identity (no alignment).")
            M_inv = np.eye(3, dtype=np.float32)

        dsize = (W_nir, H_nir)
        for t in range(N):
            frame_bgr = cv2.cvtColor(rgb_frames[t], cv2.COLOR_RGB2BGR)
            warped_bgr = cv2.warpPerspective(
                frame_bgr, M_inv, dsize,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            out[t] = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2RGB)
        return out

    def _preprocess_nir(self, nir_frames, bvps, config_preprocess, return_face_regions=False):
        """Preprocess NIR (IR) frames with DATA_TYPE=['Raw'] and return 1-ch clips.

        Args:
            return_face_regions: True이면 face box도 반환 (RGB와 공유하기 위해)

        Returns:
            nir_clips       : (N_clips, T, H, W, 1)  float32
            bvps_clips      : (N_clips, T)            float32
            face_regions    : face bounding boxes     (return_face_regions=True 시에만)
        """
        nir_config = config_preprocess.clone()
        nir_config.defrost()
        nir_config.DATA_TYPE = ['Raw']
        nir_config.freeze()

        result = self.preprocess(
            np.repeat(nir_frames, 3, axis=-1), bvps, nir_config,
            return_face_regions=return_face_regions,
        )

        if return_face_regions:
            nir_clips_3ch, bvps_clips, face_regions = result
        else:
            nir_clips_3ch, bvps_clips = result
            face_regions = None

        # Keep only channel-0
        if nir_clips_3ch.ndim == 5 and nir_clips_3ch.shape[-1] == 3:
            nir_clips = nir_clips_3ch[..., 0:1]
        elif nir_clips_3ch.ndim == 5 and nir_clips_3ch.shape[1] == 3:
            nir_clips = nir_clips_3ch[:, 0:1, ...]
        else:
            raise ValueError(f"Unexpected NIR clip shape: {nir_clips_3ch.shape}")

        if return_face_regions:
            return nir_clips, bvps_clips, face_regions
        return nir_clips, bvps_clips

    @staticmethod
    def read_wave(bvp_file):
        """Reads a BVP signal file."""
        waves = sio.loadmat(bvp_file)["BVP"].flatten()
        return np.asarray(waves)

    @staticmethod
    def single_signal_quality_assessment(signal, fs=30, method_quality='templatematch', method_peaks='elgendi'):
        assert method_quality in ['templatematch', 'dissimilarity'], "method_quality must be one of ['templatematch', 'dissimilarity']"

        signal_filtered = signal

        if len(signal_filtered) < 10 or np.all(signal_filtered == signal_filtered[0]):
            print("Warning: Signal is too short or constant. Skipping quality assessment.")
            return 0

        if method_quality in ['templatematch', 'dissimilarity']:
            method_quality = 'dissimilarity' if method_quality == 'dissimilarity' else method_quality

            try:
                _, peak_info = ppg_peaks(
                    signal_filtered,
                    sampling_rate=fs,
                    method=method_peaks
                )

                if peak_info["PPG_Peaks"].size == 0:
                    print("No peaks detected in the signal. Skipping quality assessment.")
                    return 0

                quality = ppg_quality(
                    signal_filtered,
                    ppg_pw_peaks=peak_info["PPG_Peaks"],
                    sampling_rate=fs,
                    method=method_quality
                )

                quality = np.nanmean(quality)

            except ValueError as e:
                print(f"Error in ppg_quality function: {e}")
                quality = 0

            return quality

    # -------------------------------------------------------------------------
    # save_multi_process
    # -------------------------------------------------------------------------
    def save_multi_process(self, rgb_clips, nir_clips, bvp_clips, filename):
        """Save clips as .npz (input) + .npy (label).

        Args:
            rgb_clips : list/array of RGB clip arrays, or None.
            nir_clips : list/array of NIR clip arrays, or None.
            bvp_clips : list/array of BVP label arrays.
            filename  : base filename for saved files.
        """
        os.makedirs(self.cached_path, exist_ok=True)

        if self.data_format not in ('NDCHW', 'NCDHW', 'NDHWC'):
            raise ValueError(f'Unsupported Data Format: {self.data_format}')

        def _fmt(arr):
            """Transpose clip from NDHWC to configured data_format."""
            arr = np.asarray(arr, dtype=np.float32)
            if self.data_format == 'NDCHW':
                arr = np.transpose(arr, (0, 3, 1, 2))
            elif self.data_format == 'NCDHW':
                arr = np.transpose(arr, (3, 0, 1, 2))
            return arr

        input_path_name_list = []
        label_path_name_list = []

        for count, bvp_clip in enumerate(bvp_clips):
            input_path = os.path.join(self.cached_path, f"{filename}_input{count}.npz")
            label_path = os.path.join(self.cached_path, f"{filename}_label{count}.npy")
            input_path_name_list.append(input_path)
            label_path_name_list.append(label_path)

            save_dict = {}
            if rgb_clips is not None:
                save_dict['rgb'] = _fmt(rgb_clips[count])
            if nir_clips is not None:
                save_dict['nir'] = _fmt(nir_clips[count])

            np.savez(input_path, **save_dict)
            np.save(label_path, np.float32(bvp_clip))

        return input_path_name_list, label_path_name_list

    # -------------------------------------------------------------------------
    # load_preprocessed_data
    # -------------------------------------------------------------------------
    def load_preprocessed_data(self):
        """Load preprocessed file list. Handles .npz inputs → .npy labels."""
        file_list_df = pd.read_csv(self.file_list_path)
        inputs = sorted(file_list_df['input_files'].tolist())
        # Filter out any excluded subjects that may exist in a stale cache.
        before = len(inputs)
        inputs = [f for f in inputs
                  if not self._is_subject_excluded(os.path.basename(f).split('_')[0])]
        if len(inputs) < before:
            print(f"[PhysDriveLoader] load_preprocessed_data: filtered out "
                  f"{before - len(inputs)} clips from exclusion list")
        if not inputs:
            raise ValueError(f'{self.dataset_name} dataset loading data error!')
        # Labels: same base name, _input → _label, always .npy
        labels = [
            inp.replace('_input', '_label').rsplit('.npz', 1)[0] + '.npy'
            if inp.endswith('.npz')
            else inp.replace('input', 'label')
            for inp in inputs
        ]
        self.inputs = inputs
        self.labels = labels
        self.preprocessed_data_len = len(inputs)

    # -------------------------------------------------------------------------
    # build_file_list_retroactive
    # -------------------------------------------------------------------------
    def build_file_list_retroactive(self, data_dirs, begin, end):
        """Generate file list from existing cache (searches for .npz and .npy)."""
        data_dirs_subset = self.split_raw_data(data_dirs, begin, end)
        filename_list = list({d['index'] for d in data_dirs_subset})

        file_list = []
        for fname in filename_list:
            # Prefer .npz; fall back to .npy for legacy RGB-only caches.
            found = glob.glob(os.path.join(self.cached_path, f"{fname}_input*.npz"))
            if not found:
                found = glob.glob(os.path.join(self.cached_path, f"{fname}_input*.npy"))
            file_list.extend(found)

        if not file_list:
            raise ValueError(self.dataset_name,
                             'File list empty. Check preprocessed data folder.')

        file_list_df = pd.DataFrame(file_list, columns=['input_files'])
        os.makedirs(os.path.dirname(self.file_list_path), exist_ok=True)
        file_list_df.to_csv(self.file_list_path)

    # -------------------------------------------------------------------------
    # __getitem__
    # -------------------------------------------------------------------------
    def __getitem__(self, index):
        """Return a clip and its label.

        Reads from .npz cache (keys 'rgb' and/or 'nir') and returns:
          MULTI_SPECTRAL → ``(rgb, nir), label, filename, chunk_id``
          RGB            → ``rgb,        label, filename, chunk_id``
          NIR            → ``nir,        label, filename, chunk_id``
        """
        input_path = self.inputs[index]
        label = np.float32(np.load(self.labels[index]))

        stem = os.path.splitext(os.path.basename(input_path))[0]  # e.g. AFH1_A1_input0
        filename, _, chunk_id = stem.rpartition('_input')

        if input_path.endswith('.npz'):
            with np.load(input_path) as npz:
                if self.modality != 'NIR':
                    rgb = np.float32(npz['rgb'])
                if self.modality != 'RGB':
                    nir = np.float32(npz['nir'])
        else:
            # Legacy .npy cache (RGB-only, stored as NDHWC) → transpose to NDCHW
            arr = np.float32(np.load(input_path))
            if arr.ndim == 4 and arr.shape[-1] < arr.shape[-2]:
                arr = np.transpose(arr, (0, 3, 1, 2))  # NDHWC → NDCHW
            rgb = arr

        # Convert to requested data_format at load time.
        # Cache may have been built with a different format; detect by shape:
        # NDCHW (D,C,H,W): shape[0] >> shape[1]  e.g. (160, 3, 128, 128)
        # NCDHW (C,D,H,W): shape[0] << shape[1]  e.g. (  3, 160, 128, 128)
        def _to_format(arr):
            cached_is_ncdhw = arr.ndim == 4 and arr.shape[0] < arr.shape[1]
            if self.data_format == 'NCDHW' and not cached_is_ncdhw:
                return np.transpose(arr, (1, 0, 2, 3))  # DCHW → CDHW
            elif self.data_format == 'NDCHW' and cached_is_ncdhw:
                return np.transpose(arr, (1, 0, 2, 3))  # CDHW → DCHW
            return arr

        if self.modality == 'MULTI_SPECTRAL':
            return (_to_format(rgb), _to_format(nir)), label, filename, chunk_id
        elif self.modality == 'NIR':
            return _to_format(nir), label, filename, chunk_id
        else:
            return _to_format(rgb), label, filename, chunk_id