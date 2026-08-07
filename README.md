# MS-rPPG: Multi-Spectral Remote Photoplethysmography

MS-rPPG estimates the blood-volume-pulse (BVP) signal and heart rate from facial
video. Unlike conventional RGB-only rPPG, MS-rPPG fuses **RGB and near-infrared
(NIR)** streams, making it robust to the difficult illumination found in
in-vehicle (driving) scenarios.

The core model, **`MSMamba`**, combines a lightweight 3D-conv trunk, a
**cross-spectral FiLM (CSLM)** gating module that exchanges physiological-band
information between the RGB and NIR streams, and a stack of Mamba-based temporal /
channel scans with a frequency-domain FFN. Either modality can be used alone, or
both can be fused.

---

## Highlights

- **Multi-spectral (RGB + NIR) fusion** with cross-spectral FiLM gating.
- Single entry point (`main.py`) for **training, testing, and unsupervised baselines**.
- Works with either modality: `MULTI_SPECTRAL`, `RGB`, or `NIR`.
- Ships **pretrained MS-rPPG checkpoints** for MR-NIRP and PhysDrive.
- Includes strong baselines: PhysNet, TS-CAN, DeepPhys, EfficientPhys, PhysFormer,
  PhysMamba, RhythmFormer, RhythmMamba, iBVPNet, BigSmall, and unsupervised
  methods (POS, CHROM, ICA, GREEN, LGI, PBV, OMIT).

---

## Repository structure

```
main.py                     # single entry point for all modes
config.py                   # yacs-based configuration (defaults + path resolution)
setup.sh                    # environment setup (conda or uv)
requirements.txt
configs/
  train_configs/            # YAML for training runs  (TRAIN_TESTDATA_MODEL.yaml)
  infer_configs/            # YAML for inference-only runs
neural_methods/
  model/                    # model definitions (MsMamba.py + baselines)
  trainer/                  # one trainer per model (inherits BaseTrainer)
  loss/                     # NegPearson, PhysFormer, RhythmFormer/Hybrid losses
dataset/data_loader/        # one loader per dataset (inherits BaseLoader)
evaluation/                 # metrics.py, post_process.py, Bland-Altman, cost calc
unsupervised_methods/       # signal-processing baselines
pretrained_models/          # released MS-rPPG checkpoints
tools/                      # visualization / analysis utilities + vendored mamba
```

---

## Installation

Requires Linux, an NVIDIA GPU (CUDA), and Python 3.8. The Mamba kernels
(`mamba-ssm`, `causal-conv1d`) require a CUDA-capable GPU.

```bash
# with conda
bash setup.sh conda
conda activate rppg-toolbox

# or with uv
bash setup.sh uv
source .venv/bin/activate
```

`setup.sh` creates the environment, installs PyTorch (CUDA build) and
`requirements.txt`, then builds the vendored Mamba kernels in `tools/mamba`.
If your CUDA version differs, edit the `torch==...` line in `setup.sh` accordingly.

---

## Datasets

MS-rPPG development targets multi-spectral driving datasets, and the toolbox also
supports the common public RGB datasets:

| Dataset | Modality | Notes |
|---------|----------|-------|
| **MR-NIRP** | RGB + NIR | in-vehicle multi-spectral |
| **PhysDrive**     | RGB + NIR | in-vehicle multi-spectral |
| **MSDrive** | RGB + NIR | cross-dataset evaluation |
| PURE, UBFC-rPPG, UBFC-PHYS, MMPD, SCAMPS, iBVP, COHFACE, BP4D+, LADH, SUMS | RGB | baselines |

You must obtain each dataset from its original authors.
The MS-Drive dataset proposed in this work is publicly available at the following repository:
**MS-Drive** https://github.com/ziiho08/MS-Drive

Then point the config at your local copy — see the placeholder paths (`/path/to/<DATASET>/...`) inside the
YAML files.

### Preprocessing

The first run preprocesses raw video into cached `.npy` chunks. In each config:

- Set `...DATA.DO_PREPROCESS: True` for the **first** run.
- Set `...DATA.DO_PREPROCESS: False` afterwards to reuse the cache.
- `DATA_PATH` = raw dataset; `CACHED_PATH` = where preprocessed chunks are stored.
- `BEGIN`/`END` split subjects (not frames) into train/val/test, so there is no
  subject leakage between splits.

---

## Quick start — inference with a pretrained model

Two MS-rPPG checkpoints are provided in `pretrained_models/`:

| Checkpoint | Trained on | Modality | Reported HR MAE |
|------------|-----------|----------|-----------------|
| `MRNIRP_MSMamba.pth`   | MR-NIRP   | RGB + NIR | **3.35 bpm** (MR-NIRP test) |
| `PhysDrive_MSMamba.pth`| PhysDrive | RGB + NIR | **7.56 bpm** (PhysDrive test) |

The relevant inference configs already reference these checkpoints. Edit the
dataset `DATA_PATH` / `CACHED_PATH` to your local copy, then run:

```bash
# PhysDrive multi-spectral test
python main.py --config_file configs/infer_configs/PhysDrive_MSMAMBA_MULTI_test.yaml

# MR-NIRP model, evaluated on MSDrive (cross-dataset)
python main.py --config_file configs/infer_configs/MRNIRP_MSDrive_MSMAMBA.yaml
```

`only_test` mode loads the model from `INFERENCE.MODEL_PATH`.

---

## Training

```bash
# Train + test MS-rPPG on MR-NIRP (RGB + NIR)
python main.py --config_file configs/train_configs/0MR-NIRP_MSMAMBA.yaml

# Train + test MS-rPPG on PhysDrive
python main.py --config_file configs/train_configs/2PhysDrive_MSMAMBA.yaml
```

Notes:
- `TOOLBOX_MODE: train_and_test` runs training, then testing in one command.
- `TEST.USE_LAST_EPOCH: True` uses the final epoch for testing (no validation set
  required); set it to `False` to select the best-validation checkpoint.
- Checkpoints are written to `MODEL.MODEL_DIR` as
  `{MODEL_FILE_NAME}_Epoch{N}_Seed{SEED}.pth`. For MSMamba the training modality
  (`multi` / `rgb` / `nir`) is appended to `MODEL_FILE_NAME` automatically.
- The random seed is fixed to `100` in `main.py`; override with `--seed`.

### Choosing the modality

Set `MODALITY` under each `DATA` block:

- `MULTI_SPECTRAL` — RGB + NIR fusion (default MS-rPPG setting)
- `RGB` — RGB only
- `NIR` — NIR only

---

## Testing an existing checkpoint

Use an `only_test` config (or set `TOOLBOX_MODE: only_test` in any config) and
point `INFERENCE.MODEL_PATH` to your `.pth`:

```bash
python main.py --config_file configs/infer_configs/PhysDrive_MSDrive_MSMAMBA.yaml
```

---

## Configuration system

Configs are `yacs` YAML files; every key has a default in `config.py`.
`get_config()` loads the YAML, resolves derived output paths, and freezes the
config. Naming convention: `TRAIN_TESTDATA_MODEL_VARIANT.yaml`.

Key parameters:

| Parameter | Meaning |
|-----------|---------|
| `TOOLBOX_MODE` | `train_and_test`, `only_test`, or `unsupervised_method` |
| `MODEL.NAME` | `MSMamba` or a baseline (e.g. `PhysNet`, `Tscan`, `PhysMamba`) |
| `*.DATA.MODALITY` | `MULTI_SPECTRAL` / `RGB` / `NIR` |
| `*.DATA.DO_PREPROCESS` | `True` on first run, then `False` |
| `TEST.USE_LAST_EPOCH` | `True` = last epoch, `False` = best-validation |
| `INFERENCE.MODEL_PATH` | checkpoint for `only_test` |
| `INFERENCE.EVALUATION_METHOD` | `"FFT"` or `"peak detection"` |
| `TEST.METRICS` | any of `MAE`, `RMSE`, `MAPE`, `Pearson`, `SNR`, `MACC`, `BA` |
| `TEST.OUTPUT_SAVE_DIR` | if set, saves raw predictions/labels as `.pickle` |

### MSMamba model parameters (`MODEL.MSMamba`)

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `FRAMES` | 160 | temporal length (must match `CHUNK_LENGTH`) |
| `DEPTH` | 4 | number of MSMamba blocks |
| `EMBED_DIM` | 96 | embedding dimension |

---

## Outputs

- **Stdout:** mean ± standard error for each metric.
- **Bland-Altman plots:** `{LOG.PATH}/{EXP_DATA_NAME}/bland_altman_plots/`.
- **Metrics table:** written under `TEST.EXCEL_SAVE_DIR` (derived from `LOG.PATH`
  if left empty).
- **Raw predictions/labels:** `.pickle` under `TEST.OUTPUT_SAVE_DIR` when enabled.

By default nothing is written outside the repository; experiment outputs land
under `runs/` (git-ignored).

---

## Unsupervised baselines

```bash
python main.py --config_file configs/infer_configs/PhysDrive_UNSUPERVISED.yaml
python main.py --config_file configs/infer_configs/MR-NIRP_UNSUPERVISED.yaml
```

Supported methods: POS, CHROM, ICA, GREEN, LGI, PBV, OMIT.

---

## Tools

`tools/` contains optional utilities (each with its own README):

- `output_signal_viz/` — visualize predicted vs. ground-truth BVP.
- `preprocessing_viz/` — inspect preprocessed data chunks.
- `motion_analysis/` — head-pose / action-unit motion analysis.
- `illum_summary.py` — illumination-condition metric breakdown for PhysDrive.
- `mamba/` — vendored Mamba (`mamba-ssm`) kernels built during setup.

---

## Adding new components

- **New model:** add to `neural_methods/model/`, add a trainer in
  `neural_methods/trainer/` (implement `train`, `valid`, `test`, `save_model`),
  and register it in both `train_and_test()` and `test()` in `main.py`.
- **New dataset:** add a loader in `dataset/data_loader/` (implement
  `preprocess_dataset`, `read_video`, `read_wave`) and register it in the three
  dataloader sections of `main.py`.
- **Custom splits:** set `DATA.FILE_LIST_PATH` to a CSV with an `input_files` column.

---

## Acknowledgements

Built on the [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) and
[Mamba](https://github.com/state-spaces/mamba). See `LICENSE` for license terms.



## 📄 Citation

If you use **MS-rPPG** or find this repository useful for your research, please cite our paper:

```bibtex
@article{choi2026ms,
  title={MS-rPPG: Multi-spectral State Space Model for Remote Photoplethysmography in Driver Monitoring Systems},
  author={Choi, Jiho and Lee, Sang Jun},
  journal={arXiv preprint arXiv:2606.21115},
  year={2026}
}
```

