# LoFTR: Detector-Free Local Feature Matching with Transformers (Baseline Reproduction)

### [Original Repository](https://github.com/zju3dv/LoFTR) | [Project Page](https://zju3dv.github.io/loftr) | [Paper (CVPR 2021)](https://arxiv.org/pdf/2104.00680.pdf)

> **LoFTR: Detector-Free Local Feature Matching with Transformers**  
> Jiaming Sun, Zehong Shen, Yu'ang Wang, Hujun Bao, Xiaowei Zhou (CVPR 2021)  
> **Reproduction Team:** Matthew Pinto, Avneet Bahra, Sethmini Denegama, Joel Thomas

---

## 1. Overview & Repository Provenance

This repository is a reproduction fork of the official [zju3dv/LoFTR](https://github.com/zju3dv/LoFTR) codebase. The objective of this reproduction is to validate LoFTR's pretrained detector-free feature matching pipeline and reproduce the published relative camera pose estimation benchmarks on **ScanNet-1500** (indoor) and **MegaDepth-1500** (outdoor) without retraining or fine-tuning the model weights.

* **Original Upstream Repository:** `https://github.com/zju3dv/LoFTR`
* **Primary Evaluation Notebook:** `notebooks/LoFTR_Sethmini_REPRODUCTION_SCNNT_MGDPTH.ipynb`
* **Single-Pair Demo Notebook:** `notebooks/demo_single_pair.ipynb`

---

## 2. Summary of What Changed from the Original Setup

To run the 2021 codebase on modern systems (including CPU-only machines) and achieve accurate benchmark metrics, the following environment, code, and configuration changes were implemented:

### Environment & Dependency Fixes
* **Pinned Core Packages:** Configured the Conda environment (`loftr`) with Python 3.8, `pytorch==1.8.1` (`cudatoolkit=10.2`), and `pytorch-lightning==1.3.5`.
* **PyYAML Conflict Resolution:** Force-reinstalled `PyYAML==5.4.1` because newer default versions pulled by `pytorch-lightning` break legacy YAML configuration loading.
* **Cleaned `requirements.txt`:** Removed four non-critical development packages (`autopep8`, `pylint`, `ipython`, and `jupyterlab`) that caused dependency resolution conflicts with strict legacy version pins.
* **Jupyter Kernel Registration:** Added `ipykernel` to the `loftr` Conda environment and registered it as `Python (LoFTR)` for VS Code / Jupyter notebook execution.

### Code & Hardware Adaptations
* **CPU Fallback Support (`notebooks/demo_single_pair.ipynb`):** Added `map_location=torch.device("cpu")` when loading `.ckpt` weights and replaced `.cuda()` calls with `.cpu()` / `.cpu().numpy()` conversions so inference and Matplotlib visualizations run cleanly on machines without Nvidia CUDA GPUs.
* **Custom End-to-End Evaluation Notebook (`notebooks/LoFTR_Sethmini_REPRODUCTION_SCNNT_MGDPTH.ipynb`):** Built a consolidated evaluation pipeline covering dataset loading, preprocessing, pretrained inference, RANSAC essential-matrix pose estimation, angular error calculation, and Area Under the Curve (AUC) integration.

### Critical MegaDepth Configuration Fixes
During initial diagnostics on a 50-pair MegaDepth subset, median pose error started at **36.15°** (`AUC@5° = 0.20%`). Two critical adjustments resolved this:
1. **Coordinate Scaling (`scale0`, `scale1`):** Passed `scale0` and `scale1` from the `MegaDepthDataset` loader into the inference batch dictionary to preserve geometric consistency between resized/padded image coordinates (`img_resize=840`, `df=8`, `img_padding=True`) and original camera intrinsics (reduced median error from `36.15°` to `32.61°`).
2. **Positional Encoding Compatibility (`temp_bug_fix=False`):** Set `cfg_outdoor['coarse']['temp_bug_fix'] = False` when loading `outdoor_ds.ckpt` so positional encoding matches the pretrained outdoor checkpoint architecture (reduced median error from `32.61°` to `2.68°`).

---

## 3. How to Reproduce the Baseline

### Step 1: Clone the Repository & Checkout the Reproduction Branch
```bash
git clone https://github.com/matthewpinto1102-web/LoFTR
cd LoFTR
git checkout reproduction-baseline
```
## 4. Setup the Environment

### Step 1: Create and Register Environment
#conda env create -f environment.yaml
conda activate loftr

# Fix PyYAML compatibility with pytorch-lightning 1.3.5
pip install PyYAML==5.4.1 --force-reinstall

# Install and register the Jupyter kernel for VS Code / Jupyter
pip install ipykernel
python -m ipykernel install --user --name=loftr --display-name "Python (LoFTR)"

### Step 2 Download Pretrained Weights & Benchmark Datasets
# Download the pretrained checkpoints and test datasets from the official LoFTR Google Drive:

# Pretrained Checkpoints (~45 MB each): Place indoor_ds_new.ckpt and outdoor_ds.ckpt inside weights/.

# Benchmark Test Sets: Download scannet-1500-testset (~1 GB) and megadepth-1500-testset (~600 MB) and link or place them under data/scannet/test and data/megadepth/test:
# Or you can do it this way
ln -s /path/to/scannet-1500-testset/* /path/to/LoFTR/data/scannet/test
ln -s /path/to/megadepth-1500-testset/* /path/to/LoFTR/data/megadepth/test
# > **Note on File Paths:** Replace `/path/to/scannet-1500-testset` and `/path/to/megadepth-1500-testset` with the actual absolute or relative paths where you downloaded the test sets on your local machine.


# This is the structure your LoFTR folder should have when complete
LoFTR/
├── data/
│   ├── scannet/test/
│   └── megadepth/test/
├── notebooks/
│   ├── demo_single_pair.ipynb
│   └── LoFTR_Sethmini_REPRODUCTION_SCNNT_MGDPTH.ipynb
├── scripts/
├── src/
├── weights/
│   ├── indoor_ds_new.ckpt
│   └── outdoor_ds.ckpt
├── environment.yaml
├── requirements.txt
└── README.md

### Step 4: Run Verification & Full Benchmark Evaluation

#### Quick Sanity Check (Single Image Pair)

1. Open `notebooks/demo_single_pair.ipynb` in VS Code or Jupyter.
2. Select the `Python (LoFTR)` kernel.
3. Run all notebook cells sequentially.
4. Verify that a match visualization is generated. The documented sample run produced **843 matches**.

**CPU/GPU Note:** Keep `.cpu()` calls when running without a CUDA-compatible GPU, or use `.cuda()` on a compatible GPU.

#### Reproduction Workflow (Full 1,500-Pair Benchmark)

After completing the environment and dataset setup described above:

1. Open `notebooks/LoFTR_Sethmini_REPRODUCTION_SCNNT_MGDPTH.ipynb` in Jupyter or VS Code.
2. Select the `Python (LoFTR)` kernel.
3. Verify the dataset root paths (`MEGA_ROOT`, `SCENE_ROOT`, and ScanNet paths).
4. Confirm the pretrained checkpoints: `indoor_ds_new.ckpt` for ScanNet and `outdoor_ds.ckpt` for MegaDepth.
5. Run the notebook cells in the order specified.
6. Compare the generated AUC@5°, AUC@10°, and AUC@20° values with the reproduction results listed below.

**ScanNet-1500 Pipeline:** Loads grayscale image pairs resized to 640 × 480, normalizes tensors to `[0, 1]`, runs inference with `indoor_ds_new.ckpt`, and estimates relative pose using RANSAC (`thresh=0.5`, `conf=0.99999`).

**MegaDepth-1500 Pipeline:** Uses `MegaDepthDataset` (`mode="test"`, `img_resize=840`, `df=8`, `img_padding=True`, `depth_padding=False`, `min_overlap_score=0.0`), passes `scale0`/`scale1` and padding masks, sets `temp_bug_fix=False`, and runs inference with `outdoor_ds.ckpt`.

#### Alternative: Original PyTorch Lightning Test Scripts

You can also run the original testing scripts from the terminal:

```bash
bash ./scripts/reproduce_test/indoor_ds.sh
bash ./scripts/reproduce_test/outdoor_ds.sh
```

# Sources
LoFTR: Detector-Free Local Feature Matching with Transformers. Sun, Jiaming; Shen, Zehong; Wang, Yuang; Bao, Hujun; Zhou, Xiaowei. CVPR, 2021.

## What We Did Differently from the Authors & Debugging Fixes

While this reproduction uses the authors' original model architecture, pretrained checkpoints (`indoor_ds_new.ckpt` and `outdoor_ds.ckpt`), and RANSAC pose estimation utilities without retraining, several workflow, environment, and configuration adjustments were made to run and debug the evaluation pipeline:

### 1. Workflow & Environment Differences
* **Consolidated Notebook Evaluation Pipeline:** Instead of running the original multi-file PyTorch Lightning CLI test scripts (`test.py` via `scripts/reproduce_test/*.sh`), we built an end-to-end interactive evaluation workflow in `notebooks/LoFTR_Sethmini_REPRODUCTION_SCNNT_MGDPTH.ipynb`.
* **Hardware & Dependency Adaptations:** Updated the environment to resolve legacy `PyYAML` (`5.4.1`) conflicts with `pytorch-lightning 1.3.5`, cleaned up non-essential development packages from `requirements.txt`, registered a dedicated `Python (LoFTR)` Jupyter kernel, and added CPU fallback support (`map_location=torch.device("cpu")` and `.cpu().numpy()`) so single-pair inference and visualization can run without a dedicated CUDA GPU.
* **Qualitative Visualization Filtering:** For visual inspection on ScanNet, we restricted plots to the top **150 highest-confidence matches** (out of 2,096 predicted correspondences on the sample pair) for visual clarity, while retaining all matches for full benchmark pose estimation.

### 2. Debugging & Configuration Fixes for MegaDepth Inaccuracies
Initial MegaDepth evaluations produced substantially higher pose errors than the published paper. Diagnostic testing on a 50-pair subset identified and resolved two principal configuration mismatches:

| Diagnostic Stage | Median Pose Error (50-Pair Test) | Root Cause & Fix Applied |
| :--- | :---: | :--- |
| **Initial Configuration** | `36.15°` | Omitted coordinate scaling factors and used incompatible positional encoding settings. |
| **Fix 1: Coordinate Scaling (`scale0`, `scale1`)** | `32.61°` | Explicitly passed `scale0` and `scale1` from `MegaDepthDataset` (`img_resize=840`, `df=8`, `img_padding=True`) to maintain geometric consistency between resized/padded image coordinates and original camera intrinsics. |
| **Fix 2: Checkpoint Compatibility (`temp_bug_fix=False`)** | `2.68°` | Restored `temp_bug_fix=False` in the coarse transformer config so positional encoding matched the pretrained `outdoor_ds.ckpt` weights. |

### 3. Summary of Reproduced Results vs. Published Paper
After applying the fixes above, the full 1,500-pair evaluations were run using RANSAC essential-matrix estimation (`thresh=0.5` px, `conf=0.99999`) and pose error defined as e_pose = max(e_R, e_t):

| Benchmark (1,500 Pairs) | Metric | Published Paper | Our Reproduction | Difference (Δ) | Unsuccessful Pose Estimates |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **ScanNet-1500** (`indoor_ds_new.ckpt`) | **AUC@5°** | 22.06% | **21.52%** | -0.54 pp | 74 / 1,500 |
| **ScanNet-1500** (`indoor_ds_new.ckpt`) | **AUC@10°** | 40.80% | **40.37%** | -0.43 pp | — |
| **ScanNet-1500** (`indoor_ds_new.ckpt`) | **AUC@20°** | 57.62% | **57.18%** | -0.44 pp | — |
| **MegaDepth-1500** (`outdoor_ds.ckpt`) | **AUC@5°** | 52.80% | **52.80%** | 0.00 pp | 9 / 1,500 |
| **MegaDepth-1500** (`outdoor_ds.ckpt`) | **AUC@10°** | 69.19% | **69.19%** | 0.00 pp | — |
| **MegaDepth-1500** (`outdoor_ds.ckpt`) | **AUC@20°** | 81.18% | **81.19%** | +0.01 pp | — |

* **ScanNet-1500:** Reproduced AUCs closely track the published figures across all thresholds (within 0.43–0.54 percentage points), with minor variance attributable to RANSAC stochasticity and standalone notebook preprocessing.
* **MegaDepth-1500:** Once `scale0`/`scale1` and `temp_bug_fix=False` were corrected, reproduced AUCs matched the published paper within **0.01 percentage points**.
