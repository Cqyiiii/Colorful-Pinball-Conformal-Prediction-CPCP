# Colorful Pinball: Density-Weighted Quantile Regression for Conditional Guarantee of Conformal Prediction

[![arXiv](https://img.shields.io/badge/arXiv-2512.24139-b31b1b.svg)](https://arxiv.org/abs/2512.24139)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

This repository contains the official implementation of the paper **"Colorful Pinball: Density-Weighted Quantile Regression for Conditional Guarantee of Conformal Prediction"**.

**Authors:** Qianyi Chen, Bo Li (Tsinghua University)

## 📖 Abstract

While conformal prediction provides robust marginal coverage guarantees, achieving reliable **conditional coverage** for specific inputs remains challenging. We propose **Colorful Pinball Conformal Prediction (CPCP)**, a framework that directly minimizes the Mean Squared Conditional Error (MSCE).

Our method leverages a density-weighted pinball loss implemented via a **Three-Head Network**, trained through a three-stage calibration procedure:
1.  **Estimation**: Jointly train three quantiles.
2.  **Fine-tuning**: Optimize the central quantile through minimizing the pinball loss with estimated density-based weights.
3.  **Conformalization**: Apply standard split conformal prediction on rectified nonconformity scores.

## 📂 Project Structure

The codebase is organized as follows:

```text
.
├── main.py             # Entry point: runs the full benchmark suite
├── methods.py          # High-level wrappers for CPCP, CQR, PLCP, RCP, etc.
├── models.py           # Neural network architectures 
├── trainers.py         # Training loops 
├── losses.py           # Pinball loss, ALD loss, Multivariate NLL
├── metrics.py          # Metrics: WSC, CCE, Coverage, Size
├── data_utils.py       # Data loaders and preprocessing
├── utils.py            # Random seeding and device management
```



## 🚀 Usage

### 1. Data Preparation

Please ensure the following dataset files are present in the `./Datasets/` directory (as required by `data_utils.py`):

- `bike_hour.csv` (Bike Sharing)
- `diamonds.csv` (Diamonds)
- `gt_full.csv` (Gas Turbine)
- `naval.csv` (Naval Propulsion)
- `sgemm_product.csv` (SGEMM GPU Kernel)
- `super.csv` (Superconductivity)
- `transcoding_measurement.tsv` (Video Transcoding)
- `WEC_Perth_49.csv` (Wave Energy)

### 2. Running the Benchmark

The `main.py` script is set up to run the full benchmark suite across all datasets and methods mentioned in the paper.

To reproduce the experiments:

Bash

```
python main.py
```

### 3. Customizing the Run

The script currently iterates through all datasets and methods. To run specific experiments (e.g., only CPCP on the Bike dataset), you can modify the lists in `main.py`:

Python

```python
# In main.py:

# Select specific datasets
dataset_loaders = {    
    "bike": load_bike,
    # "diamond": load_diamonds,  # Comment out others to skip
}

# Select specific methods
methods = [
    # ('Split', run_split),
    ('CPCP-Clip+Mix', lambda *a, **k: run_rcp_density_improved(*a, epsilon=0.02, mode='clip', clip_max=5.0, mix_ratio=0.5, **k)),
]
```

## 📊 Methods Implemented

- **Split**: Standard Split Conformal Prediction.

- **CQR**: Conformalized Quantile Regression 

- **RCP**: Rectified Conformal Prediction.

- **Gaussian Scoring**: Multivariate Gaussian NLL minimization.

- **PLCP**: Partition Learning Conformal Prediction ($K=20, 50$).

- **CPCP (Ours)**:

  - `CPCP-Split`: Vanilla implementation.
  - `CPCP-Clip`: With weight clipping for stability.
  - `CPCP-Mix`: With loss mixing.
  - `CPCP-Clip+Mix`: The robust version recommended in the paper.

- **Ablation Baselines**

  - `RCP/CQR-ALD`: Substitute standard quantile regression with MLE of ALD 

  - `RCP-MultiHead`: Only co-training three quantiles 

  - `CPCP-delta`: Different $\delta$ settings ($\delta=0.01,0.05$)

    


## 🔗 Citation

If you find this code or paper useful, please cite our arXiv preprint:

Code snippet

```markdown
@article{chen2025colorful,
  title={Colorful Pinball: Density-Weighted Quantile Regression for Conditional Guarantee of Conformal Prediction},
  author={Chen, Qianyi and Li, Bo},
  journal={arXiv preprint arXiv:2512.24139},
  year={2025}
}
```

## 📄 License

This project is licensed under the MIT License.



## 📝 Supplementary Notes

- We observed that the **Gaussian Scoring** baseline exhibits numerical instability on the **WEC** dataset (likely due to high dimensionality and label correlations) for exactly one random seed. Consequently, we excluded this seed and reported results averaged over the remaining 19 seeds. The processing details are provided in `table_gs_wec_refine.ipynb`.



