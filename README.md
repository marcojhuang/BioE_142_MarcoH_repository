# TorchANI Neural Network Potential — ANI Energy Prediction

Training an ANI-style neural network potential (NNP) to predict molecular energies for small organic molecules (H, C, N, O) using [TorchANI](https://github.com/aiqm/torchani) and PyTorch on the ANI-1 GDB dataset.

**Best model:** 10-fold CV mean MAE = **1.675 ± 0.506 kcal/mol** | Best single fold MAE = **1.01 kcal/mol**

---

## Table of Contents

- [Overview](#overview)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Environment Setup](#environment-setup)
- [Dataset](#dataset)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training Details](#training-details)
- [Acknowledgements](#acknowledgements)

---

## Overview

This project trains per-element atomic neural networks following the ANI framework ([Smith et al., Chem. Sci., 2017, 8, 3192](https://doi.org/10.1039/C6SC05720A)). Each atom type (H, C, N, O) has its own subnetwork. Atomic Environment Vectors (AEVs) encode each atom's local chemical environment using radial and angular symmetry functions, and the per-atom energy predictions are summed to produce a total molecular energy.

The model was developed as a final project for a UC Berkeley course (B142) and trained on the UC Berkeley Savio cluster using a single NVIDIA GPU.

---

## Results

### Per-element-category MAE (best fold model on held-out test set)

| Category | Count | MAE (kcal/mol) | RMSE (kcal/mol) |
|---|---|---|---|
| H + C only | 17,639 | 0.84 | 1.24 |
| H + C + N | 32,299 | 1.13 | 1.52 |
| H + C + O | 25,244 | 0.84 | 1.14 |
| H + C + N + O | 11,307 | 1.28 | 1.64 |

### 10-Fold Cross-Validation Summary

| Metric | Value |
|---|---|
| Mean MAE | 1.675 ± 0.506 kcal/mol |
| Mean RMSE | 2.123 ± 0.539 kcal/mol |
| Best fold MAE | 1.016 kcal/mol (fold 5) |

---

## Repository Structure

```
.
├── README.md
├── B142_MH_final_projc_6.ipynb        # Main Jupyter notebook (training + evaluation)
├── torch_env_b142.yml                  # Conda environment file
├── best_model_fold5_MAE101.pt          # Best model weights (fold 5, MAE ≈ 1.01 kcal/mol)
├── MH_final_proj_weights_6.pt          # Weights from single-split best run (MAE ≈ 1.17 kcal/mol)
└── CKPT_5_B142_MH_final_projc_6.pdf   # Rendered notebook (PDF)
```

---

## Environment Setup

### Prerequisites

- Linux (tested on Ubuntu; developed on NERSC HPC)
- NVIDIA GPU with CUDA 12.1 support
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/)

### 1. Clone this repository

```bash
git clone https://github.com/<YOUR_USERNAME>/<REPO_NAME>.git
cd <REPO_NAME>
```

### 2. Create the conda environment

The `torch_env_b142.yml` file pins every dependency. Create the environment with:

```bash
conda env create -f torch_env_b142.yml -n torchani_env
conda activate torchani_env
```

> **Note:** The yml file references an absolute prefix path from the original HPC system. Conda will ignore this and install to your local envs directory. If creation fails due to platform mismatches (the yml was exported from Linux x86_64), use the manual install below instead.

### 3. Manual install (alternative)

If the yml file doesn't work on your system, install the key packages manually:

```bash
conda create -n torchani_env python=3.11 -y
conda activate torchani_env

# PyTorch with CUDA 12.1
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# TorchANI and other dependencies
pip install torchani==2.2
pip install numpy pandas matplotlib seaborn scipy tqdm ipykernel jupyter
```

### 4. Verify the installation

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import torchani; print('TorchANI version:', torchani.__version__)"
```

You should see `CUDA available: True` and `TorchANI version: 2.2`.

> **CPU-only:** The code will fall back to CPU automatically if no GPU is available, but training will be significantly slower.

---

## Dataset

This project uses the **ANI-1 GDB s01–s04** dataset (`ani_gdb_s01_to_s04.h5`), which contains molecular conformations with up to 4 heavy atoms (C, N, O) plus hydrogens.

| Split | Samples |
|---|---|
| Train (80%) | 691,918 |
| Validation (10%) | 86,489 |
| Test (10%) | 86,489 |
| **Total** | **864,896** |

## Usage

### Training from scratch

1. Activate the environment and launch Jupyter:

```bash
conda activate torchani_env
jupyter notebook B142_MH_final_projc_6.ipynb
```

2. Run all cells sequentially. The notebook covers:
   - AEV computer setup
   - Dataset loading and splitting
   - Multiple architecture experiments (see [Training Details](#training-details))
   - 10-fold cross-validation
   - Per-element error analysis

### Loading a pretrained model

To load the best saved weights and run evaluation:

```python
import torch
import torch.nn as nn
import torchani

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Rebuild the AEV computer 
def init_aev_computer():
    Rcr = 5.2
    Rca = 3.5
    EtaR = torch.tensor([16], dtype=torch.float, device=device)
    ShfR = torch.tensor([
        0.900000, 1.168750, 1.437500, 1.706250,
        1.975000, 2.243750, 2.512500, 2.781250,
        3.050000, 3.318750, 3.587500, 3.856250,
        4.125000, 4.393750, 4.662500, 4.931250
    ], dtype=torch.float, device=device)
    EtaA = torch.tensor([8], dtype=torch.float, device=device)
    Zeta = torch.tensor([32], dtype=torch.float, device=device)
    ShfA = torch.tensor([0.90, 1.55, 2.20, 2.85], dtype=torch.float, device=device)
    ShfZ = torch.tensor([
        0.19634954, 0.58904862, 0.9817477, 1.37444680,
        1.76714590, 2.15984490, 2.5525440, 2.94524300
    ], dtype=torch.float, device=device)
    num_species = 4
    return torchani.AEVComputer(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species)

aev_computer = init_aev_computer()

#  Rebuild the network architecture 
class Bigger_lowerDO_AtomicNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(384, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.layers(x)

nets = [Bigger_lowerDO_AtomicNet() for _ in range(4)]
ani_net = torchani.ANIModel(nets)
model = nn.Sequential(aev_computer, ani_net).to(device)

#  Load weights 
model.load_state_dict(
    torch.load('best_model_fold5_MAE101.pt', map_location=device, weights_only=True)
)
model.eval()
print("Model loaded successfully.")
```

---

## Model Architecture

### AEV (Atomic Environment Vector) — 384 dimensions

| Component | Cutoff (Å) | Shifts | Elements |
|---|---|---|---|
| Radial | 5.2 | 16 | 4 species × 16 = 64 |
| Angular | 3.5 | 4 radial × 8 angular | 10 pairs × 32 = 320 |
| **Total** | | | **384** |

### Best network per atom type: `Bigger_lowerDO_AtomicNet`

```
Input (384) → Linear(512) → BN → ReLU → Dropout(0.05)
           → Linear(256) → BN → ReLU → Dropout(0.05)
           → Linear(128) → BN → ReLU → Dropout(0.05)
           → Linear(1)
```

Four copies (one per element: H, C, N, O) are combined via `torchani.ANIModel`.

**Parameters Per Individual Atomic Network:** 363,265
**Total parameters:** 1,453,060

---

## Training Details

### Hyperparameters (best configuration)

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Initial learning rate | 1 × 10⁻⁴ |
| Weight decay (L2) | 1 × 10⁻⁵ |
| Batch size | 8,192 |
| Epochs | 200 |
| LR scheduler | ReduceLROnPlateau (patience=10, factor=0.5, min_lr=1 × 10⁻⁸) |
| Loss function | MSE |
| Gradient clipping | max_norm = 1.0 |
| Early stopping | Yes (restore best val weights) |

### Experiments run (summary)

| Hidden Layers | Learning Rate | Num Epochs | L2 Regularization | Dropout | Patience | Test MAE (kcal/mol) |
|---|---|---|---|---|---|---|
| [384, 256, 192, 128, 1] | 1e-04 | 30 | 1e-05 | 0.15 | 10 | 7.89 |
| [384, 256, 192, 128, 1] | 1e-08 | 100 | 1e-05 | 0.15 | 10 | 349.59 |
| [384, 256, 192, 128, 1] | 1e-05 | 100 | 1e-05 | 0.15 | 10 | 7.57 |
| [384, 256, 192, 128, 1] | 1e-04 | 100 | 1e-05 | 0.15 | 10 | 3.51 |
| [384, 256, 192, 128, 1] | 1e-04 | 200 | 1e-05 | 0.15 | 10 | 2.42 |
| [384, 512, 256, 128, 1] | 1e-04 | 200 | 1e-05 | 0.15 | 10 | 1.81 |
| **[384, 512, 256, 128, 1]** | **1e-04** | **200** | **1e-05** | **0.05** | **10** | **1.17** |
| [384, 512, 256, 128, 1] | 1e-04 | 200 | 1e-05 | 0 | 10 | 4.92 |
| [384, 512, 256, 128, 1] | 1e-04 | 200 | 1e-05 | 0.05 | 5 | 2.68 |

---

## Acknowledgements

- [TorchANI](https://github.com/aiqm/torchani) — Xiang Gao, Farhad Ramezanghorbani, et al.
- ANI-1 dataset and methodology — [Smith, Isayev, Roitberg. *Chem. Sci.*, 2017, 8, 3192](https://doi.org/10.1039/C6SC05720A)
- Training adapted from the [TorchANI training example](https://aiqm.github.io/torchani/examples/nnp_training.html)
- UC Berkeley B142 course staff
