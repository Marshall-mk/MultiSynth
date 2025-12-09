# U-HVED 

## Example Script (`example.py`)

Demonstrates the orthogonal stack generation with detailed output showing:
- Resolution configurations for each stack
- Which axis has high resolution for each stack
- Verification that each stack has high-res in different orientations

## Usage

### Basic Training

```bash
python train_uhved.py \
  --hr_image_dir /path/to/hr/images \
  --model_dir ./models \
  --epochs 100 \
  --batch_size 2 \
  --learning_rate 1e-4
```

### Advanced Training with Custom Settings

```bash
python train_uhved.py \
  --hr_image_dir /path/to/hr/images \
  --model_dir ./models \
  --val_image_dir /path/to/val/images \
  --epochs 200 \
  --batch_size 4 \
  --learning_rate 1e-4 \
  --base_channels 32 \
  --num_scales 4 \
  --kl_weight 0.001 \
  --modality_weight 0.5 \
  --atlas_res 1.0 1.0 1.0 \
  --min_resolution 1.0 1.0 1.0 \
  --max_res_aniso 9.0 9.0 9.0 \
  --prob_motion 0.2 \
  --prob_noise 0.8 \
  --prob_bias_field 0.5 \
  --use_wandb \
  --wandb_project uhved-orthogonal \
  --mixed_precision fp16
```

### Test Orthogonal Stack Generation

```bash
python example.py
```

This will generate synthetic data and show:
- Input HR volume shape
- Three LR stacks with their resolution configurations
- Verification that each stack has high-res in the expected orientation

## Data Flow

### Training Pipeline

```
HR Volume (B, C, D, H, W)
    ↓
HRLRDataGenerator.generate_paired_data()
    ↓
Three Orthogonal Resolutions Created:
  - Stack 0: [1mm, 5mm, 7mm]  ← High-res in D
  - Stack 1: [6mm, 1mm, 8mm]  ← High-res in H
  - Stack 2: [7mm, 6mm, 1mm]  ← High-res in W
    ↓
Independent Degradations Applied:
  - Bias field (multiplicative)
  - Intensity augmentation (gamma)
  - Physics simulation (PSF, k-space, noise)
    ↓
Three LR Stacks: [lr_axial, lr_coronal, lr_sagittal]
    ↓
U-HVED Model (3 modalities)
    ↓
Fused via Product of Gaussians
    ↓
SR Output (B, C, D, H, W)
```

### Model Architecture

```
Input: List of 3 LR stacks
  ↓
MultiModalEncoder (3 independent encoders)
  ├─ Stack 0 (Axial) → mu_0, logvar_0
  ├─ Stack 1 (Coronal) → mu_1, logvar_1
  └─ Stack 2 (Sagittal) → mu_2, logvar_2
  ↓
MultiScaleFusion (Product of Gaussians at each scale)
  → Fused (mu, logvar) → Sample z
  ↓
Decoder (with skip connections)
  ↓
Output: SR volume (B, C, D, H, W)
```

## Key Parameters

### Resolution Parameters

- `--atlas_res`: HR image resolution in mm [x, y, z] (default: [1.0, 1.0, 1.0])
- `--min_resolution`: Minimum (best) resolution (default: [1.0, 1.0, 1.0])
- `--max_res_aniso`: Maximum (worst) anisotropic resolution (default: [9.0, 9.0, 9.0])
- `--no_randomise_res`: Disable randomization (use max_res_aniso for all low-res axes)

### Model Parameters

- `--base_channels`: Base feature channels (default: 32)
- `--num_scales`: Number of hierarchical scales (default: 4)

### Loss Weights

- `--kl_weight`: KL divergence weight (default: 0.001)
- `--perceptual_weight`: Perceptual loss weight (default: 0.0, disabled)
- `--modality_weight`: Modality reconstruction weight (default: 0.5)

### Physics Simulation Probabilities

- `--prob_motion`: Motion ghosting artifacts (default: 0.2)
- `--prob_spike`: K-space spike artifacts (default: 0.05)
- `--prob_aliasing`: Aliasing/wrap-around (default: 0.1)
- `--prob_bias_field`: B1 field inhomogeneity (default: 0.5)
- `--prob_noise`: Rician/Gaussian noise (default: 0.8)


## File Structure

```
MultiSynth/
├── src/
│   ├── data.py                          # Modified data pipeline
│   ├── encoder.py                       # U-HVED encoder
│   ├── decoder.py                       # U-HVED decoder
│   ├── fusion.py                        # Product of Gaussians
│   ├── losses.py                        # UHVEDLoss
│   └── uhved.py                         # Main model
├── train.py                             # Training script 
├── example.py                           # Demo script 
├── requirements.txt                     # Dependencies
└── README.md                            # This file
```

## Requirements

```bash
pip install -r requirements.txt
```
