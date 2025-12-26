# U-HVED: Hetero-Modal Variational Encoder-Decoder

A deep learning model for super-resolution of medical images using multiple orthogonal low-resolution acquisitions. U-HVED fuses information from axial, coronal, and sagittal stacks using Product of Gaussians to produce isotropic high-resolution outputs.

## Models

This repository provides two models for medical image super-resolution:

- **U-HVED** (Hetero-Modal Variational Encoder-Decoder): Advanced multi-modal fusion with Product of Gaussians, variational latent space, orientation reconstruction, and support for missing views
- **U-Net Baseline**: Standard 3D U-Net with concatenated stack inputs for comparison and baseline evaluation

Both models fuse information from orthogonal LR acquisitions (axial, coronal, sagittal) to produce isotropic HR outputs.

## Quick Start

### Basic Training

```bash
python train.py \
  --hr_image_dir /path/to/hr/images \
  --model_dir ./models \
  --epochs 100 \
  --batch_size 2 \
  --learning_rate 1e-4
```

### Training with Validation

```bash
python train.py \
  --hr_image_dir /path/to/hr/images \
  --val_image_dir /path/to/val/images \
  --model_dir ./models \
  --epochs 100 \
  --batch_size 2 \
  --val_interval 5
```

## Model Architecture Configurations

### 1. Decoder Upsampling Modes

The decoder supports three upsampling strategies, each with different trade-offs:

**Trilinear (Default - Recommended)**
```bash
python train.py --model_dir ./models --decoder_upsample_mode trilinear
```
- Uses trilinear interpolation + convolution
- Smooth outputs, general-purpose
- Best for most use cases

**Transposed Convolution**
```bash
python train.py --model_dir ./models --decoder_upsample_mode transpose
```
- Learnable upsampling via transposed convolution
- Can produce checkerboard artifacts
- Useful when you want fully learnable upsampling

**Pixel Shuffle**
```bash
python train.py --model_dir ./models --decoder_upsample_mode pixelshuffle
```
- Sub-pixel convolution (PixelShuffle3d)
- Efficient, preserves high-frequency details
- Good for sharp texture reconstruction

### 2. Architecture Options

**Use Learned Prior in Latent Space**
```bash
python train.py --model_dir ./models --use_prior
```
- Includes a learned prior in the Product of Gaussians fusion
- Can improve reconstruction when views are missing

**Use Encoder Features as Skip Connections**
```bash
python train.py --model_dir ./models --use_encoder_outputs_as_skip
```
- Passes encoder features directly to decoder as skip connections
- Helps preserve fine details from input orientations

**Final Activation Function**
```bash
# Sigmoid (default) - outputs in [0, 1]
python train.py --model_dir ./models --final_activation sigmoid

# Tanh - outputs in [-1, 1]
python train.py --model_dir ./models --final_activation tanh

# None - no final activation
python train.py --model_dir ./models --final_activation none
```

## Orientation Dropout (Handling Missing Views)

U-HVED can be trained to handle missing orientations, useful for inference when not all views are available.

### Random Dropout (Robust Training)

Randomly drops 1-2 orientations during training with specified probability:

```bash
# 50% chance to drop orientations, keep at least 1 view
python train.py \
  --model_dir ./models \
  --orientation_dropout_prob 0.5 \
  --min_orientations 1
```

### Deterministic Dropout (Controlled Experiments)

Always drop specific orientations for controlled experiments:

```bash
# Always drop axial view (train with coronal + sagittal only)
python train.py --model_dir ./models --drop_orientations 0

# Always drop axial and coronal (train with sagittal only)
python train.py --model_dir ./models --drop_orientations 0 1

# Orientation indices: 0=Axial, 1=Coronal, 2=Sagittal
```

**Note:** Deterministic and random dropout are mutually exclusive. If both are specified, deterministic takes precedence.

## Ablation Studies

### SR-Only Training (Disable Orientation Reconstruction)

For ablation studies, you can train with all encoders but only the SR decoder, disabling the auxiliary orientation reconstruction task:

```bash
# SR-only training (ablation)
python train.py \
  --model_dir ./models/ablation_sr_only \
  --no_reconstruct_orientations \
  --orientation_weight 0.0 \
  --recon_weight 0.8
```

**What this does:**
- All 3 encoders (Axial, Coronal, Sagittal) still process their respective LR stacks
- Product of Gaussians fusion still combines information from all orientations
- Only the SR decoder is used (saves ~30-40% parameters)
- No orientation reconstruction decoders
- Orientation loss automatically becomes 0.0

**When to use:**
- Ablation studies to measure the contribution of orientation reconstruction
- Reduced GPU memory requirements
- Faster training (fewer parameters to update)

**Recommended loss weight adjustment:**
When disabling orientation reconstruction, rebalance the loss weights to maintain the same total weight:
- Without flag (default): `--recon_weight 0.4 --orientation_weight 0.4` (total: 0.8)
- With flag: `--recon_weight 0.8 --orientation_weight 0.0` (total: 0.8)

**Checkpoint compatibility:**
Checkpoints saved with different `reconstruct_orientations` settings are incompatible (different decoder architectures). The training script will automatically validate this and provide a clear error message if there's a mismatch.

## Loss Configuration

U-HVED uses a multi-component loss function that balances reconstruction quality, regularization, and auxiliary tasks.

### U-HVED Loss Components

**Total Loss Formula**:
```
L_total = α·L_recon + β·L_KL + γ·L_orientation + δ·L_SSIM + ε·L_perceptual

Where:
  L_recon = Reconstruction loss (L1/L2/Charbonnier) between SR output and HR target
  L_KL = KL divergence for latent space regularization
  L_orientation = Reconstruction loss for input orientations (auxiliary task)
  L_SSIM = Structural similarity loss (optional)
  L_perceptual = 3D perceptual loss using medical imaging networks (optional)
```

### 1. Reconstruction Loss (`--recon_weight`, default: 0.4)

**Purpose**: Primary loss for SR output quality

**Options** (`--recon_loss_type`):
- **`l1`** (recommended): Mean Absolute Error - robust to outliers, produces sharper edges
- **`l2`**: Mean Squared Error - smoother outputs, penalizes large errors more
- **`charbonnier`**: Differentiable approximation of L1, combines benefits of L1 and L2

**Recommended value**: `0.4` with orientation reconstruction, `0.8` without

**Why this value**:
- Primary signal for SR quality
- Balanced with auxiliary tasks (orientation reconstruction) when both are enabled
- When orientation reconstruction is disabled, increase to 0.8 to maintain total reconstruction weight

**Adjust if**:
- **Too low**: Model prioritizes latent regularization over output quality → blurry outputs
- **Too high**: Model may overfit to pixel-wise accuracy, losing perceptual quality

### 2. KL Divergence Loss (`--kl_weight`, default: 0.1)

**Purpose**: Regularizes latent space to follow standard Gaussian distribution N(0,1)

**Formula**: `KL(q(z|x) || p(z)) = -0.5 * Σ(1 + log(σ²) - μ² - σ²)`

**Recommended value**: `0.1`

**Why this value**:
- Provides gentle regularization without dominating the loss
- Prevents posterior collapse (all latents mapping to same value)
- Enables sampling from latent space for generation
- Too high causes "KL vanishing" (encoder outputs useless latents)
- Too low allows latent space to become irregular and non-Gaussian

**Adjust if**:
- **Increase to 0.2-0.5**: If latent space is too spread out, sampling produces poor results
- **Decrease to 0.05**: If model is underfitting, latents are too constrained
- **Set to 0.0**: For deterministic autoencoder (not recommended, loses variational benefits)

**KL Annealing** (optional):
```bash
# Start with KL weight 0.0, gradually increase to 0.1 over first epochs
# Prevents KL collapse in early training
# Currently not implemented, but recommended for future enhancement
```

### 3. Orientation Reconstruction Loss (`--orientation_weight`, default: 0.4)

**Purpose**: Auxiliary task to reconstruct input LR stacks, provides additional supervision

**Formula**: `L_orientation = mean([L1(recon_i, input_i) for i in orientations])`

**Recommended value**: `0.4` when enabled, `0.0` when disabled

**Why this value**:
- Equal weight with SR reconstruction (0.4 + 0.4 = 0.8 total reconstruction)
- Acts as regularization, preventing latent space from discarding orientation-specific information
- Forces encoders to preserve information useful for reconstruction
- Improves SR quality through multi-task learning

**Adjust if**:
- **Increase to 0.6**: If orientation reconstructions are poor quality
- **Decrease to 0.2**: If SR quality is suffering, orientation task is dominating
- **Set to 0.0 with `--no_reconstruct_orientations`**: SR-only training (ablation study)

**When disabled** (`--no_reconstruct_orientations`):
- Set to 0.0 automatically
- Increase `--recon_weight` to 0.8 to compensate
- Faster training, fewer parameters, but less regularization

### 4. SSIM Loss (`--ssim_weight`, default: 0.1, requires `--use_ssim`)

**Purpose**: Structural similarity metric, focuses on perceptual quality over pixel accuracy

**Formula**: SSIM compares local patterns of luminance, contrast, and structure

**Recommended value**: `0.1` when enabled

**Why this value**:
- Complements pixel-wise losses (L1/L2) with perceptual quality
- Small weight prevents dominating the loss (SSIM range is different from L1)
- Helps preserve texture and structural patterns
- Too high can cause training instability (SSIM is non-convex)

**Adjust if**:
- **Increase to 0.2**: If outputs look sharp but lack structural coherence
- **Decrease to 0.05**: If training is unstable
- **Set to 0.0 or omit `--use_ssim`**: For pure pixel-wise training

**SSIM variants**:
- Standard SSIM (default): Window-based structural similarity
- Can be computed on 3D volumes directly

### 5. Perceptual Loss (`--perceptual_weight`, default: 0.1, requires `--use_perceptual`)

**Purpose**: Deep perceptual similarity using 3D medical imaging networks

**Formula**: L2 distance in feature space of pretrained 3D networks (MedicalNet, MONAI, Models Genesis)

**Recommended value**: `0.1` when enabled

**Backends** (`--perceptual_backend`):
- **`medicalnet`**: 3D ResNet pretrained on MedCalc dataset
- **`monai`**: MONAI's pretrained 3D networks
- **`models_genesis`**: Self-supervised pretrained 3D encoder

**Why this value**:
- Provides perceptual similarity beyond pixel-wise metrics
- Helps generate realistic medical image textures
- Too high can cause mode collapse (outputs look "typical" but not accurate)
- Too low has minimal effect

**Adjust if**:
- **Increase to 0.2-0.3**: If outputs are sharp but perceptually unrealistic
- **Decrease to 0.05**: If outputs look too smooth/averaged
- **Set to 0.0 or omit `--use_perceptual`**: Faster training, no perceptual constraint

**Note**: Adds computational cost (forward pass through feature extractor)

### Recommended Loss Configurations

**Standard U-HVED (Balanced)**:
```bash
python train.py \
  --recon_loss_type l1 \
  --recon_weight 0.4 \
  --kl_weight 0.1 \
  --orientation_weight 0.4 \
  --ssim_weight 0.1 \
  --use_ssim
```
**Why**: Balances pixel accuracy (L1), perceptual quality (SSIM), regularization (KL), and multi-task learning (orientation)

**High Quality (Perceptual Focus)**:
```bash
python train.py \
  --recon_loss_type l1 \
  --recon_weight 0.3 \
  --kl_weight 0.1 \
  --orientation_weight 0.3 \
  --ssim_weight 0.2 \
  --use_ssim \
  --use_perceptual \
  --perceptual_weight 0.2 \
  --perceptual_backend monai
```
**Why**: Emphasizes perceptual quality over pixel-wise accuracy, better for visual assessment

**Fast Training (Minimal Losses)**:
```bash
python train.py \
  --recon_loss_type l1 \
  --recon_weight 0.5 \
  --kl_weight 0.05 \
  --orientation_weight 0.4
```
**Why**: Removes expensive SSIM and perceptual losses, faster iterations

**SR-Only (Ablation)**:
```bash
python train.py \
  --recon_loss_type l1 \
  --recon_weight 0.8 \
  --kl_weight 0.1 \
  --orientation_weight 0.0 \
  --no_reconstruct_orientations
```
**Why**: No auxiliary task, all reconstruction weight on SR output

**Strict Regularization**:
```bash
python train.py \
  --recon_loss_type l1 \
  --recon_weight 0.4 \
  --kl_weight 0.3 \
  --orientation_weight 0.4 \
  --use_prior
```
**Why**: Higher KL weight keeps latent space tightly Gaussian, useful for generation/sampling

### U-Net Loss Components

For the U-Net baseline, the loss is simpler (no variational component, no orientation reconstruction):

**Total Loss Formula**:
```
L_total = α·L_L1 + β·L_SSIM + γ·L_LPIPS

Where:
  L_L1 = Mean absolute error between SR output and HR target
  L_SSIM = 1 - SSIM (structural similarity)
  L_LPIPS = 3D-LPIPS perceptual loss (optional)
```

**U-Net Loss Weights**:

```bash
python train_unet.py \
  --l1_weight 1.0 \              # L1 reconstruction loss
  --ssim_weight 1.0 \            # SSIM structural loss
  --use_lpips \                  # Enable 3D-LPIPS
  --lpips_weight 0.1 \           # Perceptual loss weight
  --lpips_backend monai          # 3D medical network backend
```

**Recommended values**:
- **`--l1_weight 1.0`**: Primary pixel-wise loss, balanced with SSIM
- **`--ssim_weight 1.0`**: Equal weight with L1 for perceptual quality
- **`--lpips_weight 0.1`**: Small perceptual loss contribution (optional)

**Why these values**:
- Equal L1 and SSIM weights balance pixel accuracy with structural similarity
- LPIPS at 0.1 adds perceptual quality without dominating
- Simpler than U-HVED (no KL, no orientation loss), easier to tune

**U-Net Configurations**:

**Standard (L1 + SSIM)**:
```bash
python train_unet.py \
  --l1_weight 1.0 \
  --ssim_weight 1.0
```

**With Perceptual Loss**:
```bash
python train_unet.py \
  --l1_weight 1.0 \
  --ssim_weight 1.0 \
  --use_lpips \
  --lpips_weight 0.1 \
  --lpips_backend monai
```

**L1-Only (Baseline)**:
```bash
python train_unet.py \
  --l1_weight 1.0 \
  --ssim_weight 0.0
```

### Loss Weight Tuning Guidelines

**General Principles**:

1. **Start with recommended defaults**: They work well for most cases
2. **Change one weight at a time**: Easier to understand effects
3. **Monitor validation metrics**: Track PSNR, SSIM, and perceptual quality
4. **Visual inspection matters**: Metrics don't tell the whole story
5. **Keep total reconstruction weight ~0.8-1.0**: Maintains primary training signal

**Common Issues and Solutions**:

| Problem | Likely Cause | Solution |
|---------|--------------|----------|
| Blurry outputs | Recon weight too low, SSIM/perceptual too high | Increase `--recon_weight` |
| Sharp but unrealistic | No perceptual loss, only L1/L2 | Add `--use_ssim` or `--use_perceptual` |
| KL vanishing (latents not used) | KL weight too high | Decrease `--kl_weight` to 0.05 |
| Posterior collapse (all similar) | KL weight too low | Increase `--kl_weight` to 0.2 |
| Unstable training | SSIM or perceptual weight too high | Decrease to 0.05-0.1 |
| Poor orientation reconstructions | Orientation weight too low | Increase `--orientation_weight` |
| SR quality poor with orientations | Orientation task dominating | Decrease `--orientation_weight`, increase `--recon_weight` |

**Hyperparameter Search** (optional):

Use Weights & Biases sweeps to explore loss weight combinations:
```bash
# Define sweep configuration
wandb sweep sweep_config.yaml

# Run sweep agents
wandb agent your-entity/your-project/sweep-id
```

## Data Augmentation & Simulation

### Resolution Randomization

```bash
python train.py \
  --model_dir ./models \
  --atlas_res 1.0 1.0 1.0 \           # HR image resolution (mm)
  --min_resolution 1.0 1.0 1.0 \      # Minimum (best) resolution
  --max_res_aniso 9.0 9.0 9.0         # Maximum (worst) anisotropic resolution

# Disable randomization (use fixed max_res_aniso)
python train.py --model_dir ./models --no_randomise_res
```

### MRI Artifact Simulation

Control the probability of various MRI artifacts:

```bash
python train.py \
  --model_dir ./models \
  --prob_motion 0.5 \                 # Motion ghosting
  --prob_spike 0.5 \                  # K-space spikes
  --prob_aliasing 0.02 \              # Aliasing/wrap-around
  --prob_bias_field 0.5 \             # B1 field inhomogeneity
  --prob_noise 0.8 \                  # Rician/Gaussian noise
  --no_intensity_aug                  # Disable intensity augmentation
```

## Advanced Training

### Full Configuration Example

```bash
python train.py \
  --hr_image_dir /path/to/hr/images \
  --val_image_dir /path/to/val/images \
  --model_dir ./models \
  --epochs 200 \
  --batch_size 4 \
  --learning_rate 1e-4 \
  --output_shape 128 128 128 \
  \
  # Model architecture
  --base_channels 32 \
  --num_scales 4 \
  --decoder_upsample_mode trilinear \
  --final_activation sigmoid \
  --use_prior \
  --use_encoder_outputs_as_skip \
  \
  # Loss configuration
  --recon_loss_type l1 \
  --recon_weight 0.4 \
  --kl_weight 0.1 \
  --ssim_weight 0.1 \
  --orientation_weight 0.4 \
  --use_ssim \
  \
  # Orientation dropout
  --orientation_dropout_prob 0.3 \
  --min_orientations 1 \
  \
  # Resolution simulation
  --atlas_res 1.0 1.0 1.0 \
  --min_resolution 1.0 1.0 1.0 \
  --max_res_aniso 9.0 9.0 9.0 \
  \
  # Artifact simulation
  --prob_motion 0.5 \
  --prob_spike 0.5 \
  --prob_aliasing 0.02 \
  --prob_bias_field 0.5 \
  --prob_noise 0.8 \
  \
  # Training optimization
  --mixed_precision fp16 \
  --gradient_accumulation_steps 2 \
  --max_grad_norm 1.0 \
  --seed 42 \
  \
  # Validation
  --val_interval 5 \
  --use_sliding_window_val \
  --val_patch_size 128 128 128 \
  --val_overlap 0.5 \
  \
  # Logging
  --use_wandb \
  --wandb_project uhved-sr \
  --wandb_entity your_team
```

### Training with CSV Dataset

```bash
python train.py \
  --csv_file /path/to/dataset.csv \
  --base_dir /path/to/images \
  --model_dir ./models \
  --mri_classes T1 T2 FLAIR \         # Filter by MRI sequence
  --acquisition_types 3D              # Filter by acquisition type (3D, 2D, or 'all')
```

## U-Net Baseline Model

In addition to U-HVED, this repository includes a 3D U-Net baseline for comparison. The U-Net uses concatenated LR stacks as input channels instead of separate encoders with Product of Gaussians fusion.

### Key Differences: U-Net vs U-HVED

| Feature | U-Net | U-HVED |
|---------|-------|---------|
| **Architecture** | Single encoder-decoder | Multi-modal encoders + fusion |
| **Input format** | Concatenated stacks (B, 3, D, H, W) | List of separate stacks |
| **Fusion method** | Channel concatenation | Product of Gaussians |
| **Latent space** | Deterministic | Variational (KL divergence) |
| **Orientation reconstruction** | No | Yes (auxiliary task) |
| **Missing views** | Not supported | Supported via orientation dropout |
| **Parameters** | ~5-10M | ~15-30M |
| **Training speed** | Faster | Slower |
| **Use case** | Baseline, all views available | Research, missing view handling |

### U-Net Training

**Basic training:**
```bash
python train_unet.py \
  --hr_image_dir /path/to/hr/images \
  --model_dir ./checkpoints_unet \
  --epochs 100 \
  --batch_size 2 \
  --learning_rate 1e-4
```

**Training with validation:**
```bash
python train_unet.py \
  --hr_image_dir /path/to/hr/images \
  --val_image_dir /path/to/val/images \
  --model_dir ./checkpoints_unet \
  --epochs 100 \
  --batch_size 4 \
  --val_interval 5 \
  --save_best_only
```

**Training with 2-stack variants:**
```bash
# Use only axial + coronal stacks
python train_unet.py \
  --hr_image_dir /path/to/hr/images \
  --model_dir ./checkpoints_unet_2stack \
  --use_stacks "01" \
  --epochs 100

# Available options: "all" (3 stacks), "01", "02", "12" (2 stacks)
```

**Full configuration:**
```bash
python train_unet.py \
  --hr_image_dir /path/to/hr/images \
  --val_image_dir /path/to/val/images \
  --model_dir ./checkpoints_unet \
  --epochs 200 \
  --batch_size 4 \
  --learning_rate 1e-4 \
  --output_shape 128 128 128 \
  \
  # Model architecture
  --nb_features 24 \
  --nb_levels 5 \
  --final_activation sigmoid \
  \
  # Loss configuration
  --l1_weight 1.0 \
  --ssim_weight 1.0 \
  --use_lpips \
  --lpips_weight 0.1 \
  --lpips_backend monai \
  \
  # Training optimization
  --mixed_precision fp16 \
  --gradient_accumulation_steps 2 \
  --max_grad_norm 1.0 \
  \
  # Validation
  --val_interval 5 \
  --use_sliding_window_val \
  --val_patch_size 128 128 128
```

### U-Net Testing (Inference)

**Single volume with stack generation:**
```bash
python test_unet.py \
  --input /path/to/hr_volume.nii.gz \
  --output /path/to/output_sr.nii.gz \
  --model checkpoints_unet/unet_best.pth
```

**Single volume with pre-existing stacks:**
```bash
python test_unet.py \
  --input_stacks axial.nii.gz coronal.nii.gz sagittal.nii.gz \
  --output output_sr.nii.gz \
  --model checkpoints_unet/unet_best.pth
```

**Batch processing:**
```bash
python test_unet.py \
  --input /path/to/hr_volumes/ \
  --output /path/to/sr_outputs/ \
  --model checkpoints_unet/unet_best.pth
```

**With sliding window for large volumes:**
```bash
python test_unet.py \
  --input /path/to/hr_volumes/ \
  --output /path/to/sr_outputs/ \
  --model checkpoints_unet/unet_best.pth \
  --use_sliding_window \
  --patch_size 96 96 96 \
  --overlap 0.5
```

### U-Net Evaluation

**Evaluate with stack generation:**
```bash
python evaluate_unet.py \
  --ground_truth /path/to/test_volumes/ \
  --output_dir ./evaluation_unet \
  --model checkpoints_unet/unet_best.pth \
  --compute_lpips \
  --lpips_backend monai \
  --save_sr_outputs \
  --verbose
```

**Evaluate with pre-existing LR stacks:**
```bash
python evaluate_unet.py \
  --ground_truth /path/to/hr_volumes/ \
  --input_lr_dir /path/to/lr_stacks/ \
  --output_dir ./evaluation_unet \
  --model checkpoints_unet/unet_best.pth \
  --track_memory
```

**With sliding window for large volumes:**
```bash
python evaluate_unet.py \
  --ground_truth /path/to/test_volumes/ \
  --output_dir ./evaluation_unet \
  --model checkpoints_unet/unet_best.pth \
  --use_sliding_window \
  --patch_size 128 128 128 \
  --overlap 0.5
```

**Evaluation Metrics:**

The U-Net evaluation script computes:
- **Standard metrics**: MAE, MSE, RMSE, PSNR, R², SSIM
- **Perceptual metrics**: 3D-LPIPS, R-LPIPS (using 3D medical imaging networks)
- **Performance**: Inference time, peak GPU memory usage
- **Outputs**: CSV, JSON, and formatted console summary

## U-HVED Evaluation

The `evaluate.py` script provides comprehensive evaluation of trained U-HVED models on test datasets, supporting multiple metrics, output formats, and orientation dropout testing.

### Basic Evaluation (Mode 1: HR Volumes Only)

Generate LR stacks from HR volumes and evaluate:

```bash
python evaluate.py \
  --input /path/to/test/hr/volumes \
  --ground_truth /path/to/test/hr/volumes \
  --model ./models/uhved_best.pth \
  --output_dir ./evaluation_results \
  --csv_output results.csv \
  --json_output results.json \
  --save_sr_outputs
```

### Evaluation with Pre-existing LR Stacks (Mode 2)

Use pre-existing LR stacks paired with HR ground truth:

```bash
python evaluate.py \
  --input_lr_dir /path/to/lr/stacks \
  --ground_truth /path/to/hr/volumes \
  --lr_stack_pattern "{case}_axial.nii.gz,{case}_coronal.nii.gz,{case}_sagittal.nii.gz" \
  --model ./models/uhved_best.pth \
  --output_dir ./evaluation_results \
  --csv_output results.csv \
  --save_sr_outputs
```

### Evaluation with Sliding Window (Large Volumes)

For volumes that don't fit in GPU memory:

```bash
python evaluate.py \
  --input /path/to/test/hr/volumes \
  --ground_truth /path/to/test/hr/volumes \
  --model ./models/uhved_best.pth \
  --output_dir ./evaluation_results \
  --use_sliding_window \
  --patch_size 128 128 128 \
  --overlap 0.5
```

### Orientation Dropout Testing

Test model robustness with missing orientations:

```bash
# Test all 7 orientation combinations
python evaluate.py \
  --input /path/to/test/hr/volumes \
  --ground_truth /path/to/test/hr/volumes \
  --model ./models/uhved_best.pth \
  --output_dir ./evaluation_results \
  --test_all_orientations \
  --csv_output orientation_dropout_results.csv

# Test specific combinations
python evaluate.py \
  --input /path/to/test/hr/volumes \
  --ground_truth /path/to/test/hr/volumes \
  --model ./models/uhved_best.pth \
  --output_dir ./evaluation_results \
  --orientation_configs "111,110,101,011"

# Test single orientation mask
python evaluate.py \
  --input /path/to/test/hr/volumes \
  --ground_truth /path/to/test/hr/volumes \
  --model ./models/uhved_best.pth \
  --output_dir ./evaluation_results \
  --orientation_mask 1 1 0
```

### Performance Profiling

Track inference time and GPU memory:

```bash
python evaluate.py \
  --input /path/to/test/hr/volumes \
  --ground_truth /path/to/test/hr/volumes \
  --model ./models/uhved_best.pth \
  --output_dir ./evaluation_results \
  --track_memory \
  --verbose
```

### Evaluation Metrics

The script computes the following metrics:

**Image Quality:**
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- PSNR (Peak Signal-to-Noise Ratio)
- R² (Coefficient of Determination)
- SSIM (Structural Similarity Index)

**Perceptual Quality (3D Medical Imaging Networks):**
- **3D-LPIPS** (3D Learned Perceptual Image Patch Similarity)
  - Uses proper 3D medical imaging encoders (not 2D slice-by-slice)
  - Backends: MedicalNet (3D ResNet), MONAI (3D models), or Models Genesis
  - Extracts hierarchical features from layer1, layer2, layer3, layer4
  - Lower is better (0 = identical)
  - Based on Zhang et al. (CVPR 2018) adapted for 3D volumes

- **R-LPIPS** (Robust LPIPS)
  - Adversarially robust perceptual similarity metric
  - Uses ensemble-based robustness for medical imaging
  - Based on Ghazanfari et al. (arXiv:2307.15157, 2023)
  - Robust to small perturbations and noise
  - Lower is better

**Performance:**
- Inference time per volume (seconds)
- Peak GPU memory usage (MB)

**Statistics:**
- Per-volume metrics
- Aggregate statistics: mean, std, median, min, max

### Output Formats

The evaluation script generates:

```
output_dir/
├── results.csv              # Per-volume metrics + aggregate statistics
├── results.json             # Structured JSON with metadata
├── evaluation.log           # Detailed execution log
├── sr_outputs/              # Super-resolved volumes (if --save_sr_outputs)
│   ├── case001_sr.nii.gz
│   └── ...
└── reconstructions/         # Orientation reconstructions (if --save_reconstructions)
    ├── case001_recon_axial.nii.gz
    ├── case001_recon_coronal.nii.gz
    └── ...
```

### Evaluation Parameters

**Data Mode Selection:**
- `--input`: HR volume file/directory (Mode 1: generate LR stacks)
- `--input_lr_dir`: LR stacks directory (Mode 2: use pre-existing stacks)
- `--ground_truth`: HR ground truth file/directory (required)
- `--lr_stack_pattern`: File pattern for matching LR stacks (Mode 2)

**Model & Device:**
- `--model`: Path to checkpoint file (.pth)
- `--device`: `cuda` or `cpu`

**Inference Options:**
- `--use_sliding_window`: Enable sliding window for large volumes
- `--patch_size`: Patch size [D H W] (default: 128 128 128)
- `--overlap`: Overlap ratio 0.0-1.0 (default: 0.5)

**Orientation Testing:**
- `--test_all_orientations`: Test all 7 combinations
- `--orientation_configs`: Specific configs (e.g., "111,110,101")
- `--orientation_mask`: Single mask [D H W] (e.g., 1 1 0)

**Output Options:**
- `--output_dir`: Results directory (required)
- `--csv_output`: CSV file path
- `--json_output`: JSON file path
- `--save_sr_outputs`: Save SR volumes as NIfTI
- `--save_reconstructions`: Save orientation reconstructions
- `--verbose`: Print detailed per-config results

**Performance:**
- `--track_memory`: Track GPU memory usage
- `--num_workers`: DataLoader workers (default: 4)

**Perceptual Metrics (3D-LPIPS & R-LPIPS):**
- `--compute_lpips`: Compute 3D-LPIPS and R-LPIPS metrics (default: True)
- `--no_lpips`: Disable LPIPS computation
- `--lpips_backend`: 3D network backend (`monai`, `medicalnet`, `models_genesis`) (default: monai)

### 3D-LPIPS and R-LPIPS Setup

The 3D perceptual metrics use existing PerceptualLoss3D infrastructure with 3D medical imaging networks:

**MONAI backend (default - recommended):**
```bash
pip install monai  # Already installed with requirements
```

**MedicalNet backend (optional):**
```bash
git clone https://github.com/Tencent/MedicalNet.git
cd MedicalNet
pip install -e .
```

**Models Genesis backend (optional):**
```bash
git clone https://github.com/MrGiovanni/ModelsGenesis.git
cd ModelsGenesis/pytorch
pip install -e .
```

To disable LPIPS metrics:
```bash
python evaluate.py --no_lpips [other args...]
```

### Example: Complete Evaluation Pipeline

```bash
# 1. Evaluate on test set with all metrics (including LPIPS)
python evaluate.py \
  --input /data/test/hr \
  --ground_truth /data/test/hr \
  --model ./models/uhved_best.pth \
  --output_dir ./eval_test \
  --csv_output test_metrics.csv \
  --json_output test_metrics.json \
  --save_sr_outputs \
  --track_memory \
  --verbose \
  --lpips_backend monai

# 2. Test orientation dropout robustness
python evaluate.py \
  --input /data/test/hr \
  --ground_truth /data/test/hr \
  --model ./models/uhved_dropout.pth \
  --output_dir ./eval_dropout \
  --test_all_orientations \
  --csv_output dropout_analysis.csv

# 3. Evaluate with sliding window (large volumes)
python evaluate.py \
  --input /data/test/hr_large \
  --ground_truth /data/test/hr_large \
  --model ./models/uhved_best.pth \
  --output_dir ./eval_large \
  --use_sliding_window \
  --patch_size 96 96 96 \
  --overlap 0.5 \
  --save_sr_outputs
```

## Model Variants

### Standard U-HVED (Default)

The standard U-HVED model maintains spatial resolution through the encoder-decoder architecture. It's designed for cases where the input LR stacks and output HR volume have similar spatial dimensions.

**Use when:** Your training data has LR and HR volumes of similar size (e.g., both 128³)

### U-HVED Lite

`UHVEDLite` is a lightweight version for faster training and inference:
- **Shared encoder/decoder** across orientations (fewer parameters)
- **Fewer scales** (typically 3 instead of 4)
- **No orientation reconstruction** branch
- **Smaller base channels** (default: 16 instead of 32)

**Use when:** You need faster training/inference or have limited GPU memory

**Example Usage:**
```python
from src.uhved import UHVEDLite

# Create lightweight model
model = UHVEDLite(
    num_orientations=3,
    in_channels=1,
    out_channels=1,
    base_channels=16,
    num_scales=3
)

output = model(lr_stacks, orientation_mask=None)
sr_volume = output['sr_output']
```

### U-HVED with Upscaling

`UHVEDWithUpscale` adds explicit upscaling layers at the end for higher super-resolution factors (2x, 4x, or 8x). This variant:
- Uses the base U-HVED to fuse multi-orientation information
- Applies pixel-shuffle 3D upscaling to increase spatial resolution
- Supports upscale factors: 2, 4, or 8

**Use when:** You need explicit spatial upscaling (e.g., 64³ → 128³, 64³ → 256³)

**Example Usage:**
```python
from src.uhved import UHVEDWithUpscale

# Create model with 4x upscaling
model = UHVEDWithUpscale(
    num_orientations=3,
    in_channels=1,
    out_channels=1,
    base_channels=32,
    num_scales=4,
    upscale_factor=4,  # 2x, 4x, or 8x
    final_activation='sigmoid'
)

# Input: List of 3 LR stacks, each (B, 1, D, H, W)
# Output: HR volume with dimensions multiplied by upscale_factor
output = model(lr_stacks, orientation_mask=None)
sr_volume = output['sr']  # Shape: (B, 1, D*4, H*4, W*4)
```

### Model Comparison

| Feature | Standard U-HVED | U-HVED Lite | U-HVED with Upscaling |
|---------|-----------------|-------------|----------------------|
| Encoder/Decoder | Independent | Shared | Independent |
| Base channels | 32 | 16 | 32 |
| Num scales | 4 | 3 | 4 |
| Spatial resolution | Maintains size | Maintains size | Explicit upscaling (2x/4x/8x) |
| Orientation reconstruction | ✓ | ✗ | ✗ |
| Use case | Best quality | Fast/lightweight | Large resolution gaps |
| Parameters | High | Low | High |
| Speed | Moderate | Fast | Slow |

## Model Architecture

### Data Flow

```
HR Volume (B, 1, D, H, W)
    ↓
Generate 3 Orthogonal LR Stacks:
  - Axial:    High-res in D, low-res in H,W
  - Coronal:  High-res in H, low-res in D,W
  - Sagittal: High-res in W, low-res in D,H
    ↓
Apply MRI Artifacts:
  - Bias field, motion, k-space spikes, noise
    ↓
MultiModalEncoder (3 independent encoders)
  ├─ Axial    → μ₀, log(σ₀²)
  ├─ Coronal  → μ₁, log(σ₁²)
  └─ Sagittal → μ₂, log(σ₂²)
    ↓
Product of Gaussians Fusion
  → Fused (μ, log(σ²))
    ↓
Sample z ~ N(μ, σ²)
    ↓
Decoder (with skip connections)
    ↓
SR Output (B, 1, D, H, W)
```

### Key Components

- **MultiModalEncoder**: Processes each orientation independently through U-Net style encoder
- **Product of Gaussians**: Fuses latent distributions by combining precision-weighted means
- **Decoder**: Progressively upsamples with skip connections from encoders
- **Orientation Reconstruction**: Auxiliary task to reconstruct input orientations

### Detailed Architecture Explanation

#### 1. Multi-Modal Encoder

The encoder processes each LR stack (axial, coronal, sagittal) independently through a U-Net style convolutional encoder:

**Input**: List of 3 tensors, each (B, 1, D, H, W)

**Processing**:
- Each orientation passes through its own encoder (or shared encoder if `share_encoder=True`)
- Encoder produces hierarchical features at multiple scales (typically 4 scales)
- At each scale, encoder outputs:
  - **μ (mu)**: Mean of latent distribution
  - **log(σ²) (logvar)**: Log-variance of latent distribution
  - **Features**: Intermediate convolutional features (for skip connections)

**Output**: For each orientation and each scale → (μ, log(σ²), features)

This variational formulation forces the network to learn a probabilistic representation of each orientation.

#### 2. Product of Gaussians Fusion

The key innovation of U-HVED is fusing multiple variational distributions using the **Product of Gaussians**:

**Mathematical Formula**:

Given N orientations, each providing a Gaussian distribution N(μᵢ, σᵢ²):

```
Precision (inverse variance): τᵢ = 1/σᵢ²

Fused precision: τ_fused = Σ τᵢ
Fused mean: μ_fused = (Σ μᵢ · τᵢ) / τ_fused
Fused variance: σ²_fused = 1/τ_fused
```

**Key Properties**:
- Product of Gaussians is also Gaussian (closed-form solution)
- Precision-weighted combination gives more weight to confident predictions (low variance)
- Naturally handles missing orientations by excluding them from the sum
- Result is a single fused latent distribution combining all available information

**Implementation**:
```python
for each orientation i:
    precision_i = 1 / (exp(logvar_i) + eps)
    precision_sum += precision_i
    weighted_mu_sum += mu_i * precision_i

posterior_var = 1 / precision_sum
posterior_mu = weighted_mu_sum * posterior_var
```

**With Prior (`--use_prior`)**:

When enabled, a learned prior distribution is added to the Product of Gaussians fusion:

```
τ_prior = 1/σ²_prior  (typically σ²_prior = 1.0)

τ_fused = Σ τᵢ + τ_prior
μ_fused = (Σ μᵢ · τᵢ + μ_prior · τ_prior) / τ_fused
```

**Effect**:
- Provides a regularization baseline when few orientations are available
- Particularly useful for orientation dropout training
- Prevents degenerate solutions when only 1 orientation is present
- Acts as a weak Gaussian prior N(0, 1) by default

**Without Prior**:
- Fusion uses only the available orientations
- More sensitive to missing orientations
- May produce unreliable reconstructions with heavy dropout

#### 3. Latent Sampling (Reparameterization Trick)

After fusion, we sample from the fused distribution:

**Training**: `z = μ_fused + σ_fused * ε`, where ε ~ N(0, 1)
**Inference**: `z = μ_fused` (deterministic, no sampling)

This allows backpropagation through the sampling operation (reparameterization trick).

#### 4. Decoder with Skip Connections

The decoder takes the latent samples and progressively upsamples to reconstruct the output.

**Skip Connections (`--use_encoder_outputs_as_skip`)**:

When **enabled**:
- Encoder features from each scale are averaged across all orientations:
  ```python
  skip_features[scale] = mean([encoder₀[scale], encoder₁[scale], encoder₂[scale]])
  ```
- These averaged features are concatenated with decoder features at matching scales:
  ```python
  decoder_input = concat([upsampled_features, skip_features], dim=channels)
  ```
- Then processed through residual blocks

**Benefits**:
- Preserves fine spatial details from input orientations
- Provides direct gradient paths from output to encoder
- Helps reconstruct high-frequency information lost in the latent bottleneck
- Similar to U-Net skip connections but averaged across orientations

When **disabled**:
- Decoder receives only the latent samples
- No direct connection to encoder features
- Relies entirely on information encoded in the latent space
- Typically produces smoother but less detailed outputs

**Decoder Architecture**:

```
Input: Latent samples z at each scale

For each scale (coarse to fine):
  1. Upsample by 2x (trilinear/transpose/pixelshuffle)
  2. If skip connections enabled:
       Concatenate upsampled features with encoder skip features
  3. Process through residual blocks
  4. Output features for next scale

Final: 1x1x1 convolution → final activation (sigmoid/tanh/none)
```

#### 5. Orientation Reconstruction Decoders

**With Orientation Reconstruction (Default)**:

Uses `MultiOutputDecoder` which creates:
- **1 SR Decoder**: Produces the super-resolved output
- **N Orientation Decoders**: Reconstruct each input orientation (axial, coronal, sagittal)

All decoders:
- Share the same latent samples z
- Can share decoder weights if `share_decoder=True`
- Receive skip connections if `--use_encoder_outputs_as_skip` is enabled

**Architecture**:
```
                    ┌─→ SR Decoder ──────────→ SR Output (HR volume)
                    │
Latent z (+ skips) ─┼─→ Decoder₀ ──────────→ Reconstruction₀ (axial)
                    │
                    ├─→ Decoder₁ ──────────→ Reconstruction₁ (coronal)
                    │
                    └─→ Decoder₂ ──────────→ Reconstruction₂ (sagittal)
```

**Skip Connection Flow with Orientation Reconstruction**:
```
Encoder₀ (axial)    ┐
Encoder₁ (coronal)  ├─→ Average features ──→ Skip connections
Encoder₂ (sagittal) ┘                             │
                                                   ↓
                                            ┌──────┴──────┐
                                            ↓             ↓
                                      SR Decoder    Orientation Decoders
```

**Training Loss**:
```
Total Loss = α·SR_loss + β·KL_loss + γ·Orientation_loss

Where:
  SR_loss = L1(SR_output, HR_target)
  KL_loss = KL(q(z|orientations) || p(z))  # Regularization
  Orientation_loss = mean([L1(recon_i, input_i) for i in orientations])
```

**Without Orientation Reconstruction (`--no_reconstruct_orientations`)**:

Uses simple `ConvDecoder` which creates only:
- **1 SR Decoder**: Produces the super-resolved output only

**Architecture**:
```
Latent z (+ skips) ─→ SR Decoder ──────────→ SR Output (HR volume)
```

**Skip Connection Flow without Orientation Reconstruction**:
```
Encoder₀ (axial)    ┐
Encoder₁ (coronal)  ├─→ Average features ──→ Skip connections
Encoder₂ (sagittal) ┘                             │
                                                   ↓
                                              SR Decoder only
```

**Training Loss**:
```
Total Loss = α·SR_loss + β·KL_loss

Where:
  SR_loss = L1(SR_output, HR_target)
  KL_loss = KL(q(z|orientations) || p(z))
  Orientation_loss = 0.0  (disabled)
```

**Key Differences**:
- **Parameters**: ~30-40% fewer without orientation decoders
- **Training speed**: ~1.5-2x faster (fewer backward passes)
- **Memory**: Lower GPU memory usage
- **Regularization**: Less regularization (no auxiliary reconstruction task)
- **Skip connections**: Still function the same way - averaged encoder features go to SR decoder

**When to disable orientation reconstruction**:
- Ablation studies to measure contribution of auxiliary task
- Limited GPU memory
- Faster training/inference required
- Only care about SR quality, not orientation reconstructions

### Architecture Configuration Examples

**Standard U-HVED (with prior and skip connections)**:
```bash
python train.py \
  --use_prior \                          # Include prior in fusion
  --use_encoder_outputs_as_skip \        # Use averaged encoder features as skips
  --reconstruct_orientations             # Include orientation decoders (default)
```

**Ablation: No skip connections**:
```bash
python train.py \
  --use_prior \
  --no_encoder_outputs_as_skip           # Disable skip connections
```

**Ablation: No prior**:
```bash
python train.py \
  --no_prior \                           # Disable prior in fusion
  --use_encoder_outputs_as_skip
```

**Ablation: SR-only (no orientation reconstruction)**:
```bash
python train.py \
  --use_prior \
  --use_encoder_outputs_as_skip \
  --no_reconstruct_orientations \        # Only SR decoder
  --recon_weight 0.8 \                   # Adjust loss weights
  --orientation_weight 0.0
```

**Minimal configuration (no prior, no skips, SR-only)**:
```bash
python train.py \
  --no_prior \
  --no_encoder_outputs_as_skip \
  --no_reconstruct_orientations
```

## Monitoring Training

### Local Tensorboard

Training metrics are automatically logged to TensorBoard:

```bash
tensorboard --logdir ./models/your_run_name
```

### Weights & Biases

```bash
python train.py \
  --model_dir ./models \
  --use_wandb \
  --wandb_project uhved-sr \
  --wandb_entity your_team \
  --wandb_run_name experiment_name
```

## Key Parameters Reference

### Model Architecture
- `--base_channels`: Base feature channels (default: 32)
- `--num_scales`: Number of hierarchical scales (default: 4)
- `--decoder_upsample_mode`: Upsampling strategy: `trilinear`, `transpose`, `pixelshuffle`
- `--final_activation`: Output activation: `sigmoid`, `tanh`, `none`
- `--use_prior`: Include learned prior in fusion
- `--use_encoder_outputs_as_skip`: Use encoder features as skip connections
- `--no_reconstruct_orientations`: Disable orientation reconstruction (SR decoder only, for ablation studies)

### Loss Configuration
- `--recon_loss_type`: Reconstruction loss: `l1`, `l2`, `charbonnier`
- `--recon_weight`: Reconstruction loss weight (default: 0.4)
- `--kl_weight`: KL divergence weight (default: 0.1)
- `--ssim_weight`: SSIM loss weight (default: 0.1)
- `--orientation_weight`: Orientation reconstruction weight (default: 0.4)
- `--perceptual_weight`: Perceptual loss weight (default: 0.1)

### Orientation Dropout
- `--orientation_dropout_prob`: Random dropout probability (0.0-1.0)
- `--min_orientations`: Minimum views to keep (1-3)
- `--drop_orientations`: Specific orientations to drop (0, 1, 2)

### Training
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 1)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--mixed_precision`: Mixed precision: `no`, `fp16`, `bf16`
- `--gradient_accumulation_steps`: Gradient accumulation steps (default: 1)
- `--max_grad_norm`: Gradient clipping max norm (default: 1.0)

### Validation
- `--val_interval`: Validate every N epochs (default: 1)
- `--use_sliding_window_val`: Use sliding window inference for validation
- `--val_patch_size`: Patch size for sliding window (default: [128, 128, 128])
- `--val_overlap`: Overlap ratio for sliding window (default: 0.5)

## File Structure

```
MultiSynth/
├── src/
│   ├── data.py          # Data generation and augmentation
│   ├── encoder.py       # Multi-modal encoder (U-HVED)
│   ├── decoder.py       # Decoder with multiple upsample modes (U-HVED)
│   ├── fusion.py        # Product of Gaussians fusion (U-HVED)
│   ├── losses.py        # Loss functions (SSIM, 3D-LPIPS, perceptual)
│   ├── uhved.py         # Main U-HVED model
│   ├── unet.py          # 3D U-Net baseline model
│   └── utils.py         # Utilities and config saving
├── train.py             # U-HVED training script
├── test.py              # U-HVED inference script
├── evaluate.py          # U-HVED evaluation script
├── train_unet.py        # U-Net training script (baseline)
├── test_unet.py         # U-Net inference script
├── evaluate_unet.py     # U-Net evaluation script
├── example.py           # Demo script
├── requirements.txt     # Dependencies
└── README.md            # This file
```

## Requirements

```bash
pip install -r requirements.txt
```

Main dependencies:
- PyTorch
- MONAI
- NumPy
- SimpleITK
- (Optional) Weights & Biases

## Citation

If you use this code, please cite:

```bibtex
@article{uhved2024,
  title={U-HVED: Hetero-Modal Variational Encoder-Decoder for Medical Image Super-Resolution},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```
