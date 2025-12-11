"""
Loss Functions for U-HVED Super-Resolution

The training objective combines:
1. Reconstruction loss: How well the network reconstructs the target
2. KL divergence: Regularization for the variational latent space
3. Perceptual loss: High-level feature matching for better visual quality
4. Adversarial loss (optional): For GAN-based training

For super-resolution, we adapt the original U-HVED losses:
- Instead of segmentation loss, we use image reconstruction loss
- KL divergence remains the same (regularizes the shared latent space)
- Added perceptual and adversarial losses for better SR quality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Literal
from torchvision import models
import warnings

class KLDivergence(nn.Module):
    """
    KL divergence between posterior q(z|x) and prior p(z).

    For standard VAE with N(0,1) prior:
    KL(q||p) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))

    Supports multi-scale KL computation for hierarchical VAE.
    """

    def __init__(self, reduction: str = 'mean'):
        """
        Args:
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence against N(0,1) prior.

        Args:
            mu: Posterior mean
            logvar: Posterior log-variance

        Returns:
            KL divergence loss
        """
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

        if self.reduction == 'mean':
            return kl.mean()
        elif self.reduction == 'sum':
            return kl.sum()
        else:
            return kl

    def multi_scale(
        self,
        posteriors: List[Tuple[torch.Tensor, torch.Tensor]],
        weights: Optional[List[float]] = None
    ) -> torch.Tensor:
        """
        Compute weighted sum of KL divergence across scales.

        Args:
            posteriors: List of (mu, logvar) tuples at each scale
            weights: Optional weights for each scale (default: equal weighting)

        Returns:
            Total KL loss
        """
        if weights is None:
            weights = [1.0] * len(posteriors)

        total_kl = 0.0
        for (mu, logvar), w in zip(posteriors, weights):
            total_kl = total_kl + w * self.forward(mu, logvar)

        return total_kl


class ReconstructionLoss(nn.Module):
    """
    Reconstruction loss for super-resolution.

    Supports multiple loss types:
    - L1 (MAE): Good for preserving edges
    - L2 (MSE): Standard choice
    - Charbonnier: Differentiable approximation of L1
    """

    def __init__(
        self,
        loss_type: str = 'l1',
        reduction: str = 'mean',
        eps: float = 1e-6
    ):
        """
        Args:
            loss_type: 'l1', 'l2', or 'charbonnier'
            reduction: 'mean' or 'sum'
            eps: Epsilon for Charbonnier loss
        """
        super().__init__()

        self.loss_type = loss_type
        self.reduction = reduction
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute reconstruction loss.

        Args:
            pred: Predicted image
            target: Ground truth image

        Returns:
            Reconstruction loss
        """
        if self.loss_type == 'l1':
            loss = F.l1_loss(pred, target, reduction=self.reduction)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(pred, target, reduction=self.reduction)
        elif self.loss_type == 'charbonnier':
            diff = pred - target
            loss = torch.sqrt(diff.pow(2) + self.eps ** 2)
            if self.reduction == 'mean':
                loss = loss.mean()
            elif self.reduction == 'sum':
                loss = loss.sum()
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return loss


class AdversarialLoss(nn.Module):
    """
    Adversarial loss for GAN training.

    Supports multiple GAN loss types:
    - Vanilla (BCE)
    - LSGAN (MSE)
    - WGAN-GP
    - Hinge
    """

    def __init__(
        self,
        loss_type: str = 'vanilla',
        real_label: float = 1.0,
        fake_label: float = 0.0
    ):
        """
        Args:
            loss_type: 'vanilla', 'lsgan', 'wgan', or 'hinge'
            real_label: Label value for real samples
            fake_label: Label value for fake samples
        """
        super().__init__()

        self.loss_type = loss_type
        self.real_label = real_label
        self.fake_label = fake_label

        if loss_type == 'vanilla':
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss_type == 'lsgan':
            self.criterion = nn.MSELoss()
        else:
            self.criterion = None

    def forward(
        self,
        pred: torch.Tensor,
        is_real: bool
    ) -> torch.Tensor:
        """
        Compute adversarial loss.

        Args:
            pred: Discriminator output
            is_real: Whether pred should be classified as real

        Returns:
            Adversarial loss
        """
        if self.loss_type in ['vanilla', 'lsgan']:
            target_val = self.real_label if is_real else self.fake_label
            target = torch.full_like(pred, target_val)
            return self.criterion(pred, target)

        elif self.loss_type == 'wgan':
            if is_real:
                return -pred.mean()
            else:
                return pred.mean()

        elif self.loss_type == 'hinge':
            if is_real:
                return F.relu(1.0 - pred).mean()
            else:
                return F.relu(1.0 + pred).mean()

        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


class PerceptualLoss3D(nn.Module):
    """
    3D Perceptual loss using pretrained 3D medical imaging models.

    Supports three backend options:
    1. 'medicalnet': 3D ResNet from MedicalNet (Tencent)
    2. 'monai': Self-supervised models from MONAI
    3. 'models_genesis': Self-supervised U-Net from Models Genesis

    Each backend extracts hierarchical features from 3D volumes and
    computes feature-space reconstruction loss.
    """

    def __init__(
        self,
        backend: Literal['medicalnet', 'monai', 'models_genesis'] = 'medicalnet',
        model_depth: int = 18,
        feature_layers: List[str] = None,
        weights: List[float] = None,
        pretrained: bool = True,
        normalize_input: bool = True,
        freeze_backbone: bool = True
    ):
        """
        Args:
            backend: Which 3D pretrained model to use
                - 'medicalnet': 3D ResNet (10, 18, 34, 50, 101)
                - 'monai': MONAI's pretrained models
                - 'models_genesis': Models Genesis pretrained U-Net
            model_depth: Depth of ResNet for MedicalNet (10, 18, 34, 50, 101)
            feature_layers: Which layers to extract features from
            weights: Weights for each layer's loss contribution
            pretrained: Whether to load pretrained weights
            normalize_input: Whether to normalize input to expected range
            freeze_backbone: Whether to freeze backbone parameters
        """
        super().__init__()

        self.backend = backend
        self.normalize_input = normalize_input

        if feature_layers is None:
            feature_layers = ['layer1', 'layer2', 'layer3']
        if weights is None:
            weights = [1.0] * len(feature_layers)

        self.feature_layers = feature_layers
        self.weights = weights

        # Initialize backbone based on selected backend
        if backend == 'medicalnet':
            self.backbone = self._build_medicalnet_backbone(model_depth, pretrained)
        elif backend == 'monai':
            self.backbone = self._build_monai_backbone(pretrained)
        elif backend == 'models_genesis':
            self.backbone = self._build_models_genesis_backbone(pretrained)
        else:
            raise ValueError(f"Unknown backend: {backend}. Choose from ['medicalnet', 'monai', 'models_genesis']")

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Hook storage for intermediate features
        self.features = {}
        self._register_hooks()

    def _build_medicalnet_backbone(self, depth: int, pretrained: bool) -> nn.Module:
        """
        Build MedicalNet 3D ResNet backbone.

        MedicalNet provides 3D ResNet models pretrained on medical imaging datasets.
        GitHub: https://github.com/Tencent/MedicalNet
        """
        try:
            # Try to import MedicalNet models
            # Note: User needs to install MedicalNet separately
            from medicalnet import resnet

            if depth == 10:
                model = resnet.resnet10(sample_input_D=64, sample_input_H=64, sample_input_W=64, num_seg_classes=1)
            elif depth == 18:
                model = resnet.resnet18(sample_input_D=64, sample_input_H=64, sample_input_W=64, num_seg_classes=1)
            elif depth == 34:
                model = resnet.resnet34(sample_input_D=64, sample_input_H=64, sample_input_W=64, num_seg_classes=1)
            elif depth == 50:
                model = resnet.resnet50(sample_input_D=64, sample_input_H=64, sample_input_W=64, num_seg_classes=1)
            elif depth == 101:
                model = resnet.resnet101(sample_input_D=64, sample_input_H=64, sample_input_W=64, num_seg_classes=1)
            else:
                raise ValueError(f"Unsupported depth {depth}. Choose from [10, 18, 34, 50, 101]")

            if pretrained:
                # User needs to download pretrained weights from MedicalNet GitHub
                warnings.warn(
                    "Pretrained MedicalNet weights need to be downloaded manually from "
                    "https://github.com/Tencent/MedicalNet/tree/master/pretrain\n"
                    "Set pretrained=False or load weights manually with model.load_state_dict()"
                )

            return model

        except ImportError:
            raise ImportError(
                "MedicalNet not found. Install it with:\n"
                "  git clone https://github.com/Tencent/MedicalNet.git\n"
                "  cd MedicalNet\n"
                "  pip install -e .\n"
                "Or choose a different backend: 'monai' or 'models_genesis'"
            )

    def _build_monai_backbone(self, pretrained: bool) -> nn.Module:
        """
        Build MONAI pretrained 3D backbone.

        MONAI provides various pretrained 3D models including ResNet, DenseNet, etc.
        """
        try:
            from monai.networks.nets import ResNet
            from monai.networks.layers import Norm

            # Create 3D ResNet50
            model = ResNet(
                spatial_dims=3,
                n_input_channels=1,
                num_classes=1,
                block='bottleneck',
                layers=[3, 4, 6, 3],  # ResNet-50 configuration
                block_inplanes=[64, 128, 256, 512],
                norm=Norm.BATCH,
            )

            if pretrained:
                # MONAI pretrained weights can be loaded from MONAI Model Zoo
                warnings.warn(
                    "For pretrained MONAI models, use MONAI Model Zoo:\n"
                    "  from monai.bundle import download\n"
                    "  download(name='model_name', bundle_dir='./pretrained')\n"
                    "See: https://monai.io/model-zoo.html"
                )

            return model

        except ImportError:
            raise ImportError(
                "MONAI not found. Install it with:\n"
                "  pip install monai\n"
                "Or choose a different backend: 'medicalnet' or 'models_genesis'"
            )

    def _build_models_genesis_backbone(self, pretrained: bool) -> nn.Module:
        """
        Build Models Genesis pretrained 3D U-Net backbone.

        Models Genesis provides self-supervised pretrained models for medical imaging.
        GitHub: https://github.com/MrGiovanni/ModelsGenesis
        """
        try:
            # Try to import Models Genesis
            from models_genesis import UNet3D

            model = UNet3D(in_channels=1, out_channels=1)

            if pretrained:
                warnings.warn(
                    "Pretrained Models Genesis weights need to be downloaded from:\n"
                    "https://github.com/MrGiovanni/ModelsGenesis/tree/master/pytorch\n"
                    "Download Genesis_Chest_CT.pt or other pretrained weights"
                )

            return model

        except ImportError:
            # Fallback: Create a simple 3D U-Net encoder
            warnings.warn(
                "Models Genesis not found. Using a simple 3D encoder instead.\n"
                "For better performance, install Models Genesis:\n"
                "  git clone https://github.com/MrGiovanni/ModelsGenesis.git\n"
                "  cd ModelsGenesis/pytorch\n"
                "  pip install -e .\n"
                "Or choose a different backend: 'medicalnet' or 'monai'"
            )
            return self._build_simple_3d_encoder()

    def _build_simple_3d_encoder(self) -> nn.Module:
        """Fallback: Simple 3D encoder if specialized models unavailable."""
        return nn.Sequential(
            # Layer 1
            nn.Conv3d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            # Layer 2
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            # Layer 3
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
        )

    def _register_hooks(self):
        """Register forward hooks to extract intermediate features."""
        def get_hook(name):
            def hook(module, input, output):
                self.features[name] = output
            return hook

        # Register hooks based on backend
        if self.backend in ['medicalnet', 'monai']:
            # For ResNet-like architectures
            for name, module in self.backbone.named_modules():
                if any(layer_name in name for layer_name in self.feature_layers):
                    module.register_forward_hook(get_hook(name))
        else:
            # For U-Net or simple encoder
            for idx, (name, module) in enumerate(self.backbone.named_children()):
                if f'layer{idx//3 + 1}' in self.feature_layers:
                    module.register_forward_hook(get_hook(f'layer{idx//3 + 1}'))

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input to expected range for medical imaging models."""
        # Most medical imaging models expect inputs in [0, 1] or [-1, 1]
        # Adjust based on model expectations
        if x.min() < 0:
            # Already in [-1, 1], normalize to [0, 1]
            x = (x + 1) / 2

        # Clip to valid range
        x = torch.clamp(x, 0, 1)

        return x

    def extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract features from input volume."""
        if self.normalize_input:
            x = self._normalize(x)

        # Clear previous features
        self.features = {}

        # Forward pass (hooks will populate self.features)
        _ = self.backbone(x)

        return self.features

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute 3D perceptual loss.

        Args:
            pred: Predicted volume (B, C, D, H, W)
            target: Ground truth volume (B, C, D, H, W)

        Returns:
            Perceptual loss (scalar)
        """
        # Extract features from both volumes
        pred_features = self.extract_features(pred)
        target_features = self.extract_features(target)

        # Compute loss at each layer
        loss = 0.0
        for layer_name, weight in zip(self.feature_layers, self.weights):
            # Find matching features
            pred_feat = None
            target_feat = None

            for name, feat in pred_features.items():
                if layer_name in name:
                    pred_feat = feat
                    break

            for name, feat in target_features.items():
                if layer_name in name:
                    target_feat = feat
                    break

            if pred_feat is not None and target_feat is not None:
                # L1 loss in feature space
                loss = loss + weight * F.l1_loss(pred_feat, target_feat)

        return loss


class SSIM3DLoss(nn.Module):
    """
    3D Structural Similarity Index (SSIM) loss using MONAI.

    MONAI provides optimized 3D SSIM implementation that properly
    handles volumetric medical imaging data.
    """

    def __init__(
        self,
        spatial_dims: int = 3,
    ):
        """
        Args:
            spatial_dims: Number of spatial dimensions (3 for 3D volumes)
        """
        super().__init__()

        try:
            from monai.losses import SSIMLoss
            self.ssim_loss = SSIMLoss(
                spatial_dims=spatial_dims,

            )

        except ImportError:
            raise ImportError(
                "MONAI not found. Install it with:\n"
                "  pip install monai"
            )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute 3D SSIM loss.

        Args:
            pred: Predicted volume (B, C, D, H, W)
            target: Ground truth volume (B, C, D, H, W)

        Returns:
            SSIM loss (scalar)
        """
        return self.ssim_loss(pred, target)


class UHVEDLoss(nn.Module):
    """
    Combined 3D loss function for U-HVED Super-Resolution on volumetric data.

    Total loss = recon_weight * L_recon
               + kl_weight * L_kl
               + perceptual_weight * L_perceptual_3d
               + ssim_weight * L_ssim_3d
               + orientation_weight * L_orientation_recon

    Features:
    - Supports multiple 3D perceptual loss backends
    - Uses MONAI's SSIM3D
    - Maintains all original loss components
    - KL annealing support
    """

    def __init__(
        self,
        recon_loss_type: str = 'l1',
        recon_weight: float = 1.0,
        kl_weight: float = 0.001,
        perceptual_weight: float = 0.1,
        ssim_weight: float = 0.0,
        orientation_weight: float = 0.5,
        use_perceptual: bool = True,
        use_ssim: bool = False,
        perceptual_backend: Literal['medicalnet', 'monai', 'models_genesis'] = 'medicalnet',
        perceptual_model_depth: int = 18,
        kl_annealing: bool = True,
        kl_anneal_steps: int = 10000
    ):
        """
        Args:
            recon_loss_type: Type of reconstruction loss ('l1', 'l2', 'charbonnier')
            recon_weight: Weight for main reconstruction loss
            kl_weight: Weight for KL divergence
            perceptual_weight: Weight for 3D perceptual loss
            ssim_weight: Weight for 3D SSIM loss
            orientation_weight: Weight for orientation reconstruction loss
            use_perceptual: Whether to use 3D perceptual loss
            use_ssim: Whether to use 3D SSIM loss
            perceptual_backend: Which 3D model to use for perceptual loss
                - 'medicalnet': 3D ResNet from MedicalNet
                - 'monai': MONAI pretrained models
                - 'models_genesis': Models Genesis U-Net
            perceptual_model_depth: Depth of ResNet for MedicalNet (10, 18, 34, 50, 101)
            kl_annealing: Whether to anneal KL weight
            kl_anneal_steps: Number of steps for KL annealing
        """
        super().__init__()

        self.recon_weight = recon_weight
        self.kl_weight = kl_weight
        self.perceptual_weight = perceptual_weight
        self.ssim_weight = ssim_weight
        self.orientation_weight = orientation_weight
        self.kl_annealing = kl_annealing
        self.kl_anneal_steps = kl_anneal_steps

        # Reconstruction loss (works with any dimension)
        self.recon_loss = ReconstructionLoss(loss_type=recon_loss_type)

        # KL divergence (works with any dimension)
        self.kl_loss = KLDivergence()

        # 3D Perceptual loss
        if use_perceptual:
            try:
                self.perceptual_loss = PerceptualLoss3D(
                    backend=perceptual_backend,
                    model_depth=perceptual_model_depth,
                    pretrained=True
                )
                print(f"✓ 3D Perceptual Loss initialized with backend: {perceptual_backend}")
            except Exception as e:
                warnings.warn(f"Failed to initialize 3D perceptual loss: {e}")
                self.perceptual_loss = None
        else:
            self.perceptual_loss = None

        # 3D SSIM loss
        if use_ssim:
            try:
                self.ssim_loss = SSIM3DLoss()
                print("✓ 3D SSIM Loss initialized using MONAI")
            except Exception as e:
                warnings.warn(f"Failed to initialize 3D SSIM loss: {e}")
                self.ssim_loss = None
        else:
            self.ssim_loss = None

        # Current step for KL annealing
        self.register_buffer('current_step', torch.tensor(0))

    def get_kl_weight(self) -> float:
        """Get current KL weight with optional annealing."""
        if not self.kl_annealing:
            return self.kl_weight

        # Linear annealing from 0 to kl_weight
        progress = min(self.current_step.item() / self.kl_anneal_steps, 1.0)
        return self.kl_weight * progress

    def forward(
        self,
        sr_output: torch.Tensor,
        sr_target: torch.Tensor,
        posteriors: List[Tuple[torch.Tensor, torch.Tensor]],
        orientation_outputs: Optional[List[torch.Tensor]] = None,
        orientation_targets: Optional[List[torch.Tensor]] = None,
        return_components: bool = False
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        """
        Compute total 3D loss.

        Args:
            sr_output: Super-resolved output (B, C, D, H, W)
            sr_target: Ground truth high-resolution volume (B, C, D, H, W)
            posteriors: List of (mu, logvar) from encoder
            orientation_outputs: Reconstructed orientations (optional)
            orientation_targets: Target orientations (optional)
            return_components: If True, return dict of individual losses

        Returns:
            Total loss (or dict of losses if return_components=True)
        """
        losses = {}

        # Main reconstruction loss (L1/L2/Charbonnier)
        losses['reconstruction'] = self.recon_loss(sr_output, sr_target) * self.recon_weight

        # KL divergence (multi-scale)
        kl_weight = self.get_kl_weight()
        losses['kl'] = self.kl_loss.multi_scale(posteriors) * kl_weight

        # 3D Perceptual loss
        if self.perceptual_loss is not None:
            try:
                losses['perceptual'] = self.perceptual_loss(sr_output, sr_target) * self.perceptual_weight
            except Exception as e:
                warnings.warn(f"Perceptual loss computation failed: {e}")
                losses['perceptual'] = torch.tensor(0.0, device=sr_output.device)
        else:
            losses['perceptual'] = torch.tensor(0.0, device=sr_output.device)

        # 3D SSIM loss
        if self.ssim_loss is not None:
            try:
                losses['ssim'] = self.ssim_loss(sr_output, sr_target) * self.ssim_weight
            except Exception as e:
                warnings.warn(f"SSIM loss computation failed: {e}")
                losses['ssim'] = torch.tensor(0.0, device=sr_output.device)
        else:
            losses['ssim'] = torch.tensor(0.0, device=sr_output.device)

        # orientation reconstruction loss
        if orientation_outputs is not None and orientation_targets is not None:
            mod_loss = 0.0
            for mod_out, mod_target in zip(orientation_outputs, orientation_targets):
                if mod_target is not None:
                    mod_loss = mod_loss + self.recon_loss(mod_out, mod_target)
            losses['orientation'] = mod_loss * self.orientation_weight
        else:
            losses['orientation'] = torch.tensor(0.0, device=sr_output.device)

        # Total loss
        total_loss = sum(losses.values())

        # Update step counter
        if self.training:
            self.current_step += 1

        if return_components:
            losses['total'] = total_loss
            return losses

        return total_loss


# Convenience function to create 3D loss
def create_uhved_loss(
    backend: str = 'medicalnet',
    use_perceptual: bool = True,
    use_ssim: bool = False,
    **kwargs
) -> UHVEDLoss:
    """
    Factory function to create UHVEDLoss with sensible defaults.

    Args:
        backend: Which 3D backbone to use ('medicalnet', 'monai', 'models_genesis')
        use_perceptual: Whether to use 3D perceptual loss
        use_ssim: Whether to use 3D SSIM loss
        **kwargs: Additional arguments for UHVEDLoss

    Returns:
        UHVEDLoss instance

    Example:
        >>> # Basic usage with L1 + KL only
        >>> loss_fn = create_uhved_loss(use_perceptual=False, use_ssim=False)

        >>> # With MedicalNet perceptual loss
        >>> loss_fn = create_uhved_loss(backend='medicalnet', use_perceptual=True)

        >>> # With MONAI perceptual + SSIM
        >>> loss_fn = create_uhved_loss(
        ...     backend='monai',
        ...     use_perceptual=True,
        ...     use_ssim=True,
        ...     ssim_weight=0.1
        ... )
    """
    return UHVEDLoss(
        perceptual_backend=backend,
        use_perceptual=use_perceptual,
        use_ssim=use_ssim,
        **kwargs
    )
