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
from typing import List, Tuple, Optional, Dict
from torchvision import models


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


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using pre-trained VGG features.

    Compares high-level feature representations instead of pixel values,
    leading to more visually pleasing super-resolution results.
    """

    def __init__(
        self,
        layers: List[str] = None,
        weights: List[float] = None,
        normalize_input: bool = True,
        resize_input: bool = False
    ):
        """
        Args:
            layers: VGG layers to use (default: ['relu2_2', 'relu3_3', 'relu4_3'])
            weights: Weights for each layer loss
            normalize_input: Whether to normalize input to VGG
            resize_input: Whether to resize input to 224x224
        """
        super().__init__()

        if layers is None:
            layers = ['relu2_2', 'relu3_3', 'relu4_3']
        if weights is None:
            weights = [1.0] * len(layers)

        self.layers = layers
        self.weights = weights
        self.normalize_input = normalize_input
        self.resize_input = resize_input

        # Load VGG
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)

        # Layer name to index mapping
        layer_map = {
            'relu1_1': 1, 'relu1_2': 3,
            'relu2_1': 6, 'relu2_2': 8,
            'relu3_1': 11, 'relu3_2': 13, 'relu3_3': 15, 'relu3_4': 17,
            'relu4_1': 20, 'relu4_2': 22, 'relu4_3': 24, 'relu4_4': 26,
            'relu5_1': 29, 'relu5_2': 31, 'relu5_3': 33, 'relu5_4': 35
        }

        # Extract required layers
        max_idx = max(layer_map[l] for l in layers)
        self.vgg = nn.Sequential(*list(vgg.features.children())[:max_idx + 1])

        # Freeze VGG
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.layer_indices = [layer_map[l] for l in layers]

        # VGG normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input to VGG expected range."""
        # Convert from [-1, 1] to [0, 1] if needed
        if x.min() < 0:
            x = (x + 1) / 2

        # Ensure 3 channels
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # VGG normalization
        x = (x - self.mean) / self.std

        return x

    def _extract_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract features at specified layers."""
        if self.normalize_input:
            x = self._normalize(x)

        if self.resize_input:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        features = []
        for idx, layer in enumerate(self.vgg):
            x = layer(x)
            if idx in self.layer_indices:
                features.append(x)

        return features

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute perceptual loss.

        Args:
            pred: Predicted image
            target: Ground truth image

        Returns:
            Perceptual loss
        """
        pred_features = self._extract_features(pred)
        target_features = self._extract_features(target)

        loss = 0.0
        for pf, tf, w in zip(pred_features, target_features, self.weights):
            loss = loss + w * F.l1_loss(pf, tf)

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


class UHVEDLoss(nn.Module):
    """
    Combined loss function for U-HVED Super-Resolution.

    Total loss = recon_weight * L_recon
               + kl_weight * L_kl
               + perceptual_weight * L_perceptual
               + modality_weight * L_modality_recon
    """

    def __init__(
        self,
        recon_loss_type: str = 'l1',
        recon_weight: float = 1.0,
        kl_weight: float = 0.001,
        perceptual_weight: float = 0.1,
        modality_weight: float = 0.5,
        use_perceptual: bool = True,
        kl_annealing: bool = True,
        kl_anneal_steps: int = 10000
    ):
        """
        Args:
            recon_loss_type: Type of reconstruction loss
            recon_weight: Weight for main reconstruction loss
            kl_weight: Weight for KL divergence
            perceptual_weight: Weight for perceptual loss
            modality_weight: Weight for modality reconstruction loss
            use_perceptual: Whether to use perceptual loss
            kl_annealing: Whether to anneal KL weight
            kl_anneal_steps: Number of steps for KL annealing
        """
        super().__init__()

        self.recon_weight = recon_weight
        self.kl_weight = kl_weight
        self.perceptual_weight = perceptual_weight
        self.modality_weight = modality_weight
        self.kl_annealing = kl_annealing
        self.kl_anneal_steps = kl_anneal_steps

        self.recon_loss = ReconstructionLoss(loss_type=recon_loss_type)
        self.kl_loss = KLDivergence()

        if use_perceptual:
            self.perceptual_loss = PerceptualLoss()
        else:
            self.perceptual_loss = None

        # Current step for KL annealing
        self.register_buffer('current_step', torch.tensor(0))

    def get_kl_weight(self) -> float:
        """Get current KL weight with optional annealing."""
        if not self.kl_annealing:
            return self.kl_weight

        # Linear annealing
        progress = min(self.current_step.item() / self.kl_anneal_steps, 1.0)
        return self.kl_weight * progress

    def forward(
        self,
        sr_output: torch.Tensor,
        sr_target: torch.Tensor,
        posteriors: List[Tuple[torch.Tensor, torch.Tensor]],
        modality_outputs: Optional[List[torch.Tensor]] = None,
        modality_targets: Optional[List[torch.Tensor]] = None,
        return_components: bool = False
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        """
        Compute total loss.

        Args:
            sr_output: Super-resolved output
            sr_target: Ground truth high-resolution image
            posteriors: List of (mu, logvar) from encoder
            modality_outputs: Reconstructed modalities (optional)
            modality_targets: Target modalities (optional)
            return_components: If True, return dict of individual losses

        Returns:
            Total loss (or dict of losses if return_components=True)
        """
        losses = {}

        # Main reconstruction loss
        losses['recon'] = self.recon_loss(sr_output, sr_target) * self.recon_weight

        # KL divergence (multi-scale)
        kl_weight = self.get_kl_weight()
        losses['kl'] = self.kl_loss.multi_scale(posteriors) * kl_weight

        # Perceptual loss
        if self.perceptual_loss is not None:
            losses['perceptual'] = self.perceptual_loss(sr_output, sr_target) * self.perceptual_weight
        else:
            losses['perceptual'] = torch.tensor(0.0, device=sr_output.device)

        # Modality reconstruction loss
        if modality_outputs is not None and modality_targets is not None:
            mod_loss = 0.0
            for mod_out, mod_target in zip(modality_outputs, modality_targets):
                if mod_target is not None:
                    mod_loss = mod_loss + self.recon_loss(mod_out, mod_target)
            losses['modality'] = mod_loss * self.modality_weight
        else:
            losses['modality'] = torch.tensor(0.0, device=sr_output.device)

        # Total loss
        total_loss = sum(losses.values())

        # Update step counter
        if self.training:
            self.current_step += 1

        if return_components:
            losses['total'] = total_loss
            return losses

        return total_loss


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index (SSIM) loss.

    SSIM measures perceived quality based on structural information,
    luminance, and contrast.
    """

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        channels: int = 1
    ):
        """
        Args:
            window_size: Size of Gaussian window
            sigma: Standard deviation of Gaussian
            channels: Number of input channels
        """
        super().__init__()

        self.window_size = window_size
        self.channels = channels

        # Create Gaussian window
        gauss = torch.Tensor([
            torch.exp(torch.tensor(-(x - window_size // 2) ** 2 / (2 * sigma ** 2)))
            for x in range(window_size)
        ])
        gauss = gauss / gauss.sum()

        # 2D window
        window_1d = gauss.unsqueeze(1)
        window_2d = window_1d.mm(window_1d.t()).unsqueeze(0).unsqueeze(0)
        window = window_2d.expand(channels, 1, window_size, window_size).contiguous()

        self.register_buffer('window', window)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute SSIM loss (1 - SSIM).

        Args:
            pred: Predicted image
            target: Ground truth image

        Returns:
            SSIM loss (lower is better)
        """
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        channels = pred.shape[1]

        # Compute means
        mu1 = F.conv2d(pred, self.window, padding=self.window_size // 2, groups=channels)
        mu2 = F.conv2d(target, self.window, padding=self.window_size // 2, groups=channels)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        # Compute variances
        sigma1_sq = F.conv2d(pred * pred, self.window, padding=self.window_size // 2, groups=channels) - mu1_sq
        sigma2_sq = F.conv2d(target * target, self.window, padding=self.window_size // 2, groups=channels) - mu2_sq
        sigma12 = F.conv2d(pred * target, self.window, padding=self.window_size // 2, groups=channels) - mu1_mu2

        # SSIM formula
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        # Return 1 - SSIM as loss
        return 1 - ssim_map.mean()
