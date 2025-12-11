"""
Product of Gaussians Fusion for U-HVED

This module implements the key innovation of U-HVED: fusing variational
distributions from multiple orientations via the product of Gaussians.

The idea is that each orientation provides its own estimate of the latent
representation as a Gaussian distribution (mu, sigma). By taking the
product of these distributions, we get a fused distribution that combines
information from all available orientations.

Key property: The product of Gaussians is also a Gaussian, and its
parameters can be computed in closed form:
    T_fused = sum(T_i)  where T_i = 1/var_i (precision)
    mu_fused = sum(mu_i * T_i) / T_fused

This allows the network to handle missing orientations gracefully - we simply
exclude missing orientations from the sum.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple


class ProductOfGaussians(nn.Module):
    """
    Fuses variational parameters from multiple orientations via Product of Gaussians.

    Given mu and logvar from each orientation, computes the fused posterior
    using precision-weighted combination with an optional prior.
    """

    def __init__(
        self,
        use_prior: bool = True,
        prior_mu: float = 0.0,
        prior_logvar: float = 0.0,
        eps: float = 1e-7
    ):
        """
        Args:
            use_prior: Whether to include a weak prior in the product
            prior_mu: Prior mean (default 0)
            prior_logvar: Prior log-variance (default 0, i.e., var=1)
            eps: Small constant for numerical stability
        """
        super().__init__()

        self.use_prior = use_prior
        self.prior_mu = prior_mu
        self.prior_logvar = prior_logvar
        self.eps = eps

    def forward(
        self,
        mus: Dict[int, torch.Tensor],
        logvars: Dict[int, torch.Tensor],
        orientation_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute fused posterior via product of Gaussians.

        Args:
            mus: Dict mapping orientation index to mu tensor (B, C, D, H, W)
            logvars: Dict mapping orientation index to logvar tensor (B, C, D, H, W)
            orientation_mask: Optional boolean tensor indicating which orientations to include
                          - Shape (num_orientations,): same mask for all batch elements
                          - Shape (B, num_orientations): different mask per batch element

        Returns:
            posterior_mu: Fused mean (B, C, D, H, W)
            posterior_logvar: Fused log-variance (B, C, D, H, W)
        """
        if len(mus) == 0:
            raise ValueError("At least one orientation must be present")

        # Get reference tensor for shape
        ref_tensor = next(iter(mus.values()))
        device = ref_tensor.device
        dtype = ref_tensor.dtype
        batch_size = ref_tensor.shape[0]

        # Determine mask type
        mask_is_batched = orientation_mask is not None and orientation_mask.dim() == 2

        # Initialize accumulators for precision-weighted sum
        precision_sum = torch.zeros_like(ref_tensor)
        weighted_mu_sum = torch.zeros_like(ref_tensor)

        # Accumulate contributions from each orientation
        for mod_idx, mu in mus.items():
            logvar = logvars[mod_idx]

            # Handle masking
            if orientation_mask is not None:
                if mask_is_batched:
                    # Batched mask: (B, num_orientations)
                    # Create mask with shape (B, 1, 1, 1, 1) for broadcasting
                    batch_mask = orientation_mask[:, mod_idx].view(batch_size, 1, 1, 1, 1).float()
                else:
                    # Global mask: (num_orientations,)
                    # Skip this orientation entirely if masked out
                    if not orientation_mask[mod_idx]:
                        continue
                    batch_mask = 1.0
            else:
                batch_mask = 1.0

            # Compute precision (inverse variance)
            precision = 1.0 / (torch.exp(logvar) + self.eps)

            # Apply mask to precision (zeros out contribution from masked batch elements)
            if isinstance(batch_mask, torch.Tensor):
                precision = precision * batch_mask
                weighted_mu = mu * precision
            else:
                weighted_mu = mu * precision

            # Accumulate
            precision_sum = precision_sum + precision
            weighted_mu_sum = weighted_mu_sum + weighted_mu

        # Add prior contribution if enabled
        if self.use_prior:
            prior_precision = 1.0 / (torch.exp(torch.tensor(self.prior_logvar, device=device, dtype=dtype)) + self.eps)
            precision_sum = precision_sum + prior_precision
            weighted_mu_sum = weighted_mu_sum + self.prior_mu * prior_precision

        # Compute fused posterior
        posterior_var = 1.0 / (precision_sum + self.eps)
        posterior_mu = weighted_mu_sum * posterior_var
        posterior_logvar = torch.log(posterior_var + self.eps)

        return posterior_mu, posterior_logvar


class GaussianSampler(nn.Module):
    """
    Samples from Gaussian distribution using the reparameterization trick.

    During training: z = mu + sigma * epsilon, where epsilon ~ N(0, 1)
    During inference: z = mu (deterministic)
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        deterministic: bool = False
    ) -> torch.Tensor:
        """
        Sample from N(mu, exp(logvar)).

        Args:
            mu: Mean tensor
            logvar: Log-variance tensor
            deterministic: If True, return mean without sampling

        Returns:
            Sampled latent tensor
        """
        if deterministic or not self.training:
            return mu

        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(mu)
        return mu + std * noise


class MultiScaleFusion(nn.Module):
    """
    Applies Product of Gaussians fusion at multiple scales.

    This is the main fusion module used in U-HVED, combining variational
    parameters from all orientations at each spatial scale.
    """

    def __init__(
        self,
        num_scales: int = 4,
        use_prior: bool = True
    ):
        """
        Args:
            num_scales: Number of spatial scales
            use_prior: Whether to use prior in PoG fusion
        """
        super().__init__()

        self.num_scales = num_scales
        self.fusion = ProductOfGaussians(use_prior=use_prior)
        self.sampler = GaussianSampler()

    def forward(
        self,
        encoder_outputs: List[Dict[str, Dict[int, torch.Tensor]]],
        orientation_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> Tuple[List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Fuse and sample at all scales.

        Args:
            encoder_outputs: List (per scale) of dicts with 'mu' and 'logvar'
                            dicts mapping orientation indices to tensors
            orientation_mask: Boolean tensor indicating present orientations
            deterministic: If True, return means without sampling

        Returns:
            samples: List of sampled latent tensors at each scale
            posteriors: List of (mu, logvar) tuples for KL computation
        """
        samples = []
        posteriors = []

        for scale_idx, scale_data in enumerate(encoder_outputs):
            mus = scale_data['mu']
            logvars = scale_data['logvar']

            # Skip if no orientations present at this scale
            if len(mus) == 0:
                continue

            # Fuse via Product of Gaussians
            fused_mu, fused_logvar = self.fusion(mus, logvars, orientation_mask)

            # Sample
            z = self.sampler(fused_mu, fused_logvar, deterministic)

            samples.append(z)
            posteriors.append((fused_mu, fused_logvar))

        return samples, posteriors


class AttentionFusion(nn.Module):
    """
    Alternative fusion strategy using attention mechanism.

    Instead of Product of Gaussians, this uses learned attention weights
    to combine features from different orientations. Can be used as an
    ablation or alternative approach.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4
    ):
        """
        Args:
            hidden_dim: Feature dimension
            num_heads: Number of attention heads
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Attention layers
        self.query = nn.Conv3d(hidden_dim, hidden_dim, 1)
        self.key = nn.Conv3d(hidden_dim, hidden_dim, 1)
        self.value = nn.Conv3d(hidden_dim, hidden_dim, 1)

        # Output projection
        self.proj = nn.Conv3d(hidden_dim, hidden_dim * 2, 1)  # For mu and logvar

    def forward(
        self,
        features: Dict[int, torch.Tensor],
        orientation_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse features using attention.

        Args:
            features: Dict mapping orientation index to feature tensor
            orientation_mask: Optional mask for present orientations

        Returns:
            mu, logvar: Fused variational parameters
        """
        if len(features) == 0:
            raise ValueError("At least one orientation must be present")

        # Stack features
        feat_list = []
        for mod_idx, feat in features.items():
            if orientation_mask is not None and not orientation_mask[mod_idx]:
                continue
            feat_list.append(feat)

        if len(feat_list) == 1:
            # Single orientation, no attention needed
            fused = feat_list[0]
        else:
            # Stack: (B, num_mod, C, D, H, W)
            stacked = torch.stack(feat_list, dim=1)
            B, num_mod, C, D, H, W = stacked.shape

            # Reshape for attention: (B*D*H*W, num_mod, C)
            stacked = stacked.permute(0, 3, 4, 5, 1, 2).reshape(B * D * H * W, num_mod, C)

            # Self-attention
            Q = self.query(feat_list[0]).permute(0, 2, 3, 4, 1).reshape(B * D * H * W, 1, C)
            K = stacked
            V = stacked

            # Attention weights
            attn = torch.softmax(torch.bmm(Q, K.transpose(1, 2)) / (C ** 0.5), dim=-1)

            # Weighted sum
            fused = torch.bmm(attn, V).squeeze(1)  # (B*D*H*W, C)
            fused = fused.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3)

        # Project to mu and logvar
        out = self.proj(fused)
        mu = out[:, :self.hidden_dim]
        logvar = torch.clamp(out[:, self.hidden_dim:], -10, 10)

        return mu, logvar
