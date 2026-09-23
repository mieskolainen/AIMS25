# Analytic transforms with Jacobians
#
# m.mieskolainen@imperial.ac.uk, 2025

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def momentum_to_angles(x: torch.Tensor):
    """Convert nonzero three-momenta to scattering angles

    Args:
        x (Tensor): Momenta (N, 3)

    Returns:
        Tensor: (cos(theta), phi) pairs (N, 2), phi in [-pi, pi]
    """

    radius = torch.linalg.vector_norm(x, dim=1)
    if x.ndim != 2 or x.shape[1] != 3 or torch.any(radius <= 0):
        raise ValueError("Expected nonzero three-momenta with shape (N, 3).")

    return torch.stack([x[:, 2] / radius, torch.atan2(x[:, 1], x[:, 0])], dim=1)


def angles_to_momentum(angles: torch.Tensor, sqrts, mass=5.11e-4):
    """Reconstruct on-shell center-of-mass momenta

    Args:
        angles (Tensor): (cos(theta), phi) pairs (N, 2)
        sqrts (float | Tensor): CM energy in GeV, scalar or (N,), above 2 * mass
        mass (float): Particle mass in GeV

    Returns:
        Tensor: Momenta (N, 3) in GeV
    """

    if angles.ndim != 2 or angles.shape[1] != 2:
        raise ValueError("Expected angles with shape (N, 2).")

    energy = torch.as_tensor(sqrts, dtype=angles.dtype, device=angles.device).reshape(-1)
    if energy.numel() not in (1, len(angles)) or torch.any(energy <= 2 * mass):
        raise ValueError("Provide one physical CM energy, or one per event.")

    costheta, phi = angles.unbind(dim=1)
    if torch.any(costheta.abs() > 1):
        raise ValueError("cos(theta) must lie in [-1, 1].")

    radius = torch.sqrt(energy**2 / 4 - mass**2)
    pt = radius * torch.sqrt((1 - costheta**2).clamp(min=0))

    return torch.stack([pt * torch.cos(phi), pt * torch.sin(phi), radius * costheta], dim=1)


class ScatteringAngleTransform(nn.Module):
    """Map an open angular rectangle to R^2 with Jacobians relative to solid angle"""

    def __init__(self, costheta_min=-1.0, costheta_max=1.0):
        """Set the open angular rectangle for a logit transform

        Args:
            costheta_min (float): Lower cos(theta) bound >= -1
            costheta_max (float): Upper cos(theta) bound <= 1

        Returns:
            None
        """

        super().__init__()
        if not -1 <= costheta_min < costheta_max <= 1:
            raise ValueError("Invalid angular acceptance.")

        # Keep bounds in double precision, cast to input dtype on use
        self.register_buffer("lower", torch.tensor([costheta_min, -math.pi], dtype=torch.float64))
        self.register_buffer("upper", torch.tensor([costheta_max, math.pi], dtype=torch.float64))

    def forward(self, angles, *, return_logdet=False):
        """Map the open angular rectangle to unconstrained logits

        Args:
            angles (Tensor): (cos(theta), phi) pairs (B, 2)
            return_logdet (bool): Include the log Jacobian with respect to solid angle

        Returns:
            Tensor | tuple[Tensor, Tensor]: Logits (B, 2), optionally with log Jacobians (B,)
        """

        lower, upper = self.lower.to(angles), self.upper.to(angles)
        width = upper - lower
        u = (angles - lower) / width
        if torch.any(~torch.isfinite(u)) or torch.any((u <= 0) | (u >= 1)):
            raise ValueError("Angles must be strictly inside the fiducial rectangle.")

        z = torch.log(u) - torch.log1p(-u)
        if return_logdet:
            ldj = -(torch.log(width) + torch.log(u) + torch.log1p(-u)).sum(dim=1)

            return z, ldj

        return z

    def reverse(self, z, *, return_logdet=False):
        """Map logits back to the angular rectangle

        Args:
            z (Tensor): Logits (B, 2)
            return_logdet (bool): Include the inverse log Jacobian

        Returns:
            Tensor | tuple[Tensor, Tensor]: Angles (B, 2), optionally with log Jacobians (B,)
        """

        lower, upper = self.lower.to(z), self.upper.to(z)
        width = upper - lower
        angles = lower + width * torch.sigmoid(z)
        if return_logdet:
            ldj = (torch.log(width) + F.logsigmoid(z) + F.logsigmoid(-z)).sum(dim=1)

            return angles, ldj

        return angles

class PhysicsTransform(nn.Module):
    """Encode momenta as four constrained features and project them back

    This map is not bijective and has no density Jacobian
    Use ScatteringAngleTransform for densities at fixed CM energy
    """

    def __init__(
        self,
        transverse_mode: str = "log_pt",   # {"pt", "pt2", "log_pt"}
        longitudinal_mode: str = "pz",     # {"pz", "eta"}
        log_pt_clamp: float = 20.0,
        eta_clamp: float = 20.0,
    ):
        """Build a momentum feature encoder without a density Jacobian

        Args:
            transverse_mode (str): pt, pt2, or log_pt
            longitudinal_mode (str): pz or eta
            log_pt_clamp (float): Absolute log-pt limit when decoding
            eta_clamp (float): Absolute pseudorapidity limit when decoding

        Returns:
            None
        """

        super().__init__()

        assert transverse_mode in ("pt", "pt2", "log_pt"), \
            f"Invalid transverse_mode: {transverse_mode}"
        assert longitudinal_mode in ("pz", "eta"), \
            f"Invalid longitudinal_mode: {longitudinal_mode}"

        self.transverse_mode   = transverse_mode
        self.longitudinal_mode = longitudinal_mode

        self.log_pt_clamp = log_pt_clamp
        self.eta_clamp    = eta_clamp
        self.eps          = 1e-12       # numeric safety

    def _forward_physics(self, x: torch.Tensor) -> torch.Tensor:
        """Encode momenta as four constrained physics features

        Args:
            x (Tensor): Momenta (B, 3)

        Returns:
            Tensor: Transverse, longitudinal, cos(phi), sin(phi) features (B, 4)
        """

        px, py, pz = x[:, 0], x[:, 1], x[:, 2]
        pt2 = px * px + py * py
        pt  = torch.sqrt(pt2.clamp(min=self.eps))

        # 1) transverse feature ------------------------------------------------
        if self.transverse_mode == "pt2":
            feat_pt = pt2
        elif self.transverse_mode == "log_pt":
            feat_pt = torch.log(pt)
        else:
            feat_pt = pt

        # 2) longitudinal feature ---------------------------------------------
        if self.longitudinal_mode == "eta":
            p = torch.sqrt((pt2 + pz * pz).clamp(min=self.eps))
            num = (p + pz).clamp(min=self.eps)
            den = (p - pz).clamp(min=self.eps)
            feat_z = 0.5 * torch.log(num / den)
        else:
            feat_z = pz

        # 3) angular features --------------------------------------------------
        phi     = torch.atan2(py, px)
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)

        return torch.stack([feat_pt, feat_z, cos_phi, sin_phi], dim=1)

    def _inverse_physics(self, z: torch.Tensor) -> torch.Tensor:
        """Project physics features back to three-momenta

        Args:
            z (Tensor): Physics features (B, 4)

        Returns:
            Tensor: Momenta (B, 3)
        """

        feat_pt, feat_z, cos_phi, sin_phi = z.unbind(dim=1)

        # Clamp to avoid NaNs in exponentials/logs
        if self.transverse_mode == "log_pt":
            feat_pt = feat_pt.clamp(
                min=-self.log_pt_clamp,
                max= self.log_pt_clamp)
        else:                                   # "pt" or "pt2"
            feat_pt = feat_pt.clamp(min=0.0)

        if self.longitudinal_mode == "eta":
            feat_z = feat_z.clamp(
                min=-self.eta_clamp,
                max= self.eta_clamp)

        # Re‑normalise angle block (safety against numerical drift)
        norm = torch.sqrt(
            (sin_phi * sin_phi + cos_phi * cos_phi).clamp(min=self.eps))
        sin_phi = sin_phi / norm
        cos_phi = cos_phi / norm

        # 1) recover pt --------------------------------------------------------
        if self.transverse_mode == "pt2":
            pt = torch.sqrt(feat_pt.clamp(min=self.eps))
        elif self.transverse_mode == "log_pt":
            pt = torch.exp(feat_pt)
        else:
            pt = feat_pt

        # 2) recover (p_x, p_y) ------------------------------------------------
        phi = torch.atan2(sin_phi, cos_phi)
        px  = pt * torch.cos(phi)
        py  = pt * torch.sin(phi)

        # 3) recover p_z -------------------------------------------------------
        if self.longitudinal_mode == "eta":
            pz = pt * torch.sinh(feat_z)
        else:
            pz = feat_z

        return torch.stack([px, py, pz], dim=1)

    def forward(self, x: torch.Tensor, *, return_logdet: bool = False):
        """Encode momenta as four constrained physics features

        Args:
            x (Tensor): Momenta (B, 3)
            return_logdet (bool): Must be False, Jacobian requests raise ValueError

        Returns:
            Tensor: Transverse, longitudinal, cos(phi), sin(phi) features (B, 4)
        """

        if return_logdet:
            raise ValueError("PhysicsTransform is a feature encoder, not a density transform.")

        return self._forward_physics(x)

    def reverse(self, z: torch.Tensor, *, return_logdet: bool = False):
        """Project physics features back to three-momenta

        Args:
            z (Tensor): Physics features (B, 4)
            return_logdet (bool): Must be False, Jacobian requests raise ValueError

        Returns:
            Tensor: Momenta (B, 3)
        """

        if return_logdet:
            raise ValueError("PhysicsTransform's projection has no invertible density Jacobian.")

        return self._inverse_physics(z)


class ZScoreTransform(nn.Module):
    """Applies mean-variance normalization and its inverse"""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        """Store feature means and standard deviations

        Args:
            mean (Tensor): Feature means (D,)
            std (Tensor): Feature standard deviations (D,)

        Returns:
            None
        """

        super().__init__()
        self.register_buffer('mean', mean.view(1, -1))
        self.register_buffer('std',  std.view(1, -1))

    def forward(self, x: torch.Tensor, return_logdet: bool = False):
        """Standardize features

        Args:
            x (Tensor): Input features (B, D)
            return_logdet (bool): Include the log Jacobian

        Returns:
            Tensor | tuple[Tensor, Tensor]: Features (B, D), optionally with log Jacobians (1,)
        """

        normed = (x - self.mean) / self.std
        if return_logdet:
            # log-det of scaling by 1/std
            logdet = -torch.sum(torch.log(self.std), dim=1)

            return normed, logdet

        return normed

    def reverse(self, z: torch.Tensor, return_logdet: bool = False):
        """Restore feature means and scales

        Args:
            z (Tensor): Input features (B, D)
            return_logdet (bool): Include the log Jacobian

        Returns:
            Tensor | tuple[Tensor, Tensor]: Features (B, D), optionally with log Jacobians (1,)
        """

        x = z * self.std + self.mean
        if return_logdet:
            logdet = torch.sum(torch.log(self.std), dim=1)

            return x, logdet

        return x

class MinMaxTransform(nn.Module):
    """Applies min‑max normalization and its inverse, with support for
    constant features (min==max)

    The log‑Jacobian sums only over the non‑constant dims
    """

    def __init__(self, min_val: torch.Tensor, max_val: torch.Tensor):
        """Store feature bounds, mapping constant features to zero

        Args:
            min_val (Tensor): Lower bounds (D,)
            max_val (Tensor): Upper bounds (D,)

        Returns:
            None
        """

        super().__init__()

        # shape (1, D)
        self.register_buffer('min',   min_val.view(1, -1))
        self.register_buffer('max',   max_val.view(1, -1))
        self.register_buffer('range', (max_val - min_val).view(1, -1))

        # mask of non‑constant dims: shape (1, D), dtype float (0 or 1)
        self.register_buffer('nonconst', (self.range != 0).float())

    def forward(self, x: torch.Tensor, return_logdet: bool = False):
        """Scale nonconstant features to the unit interval

        Args:
            x (Tensor): Input features (B, D)
            return_logdet (bool): Include the log Jacobian, excluding constant features

        Returns:
            Tensor | tuple[Tensor, Tensor]: Features (B, D), optionally with log Jacobians (1,)
        """

        # avoid div‑zero by replacing zero ranges with 1
        range_safe = self.range.clone()
        range_safe[range_safe == 0] = 1.0

        # compute normalized
        x_norm = (x - self.min) / range_safe

        # force constant dims to zero
        x_norm = x_norm * self.nonconst

        if not return_logdet:
            return x_norm

        # log‑Jacobian: −\sum_i log(range[i]) over nonconst dims
        log_range = torch.log(range_safe)
        logdet = - (self.nonconst * log_range).sum(dim=1)  # shape (B,)

        return x_norm, logdet

    def reverse(self, z: torch.Tensor, return_logdet: bool = False):
        """Restore feature bounds, fixing constant features at their minimum

        Args:
            z (Tensor): Input features (B, D)
            return_logdet (bool): Include the log Jacobian, excluding constant features

        Returns:
            Tensor | tuple[Tensor, Tensor]: Features (B, D), optionally with log Jacobians (1,)
        """

        range_safe = self.range.clone()
        range_safe[range_safe == 0] = 1.0

        # invert: z * range + min
        x = z * range_safe + self.min

        # force constant dims = min
        x = x * self.nonconst + self.min * (1 - self.nonconst)

        if not return_logdet:
            return x

        # inverse log‑Jacobian: +\sum_i log(range[i]) over nonconst dims
        log_range = torch.log(range_safe)
        logdet = (self.nonconst * log_range).sum(dim=1)  # shape (B,)

        return x, logdet

class CompositeTransform(nn.Module):
    """Chain transforms with optional log Jacobians, using identity for an empty list"""

    def __init__(self, transforms: list[nn.Module]):
        """Chain transforms in order, using identity for an empty list

        Args:
            transforms (list[nn.Module]): Transforms with forward and reverse methods

        Returns:
            None
        """

        super().__init__()
        self.transforms = nn.ModuleList(transforms)

    def forward(self, x: torch.Tensor, return_logdet: bool = False):
        """Apply the transform sequence

        Args:
            x (Tensor): Input features (B, D)
            return_logdet (bool): Include the log Jacobian

        Returns:
            Tensor | tuple[Tensor, Tensor]: Features (B, D_out), optionally with log Jacobians (B,)
        """

        out = x
        if return_logdet:
            # initialize a zero‐logdet tensor matching batch size, device, dtype
            total_logdet = x.new_zeros(x.size(0))

            for t in self.transforms:
                out, logdet = t.forward(out, return_logdet=True)
                total_logdet = total_logdet + logdet

            return out, total_logdet
        else:
            for t in self.transforms:
                out = t.forward(out, return_logdet=False)

            return out

    def reverse(self, z: torch.Tensor, return_logdet: bool = False):
        """Apply inverse transforms in reverse order

        Args:
            z (Tensor): Input features (B, D)
            return_logdet (bool): Include the log Jacobian

        Returns:
            Tensor | tuple[Tensor, Tensor]: Features (B, D_out), optionally with log Jacobians (B,)
        """

        out = z
        if return_logdet:
            total_logdet = z.new_zeros(z.size(0))

            for t in reversed(self.transforms):
                out, logdet = t.reverse(out, return_logdet=True)
                total_logdet = total_logdet + logdet

            return out, total_logdet
        else:
            for t in reversed(self.transforms):
                out = t.reverse(out, return_logdet=False)

            return out
