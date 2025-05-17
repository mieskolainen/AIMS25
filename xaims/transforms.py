# Analytic transforms with Jacobians
#
# m.mieskolainen@imperial.ac.uk, 2025

import torch
import torch.nn as nn

class PhysicsTransform(nn.Module):
    r"""
    Invertible map
    
    (p_x, p_y, p_z) -> (X , Z , cos_phi , sin_phi),
    
    where
    
        X  = {"pt", "pt2", "log_pt"}  (transverse_mode)
        Z  = {"p_z", "eta"}           (longitudinal_mode)
    
    The (cos_phi, sin_phi) block is volume‑preserving but non‑rectangular, so
    its Jacobian determinant is 1 and does not contribute to log |J|
    (see manifold embeddings and Gram determinants).
    """
    
    def __init__(
        self,
        transverse_mode: str = "log_pt",   # {"pt", "pt2", "log_pt"}
        longitudinal_mode: str = "pz",     # {"pz", "eta"}
        log_pt_clamp: float = 20.0,
        eta_clamp: float = 20.0,
    ):
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
        feat = self._forward_physics(x)

        if not return_logdet:
            return feat

        px, py, pz = x[:, 0], x[:, 1], x[:, 2]
        pt2 = px * px + py * py
        pt  = torch.sqrt(pt2.clamp(min=self.eps))

        # -------- transverse contribution ------------------------------------
        if   self.transverse_mode == "pt":
            dfeat_dpt = torch.ones_like(pt)            # d(pt)/dpt = 1
        elif self.transverse_mode == "pt2":
            dfeat_dpt = 2.0 * pt                       # d(pt^2)/dpt = 2pt
        else:  # "log_pt"
            dfeat_dpt = 1.0 / (pt + self.eps)          # d(log pt)/dpt = 1/pt
        
        # derivative + polar Jacobian
        logdet = torch.log(dfeat_dpt + self.eps) - torch.log(pt + self.eps)
        
        # -------- longitudinal contribution ----------------------------------
        if self.longitudinal_mode == "eta":
            p = torch.sqrt((pt2 + pz * pz).clamp(min=self.eps))
            logdet = logdet + torch.log((pt / p).clamp(min=self.eps))

        return feat, logdet
    
    def reverse(self, z: torch.Tensor, *, return_logdet: bool = False):
        x_recon = self._inverse_physics(z)

        if not return_logdet:
            return x_recon

        px, py, pz = x_recon[:, 0], x_recon[:, 1], x_recon[:, 2]
        pt2 = px * px + py * py
        pt  = torch.sqrt(pt2.clamp(min=self.eps))

        # -------- transverse contribution (sign flipped) ----------------------
        if   self.transverse_mode == "pt":
            dfeat_dpt = torch.ones_like(pt)
        elif self.transverse_mode == "pt2":
            dfeat_dpt = 2.0 * pt
        else:  # "log_pt"
            dfeat_dpt = 1.0 / (pt + self.eps)

        logdet = - ( torch.log(dfeat_dpt + self.eps) - torch.log(pt + self.eps) )

        # -------- longitudinal contribution (sign flipped) --------------------
        if self.longitudinal_mode == "eta":
            p = torch.sqrt((pt2 + pz * pz).clamp(min=self.eps))
            logdet = logdet - torch.log((pt / p).clamp(min=self.eps))
        
        return x_recon, logdet


class ZScoreTransform(nn.Module):
    """
    Applies mean-variance normalization and its inverse
    """
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.register_buffer('mean', mean.view(1, -1))
        self.register_buffer('std',  std.view(1, -1))

    def forward(self, x: torch.Tensor, return_logdet: bool = False):
        normed = (x - self.mean) / self.std
        if return_logdet:
            # log-det of scaling by 1/std
            logdet = -torch.sum(torch.log(self.std), dim=1)
            return normed, logdet
        return normed

    def reverse(self, z: torch.Tensor, return_logdet: bool = False):
        x = z * self.std + self.mean
        if return_logdet:
            logdet = torch.sum(torch.log(self.std), dim=1)
            return x, logdet
        return x

class MinMaxTransform(nn.Module):
    """
    Applies min‑max normalization and its inverse, with support for
    constant features (min==max)
    
    The log‑Jacobian sums only over the non‑constant dims
    """
    def __init__(self, min_val: torch.Tensor, max_val: torch.Tensor):
        super().__init__()
        
        # shape (1, D)
        self.register_buffer('min',   min_val.view(1, -1))
        self.register_buffer('max',   max_val.view(1, -1))
        self.register_buffer('range', (max_val - min_val).view(1, -1))
        
        # mask of non‑constant dims: shape (1, D), dtype float (0 or 1)
        self.register_buffer('nonconst', (self.range != 0).float())
        
    def forward(self, x: torch.Tensor, return_logdet: bool = False):
        """
        x: (B, D)
        returns x_norm: (B, D), and optionally logdet: (B,)
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
        """
        z: (B, D)
        returns x: (B, D), and optionally logdet: (B,)
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
    """
    Chains multiple transforms sequentially, tracking log-Jacobian.
    If `transforms` is empty, acts as the identity (logdet = 0)
    """
    def __init__(self, transforms: list[nn.Module]):
        super().__init__()
        self.transforms = nn.ModuleList(transforms)

    def forward(self, x: torch.Tensor, return_logdet: bool = False):
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

