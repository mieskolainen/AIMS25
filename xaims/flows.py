# Coupling layer flow Normalizing Flows with affine RealNVP and spline (RQS) designs
#
# Dinh et al.,   https://arxiv.org/abs/1605.08803
# Durkan et al., https://arxiv.org/abs/1906.04032
#
# m.mieskolainen@imperial.ac.uk, 2025

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .aux import ResidualMLP
from . import splines

class CouplingFlow(nn.Module):
    def __init__(self,
                 x_dim: int,
                 cond_dim: int    = None,
                 flow_layers: int = 4,
                 flow_type: str   = "realnvp",
                 permute: bool    = True,
                 nn_param: list[dict] = [{"hidden_dim": [64, 64], "act": "relu", "layer_norm": True},
                                         {"hidden_dim": [64, 64], "act": "relu", "layer_norm": True}],
                 conditional_affine: bool = False):

        """Build a conditional coupling flow with a Gaussian base

        Args:
            x_dim (int): Data width D
            cond_dim (int | None): Condition width C
            flow_layers (int): Coupling layer count
            flow_type (str): realnvp or spline
            permute (bool): Permute features between layers
            nn_param (list[dict]): Options for the two coupling networks
            conditional_affine (bool): Learn a condition-dependent location and scale

        Returns:
            None
        """

        super().__init__()
        self.x_dim = x_dim
        self.flow_type = flow_type
        # An outer context-dependent location/scale lets the full distribution,
        # including a spline's linear tails, move and contract with the data
        self.context_affine = None
        if conditional_affine:
            if not cond_dim or cond_dim < 1:
                raise ValueError("conditional_affine requires a positive cond_dim.")

            self.context_affine = ResidualMLP(cond_dim, 2 * x_dim, **nn_param[0])
            nn.init.zeros_(self.context_affine.output_proj.weight)
            nn.init.zeros_(self.context_affine.output_proj.bias)

        print(f'CouplingFlow: using flow_type = {flow_type} with flow_layers = {flow_layers} and permute = {permute}')

        # Register mask buffers and create layers
        self.coupling_layers = nn.ModuleList()

        # Create alternating "1D checkerboard" binary masks
        for i in range(flow_layers):

            # Start with 0 or 1 depending on layer index
            pattern = [(j + i) % 2 for j in range(x_dim)]
            mask = torch.tensor(pattern, dtype=torch.bool)

            if flow_type in ("realnvp", "affine"):
                self.coupling_layers.append(
                    RealNVPCoupling(mask=mask, cond_dim=cond_dim, nn_param=nn_param)
                )
            elif flow_type == "spline":
                self.coupling_layers.append(
                    ScalarSpline(cond_dim=cond_dim, nn_param=nn_param)
                    if x_dim == 1 else
                    SplineCoupling(mask=mask, cond_dim=cond_dim, nn_param=nn_param)
                )
            else:
                raise Exception("Unknown type (choose 'affine' or 'spline')")

            # Add random permutation (except after last layer)
            if permute and i < flow_layers - 1:
                self.coupling_layers.append(RandomPermutation(x_dim))

    def log_base(self, z):
        """Evaluate the joint standard-normal log density

        Args:
            z (Tensor): Base coordinates (B, D)

        Returns:
            Tensor: Log densities (B,)
        """

        return -0.5 * torch.sum(z**2, dim=1) - 0.5 * self.x_dim * math.log(2*math.pi)

    def loss(self, x, cond):
        """Compute negative log likelihood

        Args:
            x (Tensor): Data (B, D)
            cond (Tensor | None): Conditions (B, C)

        Returns:
            Tensor: Per-sample losses (B,)
        """

        return -self.log_prob(x, cond)

    def _affine_parameters(self, cond):
        """Predict location and bounded log scale from conditions

        Args:
            cond (Tensor): Conditions (B, C)

        Returns:
            tuple[Tensor, Tensor]: Location and log scale, each (B, D)
        """

        location, log_scale = self.context_affine(cond).chunk(2, dim=-1)

        return location, log_scale.clamp(-7, 7)

    def forward(self, x, cond):
        """Apply the flow

        Args:
            x (Tensor): Coordinates (B, D)
            cond (Tensor | None): Conditions (B, C)

        Returns:
            tuple[Tensor, Tensor]: Transformed coordinates (B, D) and log Jacobians (B,)
        """

        log_det_total = x.new_zeros(x.shape[0])
        z = x
        if self.context_affine is not None:
            location, log_scale = self._affine_parameters(cond)
            z = (z - location) * torch.exp(-log_scale)
            log_det_total -= log_scale.sum(dim=1)

        for layer in self.coupling_layers:
            z, log_det = layer(z, cond)
            log_det_total += log_det

        return z, log_det_total

    def inverse(self, z, cond):
        """Invert the flow

        Args:
            z (Tensor): Coordinates (B, D)
            cond (Tensor | None): Conditions (B, C)

        Returns:
            tuple[Tensor, Tensor]: Transformed coordinates (B, D) and log Jacobians (B,)
        """

        log_det_total = z.new_zeros(z.shape[0])
        x = z

        for layer in reversed(self.coupling_layers):
            x, log_det = layer.inverse(x, cond)
            log_det_total += log_det

        if self.context_affine is not None:
            location, log_scale = self._affine_parameters(cond)
            x = x * torch.exp(log_scale) + location
            log_det_total += log_scale.sum(dim=1)

        return x, log_det_total

    def log_prob(self, x, cond):
        """Evaluate the joint data log density

        Args:
            x (Tensor): Data (B, D)
            cond (Tensor | None): Conditions (B, C)

        Returns:
            Tensor: Log densities (B,)
        """

        z, log_det = self.forward(x, cond)

        return self.log_base(z) + log_det

    def sample(self, num_samples, cond, return_ldj=False):
        """Transform Gaussian draws into data samples

        Args:
            num_samples (int): Draw count S
            cond (Tensor | None): Conditions (S, C)
            return_ldj (bool): Include the inverse log Jacobian

        Returns:
            Tensor | tuple[Tensor, Tensor]: Samples (S, D), optionally with log Jacobians (S,)
        """

        reference = next(self.parameters())
        z = torch.randn(num_samples, self.x_dim, device=reference.device, dtype=reference.dtype)
        x, ldj = self.inverse(z, cond)

        if return_ldj:
            return x, ldj
        else:
            return x

class RandomPermutation(nn.Module):
    def __init__(self, dim: int):
        """Store a random feature permutation and its inverse

        Args:
            dim (int): Feature count D

        Returns:
            None
        """

        super().__init__()
        perm = torch.randperm(dim)
        inv_perm = torch.argsort(perm)
        self.register_buffer("perm", perm)
        self.register_buffer("inv_perm", inv_perm)

    def forward(self, x, cond=None):
        """Apply the feature permutation

        Args:
            x (Tensor): Coordinates (B, D)
            cond (Tensor | None): Unused condition input

        Returns:
            tuple[Tensor, Tensor]: Transformed coordinates (B, D) and log Jacobians (B,)
        """

        return x[:, self.perm], x.new_zeros(x.shape[0])

    def inverse(self, z, cond=None):
        """Invert the feature permutation

        Args:
            z (Tensor): Coordinates (B, D)
            cond (Tensor | None): Unused condition input

        Returns:
            tuple[Tensor, Tensor]: Transformed coordinates (B, D) and log Jacobians (B,)
        """

        return z[:, self.inv_perm], z.new_zeros(z.shape[0])

class RealNVPCoupling(nn.Module):
    """RealNVP type affine flow coupling type layers, with optional conditioning"""

    def __init__(self,
                 mask: torch.Tensor,
                 cond_dim: int = None,
                 nn_param: list[dict] = [{}, {}]):

        """Build an affine coupling layer

        Args:
            mask (Tensor): Fixed-feature mask (D,)
            cond_dim (int | None): Condition width C
            nn_param (list[dict]): Scale and translation network options

        Returns:
            None
        """

        super().__init__()
        mask = mask.to(torch.bool)
        assert mask.dim() == 1, "mask must be a 1D tensor"

        self.cond_dim = cond_dim or 0

        # register mask + its inverse so they move with .to(device)
        self.register_buffer('mask',     mask)
        self.register_buffer('inv_mask', ~mask)

        # how many coords in each partition?
        self.pass_dim  = int(mask.sum().item())
        self.trans_dim = mask.numel() - self.pass_dim

        # sub‐nets see (passive + cond) as input, and output one param per trans‐dim
        net_in_dim = self.pass_dim + self.cond_dim

        self.scale_net     = ResidualMLP(in_dim = net_in_dim, out_dim = self.trans_dim, **nn_param[0])
        self.translate_net = ResidualMLP(in_dim = net_in_dim, out_dim = self.trans_dim, **nn_param[1])

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None):

        # split x into the "passive" and "to-transform" parts
        """Apply the affine coupling

        Args:
            x (Tensor): Coordinates (B, D)
            cond (Tensor | None): Conditions (B, C)

        Returns:
            tuple[Tensor, Tensor]: Transformed coordinates (B, D) and log Jacobians (B,)
        """

        x_pass  = x[:, self.mask]
        x_trans = x[:, self.inv_mask]

        # build net input = [x_pass, cond?]
        net_in = torch.cat([x_pass, cond], dim=1) if cond is not None else x_pass

        # compute scale & translation
        s = self.scale_net(net_in)
        t = self.translate_net(net_in)

        # affine transform
        y_trans = x_trans * torch.exp(s) + t

        y = x.clone()
        y[:, self.inv_mask] = y_trans
        log_det = s.sum(dim=1)

        return y, log_det

    def inverse(self, y: torch.Tensor, cond: torch.Tensor | None = None):

        # same partitioning
        """Invert the affine coupling

        Args:
            y (Tensor): Coordinates (B, D)
            cond (Tensor | None): Conditions (B, C)

        Returns:
            tuple[Tensor, Tensor]: Transformed coordinates (B, D) and log Jacobians (B,)
        """

        y_pass  = y[:, self.mask]
        y_trans = y[:, self.inv_mask]

        net_in = torch.cat([y_pass, cond], dim=1) if cond is not None else y_pass

        s = self.scale_net(net_in)
        t = self.translate_net(net_in)

        # invert the affine
        x_trans = (y_trans - t) * torch.exp(-s)

        x = y.clone()
        x[:, self.inv_mask] = x_trans
        log_det = -s.sum(dim=1)

        return x, log_det

class ScalarSpline(nn.Module):
    """Conditional or unconditional scalar spline with identity tails"""

    def __init__(self, cond_dim=None, K=5, B=3, nn_param=[{}, {}]):
        """Build a scalar spline with conditional or learned parameters

        Args:
            cond_dim (int | None): Condition width C
            K (int): Spline bin count
            B (float): Tail boundary, identity outside [-B, B]
            nn_param (list[dict]): Spline network options

        Returns:
            None
        """

        super().__init__()
        self.cond_dim = cond_dim or 0
        self.K, self.B = K, B
        if self.cond_dim:
            self.param_net = ResidualMLP(
                in_dim=self.cond_dim, out_dim=3 * K - 1, **nn_param[0])
        else:
            self.spline_params = nn.Parameter(torch.zeros(1, 1, 3 * K - 1))

    def _transform(self, x, cond, inverse):
        """Evaluate the scalar spline in either direction

        Args:
            x (Tensor): Coordinates (B, 1)
            cond (Tensor | None): Conditions (B, C)
            inverse (bool): Apply the inverse spline

        Returns:
            tuple[Tensor, Tensor]: Coordinates (B, 1) and log Jacobians (B,)
        """

        if x.ndim != 2 or x.shape[1] != 1:
            raise ValueError("ScalarSpline expects (batch, 1) inputs.")

        if self.cond_dim:
            if cond is None or cond.shape != (len(x), self.cond_dim):
                raise ValueError("Expected one condition vector per scalar input.")

            params = self.param_net(cond).reshape(len(x), 1, 3 * self.K - 1)
        else:
            params = self.spline_params.expand(len(x), -1, -1)

        widths, heights, derivatives = torch.split(params, self.K, dim=-1)
        y, logdet = splines.unconstrained_RQS(
            x, widths, heights, derivatives, inverse=inverse, tail_bound=self.B)

        return y, logdet.sum(dim=1)

    def forward(self, x, cond=None):
        """Apply the scalar spline

        Args:
            x (Tensor): Coordinates (B, 1)
            cond (Tensor | None): Conditions (B, C)

        Returns:
            tuple[Tensor, Tensor]: Transformed coordinates (B, 1) and log Jacobians (B,)
        """

        return self._transform(x, cond, inverse=False)

    def inverse(self, z, cond=None):
        """Invert the scalar spline

        Args:
            z (Tensor): Coordinates (B, 1)
            cond (Tensor | None): Conditions (B, C)

        Returns:
            tuple[Tensor, Tensor]: Transformed coordinates (B, 1) and log Jacobians (B,)
        """

        return self._transform(z, cond, inverse=True)


class SplineCoupling(nn.Module):
    """Neural spline flow with coupling type layers, with optional conditioning"""

    def __init__(self,
                 mask: torch.Tensor,
                 cond_dim: int = None,
                 K: int = 5,
                 B: int = 3,
                 nn_param=[{}, {}]):
        """Build a two-stage rational-quadratic spline coupling

        Args:
            mask (Tensor): Fixed-feature mask (D,)
            cond_dim (int | None): Condition width C
            K (int): Spline bin count
            B (float): Tail boundary, identity outside [-B, B]
            nn_param (list[dict]): Spline network options

        Returns:
            None
        """

        super().__init__()

        # mask: 1d Boolean or 0/1 Tensor of length D
        mask = mask.to(torch.bool)
        assert mask.dim() == 1, "mask must be a vector"
        D = mask.numel()
        assert D >= 2, "Dimension must be at least 2 for coupling layers."

        self.register_buffer('mask', mask)
        self.register_buffer('inv_mask', ~mask)

        # how many dims in each part
        self.upper_dim = int(mask.sum().item())
        self.lower_dim = D - self.upper_dim
        self.K = K
        self.B = B
        self.cond_dim = cond_dim or 0

        # sub‑network inputs include conditioning if provided
        x_dim_lower = self.lower_dim + self.cond_dim
        x_dim_upper = self.upper_dim + self.cond_dim

        # networks
        self.f1 = ResidualMLP(in_dim=x_dim_lower, out_dim=(3 * K - 1) * self.upper_dim, **nn_param[0])
        self.f2 = ResidualMLP(in_dim=x_dim_upper, out_dim=(3 * K - 1) * self.lower_dim, **nn_param[1])

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None):
        # x: (batch, D)
        """Apply the spline coupling

        Args:
            x (Tensor): Coordinates (B, D)
            cond (Tensor | None): Conditions (B, C)

        Returns:
            tuple[Tensor, Tensor]: Transformed coordinates (B, D) and log Jacobians (B,)
        """

        log_det = x.new_zeros(x.shape[0])

        # pick out lower / upper by mask
        x_lower = x[:, self.inv_mask]  # passive half
        x_upper = x[:, self.mask]      # to-be-transformed

        # 1: lower -> transform upper
        inp1 = torch.cat([x_lower, cond], dim=1) if cond is not None else x_lower
        params1 = self.f1(inp1).view(-1, self.upper_dim, 3 * self.K - 1)
        W1, H1, D1 = torch.split(params1, self.K, dim=2)
        x_upper, ld1 = splines.unconstrained_RQS(x_upper, W1, H1, D1, inverse=False, tail_bound=self.B)
        log_det = log_det + ld1.sum(1)

        # 2: transformed upper -> transform lower
        inp2 = torch.cat([x_upper, cond], dim=1) if cond is not None else x_upper
        params2 = self.f2(inp2).view(-1, self.lower_dim, 3 * self.K - 1)
        W2, H2, D2 = torch.split(params2, self.K, dim=2)
        x_lower, ld2 = splines.unconstrained_RQS(x_lower, W2, H2, D2, inverse=False, tail_bound=self.B)
        log_det = log_det + ld2.sum(1)

        # scatter back into full D‑dim vector
        z = x.clone()
        z[:, self.mask]     = x_upper
        z[:, self.inv_mask] = x_lower

        return z, log_det

    def inverse(self, z: torch.Tensor, cond: torch.Tensor = None):
        """Invert the spline coupling

        Args:
            z (Tensor): Coordinates (B, D)
            cond (Tensor | None): Conditions (B, C)

        Returns:
            tuple[Tensor, Tensor]: Transformed coordinates (B, D) and log Jacobians (B,)
        """

        log_det = z.new_zeros(z.shape[0])

        # same indexing but reverse the two steps
        z_lower = z[:, self.inv_mask]
        z_upper = z[:, self.mask]

        # 1. inverse: first undo f2 on lower
        inp2 = torch.cat([z_upper, cond], dim=1) if cond is not None else z_upper
        params2 = self.f2(inp2).view(-1, self.lower_dim, 3 * self.K - 1)
        W2, H2, D2 = torch.split(params2, self.K, dim=2)
        z_lower, ld2 = splines.unconstrained_RQS(z_lower, W2, H2, D2, inverse=True, tail_bound=self.B)
        log_det = log_det + ld2.sum(1)

        # 2. then reverse f1 on upper
        inp1 = torch.cat([z_lower, cond], dim=1) if cond is not None else z_lower
        params1 = self.f1(inp1).view(-1, self.upper_dim, 3 * self.K - 1)
        W1, H1, D1 = torch.split(params1, self.K, dim=2)
        z_upper, ld1 = splines.unconstrained_RQS(z_upper, W1, H1, D1, inverse=True, tail_bound=self.B)
        log_det = log_det + ld1.sum(1)

        # scatter back into full D‑dim vector
        x = z.clone()
        x[:, self.mask]     = z_upper
        x[:, self.inv_mask] = z_lower

        return x, log_det
