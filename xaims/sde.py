# SDE continuous-time diffusion
#
# Song et al., https://arxiv.org/abs/2011.13456
#
# m.mieskolainen@imperial.ac.uk, 2025

import torch
import torch.nn as nn
import math
from abc import ABC, abstractmethod

from .aux import sinusoidal_embedding, div_hutchinson

# Base SDE interface
class BaseSDE(ABC):

    # Drift function
    @abstractmethod
    def f(self, x, t):
        """Evaluate the forward drift

        Args:
            x (Tensor): States (B, D)
            t (Tensor): Times (B, 1)

        Returns:
            Tensor: Drift (B, D)
        """

        pass

    # Diffusion coefficient
    @abstractmethod
    def g(self, t):
        """Evaluate the diffusion coefficient

        Args:
            t (Tensor): Times (B, 1)

        Returns:
            Tensor: Diffusion scales (B, 1)
        """

        pass

    # Method to sample the forward (noising) process given an initial condition
    @abstractmethod
    def forward_sample(self, x0, t):
        """Draw noisy states from the forward transition

        Args:
            x0 (Tensor): Initial states (B, D)
            t (Tensor): Times (B, 1)

        Returns:
            tuple[Tensor, Tensor]: Noisy states and standard-normal noise, each (B, D)
        """

        pass

class VPSDE(BaseSDE):
    """Variance Preserving (alpha^2 + sigma^2 = 1) SDE"""

    def __init__(self, beta_0=0.1, beta_1=20.0):
        """Set the linear variance-preserving noise schedule

        Args:
            beta_0 (float): Noise rate at t=0
            beta_1 (float): Noise rate at t=1

        Returns:
            None
        """

        self.beta_0, self.beta_1 = beta_0, beta_1

    def beta(self, t):
        # Linear schedule: beta(t) = beta_0 + (beta_1 - beta_0) * t
        """Evaluate the linear noise rate

        Args:
            t (Tensor): Times (...)

        Returns:
            Tensor: Noise rates with the same shape as t
        """

        return self.beta_0 + (self.beta_1 - self.beta_0) * t

    def integral_beta(self, t):
        # Integral from 0 to t of beta(s) ds
        """Integrate the noise rate from zero to each time

        Args:
            t (Tensor): Times (...)

        Returns:
            Tensor: Integrated rates with the same shape as t
        """

        return self.beta_0 * t + 0.5 * (self.beta_1 - self.beta_0) * t**2

    def alpha_sigma(self, t):
        """Evaluate signal and conditional noise scales

        Args:
            t (Tensor): Times (...)

        Returns:
            tuple[Tensor, Tensor]: Alpha and sigma, each with the same shape as t
        """

        integral = self.integral_beta(t)
        alpha = torch.exp(-0.5 * integral)
        sigma = torch.sqrt(-torch.expm1(-integral))

        return alpha, sigma

    def sigma(self, t):
        """Evaluate the forward conditional noise scale

        Args:
            t (Tensor): Times (...)

        Returns:
            Tensor: Noise scales with the same shape as t
        """

        alpha, sigma = self.alpha_sigma(t)

        return sigma

    def f(self, x, t):
        # Drift: f(x,t) = -0.5 * beta(t) * x
        """Evaluate the forward drift

        Args:
            x (Tensor): States (B, D)
            t (Tensor): Times (B, 1)

        Returns:
            Tensor: Drift (B, D)
        """

        beta_t = self.beta(t).view(-1, *[1]*(x.ndim-1))

        return -0.5 * beta_t * x

    def g(self, t):
        # For VP SDE, we define g(t)=sqrt(beta(t)) for the reverse SDE drift
        """Evaluate the diffusion coefficient

        Args:
            t (Tensor): Times (B, 1)

        Returns:
            Tensor: Diffusion scales (B, 1)
        """

        beta_t = self.beta(t)

        return torch.sqrt(beta_t)

    def forward_sample(self, x0, t):
        # Forward SDE sample: x_t = alpha(t) * x0 + sigma(t) * eps, where eps ~ N(0,I)
        """Draw noisy states from the forward transition

        Args:
            x0 (Tensor): Initial states (B, D)
            t (Tensor): Times (B, 1)

        Returns:
            tuple[Tensor, Tensor]: Noisy states and standard-normal noise, each (B, D)
        """

        noise = torch.randn_like(x0)
        alpha, sigma = self.alpha_sigma(t)

        alpha = alpha.view(-1, *([1] * (x0.ndim - 1)))
        sigma = sigma.view(-1, *([1] * (x0.ndim - 1)))

        return alpha * x0 + sigma * noise, noise

    def lambda_weight(self, t):
        # Loss weight term
        """Evaluate squared diffusion coefficients for loss weighting

        Args:
            t (Tensor): Times (...)

        Returns:
            Tensor: Loss weights with the same shape as t
        """

        return self.g(t)**2


# Unified SDE Model class with conditional support
class SDEModel(nn.Module):
    """Conditional diffusion with a standard-normal terminal law

    Run the forward process long enough to approach the terminal law
    For narrow posteriors, v prediction, log_snr sampling, and time_grid_power=2
    improve training and low-noise resolution
    """

    def __init__(self, sde, x_dim, nnet, cond_dim=None, time_embed_dim=8,
                 loss_weighting=False, EPS=1e-5, prediction_type="score",
                 time_sampling="uniform", time_grid_power=1.0):

        """Build a conditional score model with a Gaussian terminal law

        Args:
            sde (BaseSDE): Forward process with sigma and lambda_weight methods
            x_dim (int): Data width D
            nnet (nn.Module): Predict score or velocity from data, time, and conditions
            cond_dim (int | None): Condition width C
            time_embed_dim (int): Even time embedding width
            loss_weighting (bool): Use likelihood-weighted denoising loss
            EPS (float): Minimum diffusion time
            prediction_type (str): score or v, velocity requires VPSDE
            time_sampling (str): uniform or log_snr, log_snr requires VPSDE
            time_grid_power (float): Grid exponent >= 1, larger values resolve low noise

        Returns:
            None
        """

        super().__init__()
        self.sde = sde
        self.time_embed_dim = time_embed_dim
        self.x_dim = x_dim
        self.cond_dim = cond_dim if cond_dim is not None else 0
        self.EPS     = EPS
        self.score_net = nnet
        self.loss_weighting = loss_weighting
        if prediction_type not in ("score", "v") or time_sampling not in ("uniform", "log_snr"):
            raise ValueError("Unknown diffusion prediction type or time sampling rule.")

        if (prediction_type == "v" or time_sampling == "log_snr") and not isinstance(sde, VPSDE):
            raise ValueError("Velocity prediction and log-SNR sampling require VPSDE.")

        if time_grid_power < 1 or not math.isfinite(time_grid_power):
            raise ValueError("time_grid_power must be finite and >= 1.")

        if (prediction_type == "v" or time_sampling == "log_snr") and not 0 < EPS < 1:
            raise ValueError("Noise-aware training requires 0 < EPS < 1.")

        self.prediction_type = prediction_type
        self.time_sampling = time_sampling
        self.time_grid_power = time_grid_power

    def base_log_prob_fn(self, x: torch.Tensor):
        """Evaluate the joint standard-normal terminal log density

        Args:
            x (Tensor): Terminal states (B, D)

        Returns:
            Tensor: Log densities (B,)
        """

        return -0.5 * torch.sum(x**2, dim=1) - 0.5 * self.x_dim * math.log(2 * math.pi)

    def network_prediction(self, x, t, cond=None):
        """Predict score or VP velocity with noise-aware time features

        Args:
            x (Tensor): Noisy states (B, D)
            t (Tensor): Times (B, 1)
            cond (Tensor | None): Conditions (B, C)

        Returns:
            Tensor: Network predictions (B, D)
        """

        embedding_time = t
        if self.prediction_type == "v":
            alpha, sigma = self.sde.alpha_sigma(t)
            embedding_time = (sigma / alpha).log()

        time_features = sinusoidal_embedding(embedding_time, self.time_embed_dim)
        inputs = [x, time_features] if cond is None else [x, time_features, cond]

        return self.score_net(torch.cat(inputs, dim=1))

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor=None):
        """Evaluate the score regardless of network parameterization

        Args:
            x (Tensor): Noisy states (B, D)
            t (Tensor): Times (B, 1)
            cond (Tensor | None): Conditions (B, C)

        Returns:
            Tensor: Scores of the noised density (B, D)
        """

        prediction = self.network_prediction(x, t, cond)
        if self.prediction_type == "v":
            alpha, sigma = self.sde.alpha_sigma(t)

            return -x - (alpha / sigma) * prediction

        return prediction

    def training_times(self, x):
        """Draw times uniformly in time or log noise-to-signal ratio

        Args:
            x (Tensor): Batch (B, D), supplying device and dtype

        Returns:
            Tensor: Training times (B, 1) in [EPS, 1]
        """

        u = torch.rand(len(x), 1, device=x.device, dtype=x.dtype)
        if self.time_sampling == "uniform":
            return self.EPS + (1 - self.EPS) * u

        ends = x.new_tensor([self.EPS, 1.0])
        alpha, sigma = self.sde.alpha_sigma(ends)
        log_noise = (sigma / alpha).log()
        log_ratio = log_noise[0] + u * (log_noise[1] - log_noise[0])
        integral = torch.nn.functional.softplus(2 * log_ratio)
        beta0, beta1 = self.sde.beta_0, self.sde.beta_1

        return 2 * integral / (beta0 + torch.sqrt(beta0**2 + 2 * (beta1 - beta0) * integral))

    def loss(self, x0: torch.Tensor, cond: torch.Tensor=None):
        """Compute denoising error in noise or velocity coordinates

        Args:
            x0 (Tensor): Clean data (B, D)
            cond (Tensor | None): Conditions (B, C)

        Returns:
            Tensor: Per-sample losses (B,), optionally likelihood weighted
        """

        t = self.training_times(x0)
        xt, noise = self.sde.forward_sample(x0, t)
        sigma = self.sde.sigma(t).view(-1, 1)
        prediction = self.network_prediction(xt, t, cond)
        if self.prediction_type == "v":
            alpha, _ = self.sde.alpha_sigma(t)
            error = prediction - (alpha * noise - sigma * x0)
            weight = self.sde.lambda_weight(t) * alpha**2 / sigma**2 if self.loss_weighting else 1.0
        else:
            error = -sigma * prediction - noise
            weight = self.sde.lambda_weight(t) / sigma**2 if self.loss_weighting else 1.0

        return (weight * error**2).sum(-1)

    def integration_grid(self, reference, T, EPS, steps, reverse=False):
        """Construct a power-spaced integration grid

        Args:
            reference (Tensor): Supplies device and dtype
            T (float): Terminal time
            EPS (float): Minimum time
            steps (int): Integration step count
            reverse (bool): Order times from T to EPS

        Returns:
            Tensor: Integration times (steps + 1,)
        """

        if steps < 1 or not 0 <= EPS < T or (EPS == 0 and self.prediction_type == "v"):
            raise ValueError("Require steps >= 1 and 0 <= EPS < T, with EPS > 0 for velocity prediction.")

        grid = torch.linspace(0, 1, steps + 1, device=reference.device, dtype=reference.dtype)
        grid = EPS + (T - EPS) * grid.pow(self.time_grid_power)

        return grid.flip(0) if reverse else grid

    def divergence(self, v, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor=None, exact: bool=True, n_hutchinson: int=10, return_value: bool=False):
        """Evaluate vector-field divergence exactly or with Hutchinson probes

        Args:
            v (Callable): Vector field (x, t, cond) -> Tensor (B, D)
            x (Tensor): Differentiable coordinates (B, D)
            t (Tensor): Times (B, 1)
            cond (Tensor | None): Conditions (B, C)
            exact (bool): Use the exact Jacobian trace
            n_hutchinson (int): Probe count when exact=False
            return_value (bool): Also return the detached vector field

        Returns:
            Tensor | tuple[Tensor, Tensor]: Divergence (B,), optionally with field (B, D)
        """

        with torch.enable_grad():

            x = x.detach().requires_grad_(True)

            # Exact Jacobian trace with autograd
            if exact:

                y = v(x,t,cond)
                assert y.shape == x.shape, f"v(x) must return shape {x.shape}, got {y.shape}"

                div = x.new_zeros(x.shape[0])
                d   = x.shape[1]

                for i in range(d):
                    grad = torch.autograd.grad(outputs=y[:, i].sum(), inputs=x, create_graph=False, retain_graph=(not return_value or i < d - 1))[0]  # shape: [N, D]
                    div += grad[:, i]  # ∂v_i / ∂x_i

                return (div, y.detach()) if return_value else div

            # Hutchinson approximate trace with autograd
            else:
                div = div_hutchinson(v=v, x=x, t=t, cond=cond, n_samples=n_hutchinson)

                return (div, v(x, t, cond).detach()) if return_value else div

    def log_prob(self, x0: torch.Tensor, cond: torch.Tensor=None, T: float=1.0, steps: int=1000, exact: bool=True, n_hutchinson: int=10, EPS: float=None):
        """Estimate joint log density with the probability-flow ODE

        Args:
            x0 (Tensor): Data (B, D)
            cond (Tensor | None): Conditions (B, C)
            T (float): Terminal time
            steps (int): Euler integration step count
            exact (bool): Use exact divergence
            n_hutchinson (int): Probe count when exact=False
            EPS (float | None): Minimum time, default model EPS

        Returns:
            Tensor: Log densities (B,) under the Gaussian terminal law
        """

        if EPS is None:
            EPS = self.EPS

        x       = x0.clone()
        time_grid = self.integration_grid(x, T, EPS, steps)
        div_tot = x.new_zeros(x.shape[0])

        batch_size = x.shape[0]
        device     = next(self.parameters()).device

        for i in range(steps):
            dt = time_grid[i + 1] - time_grid[i]
            t = time_grid[i].expand(batch_size, 1)

            # Probability flow (ODE)
            def vf(x_, t_, c_):

                """Evaluate the probability-flow vector field

                Args:
                    x_ (Tensor): States (B, D)
                    t_ (Tensor): Times (B, 1)
                    c_ (Tensor | None): Conditions (B, C)

                Returns:
                    Tensor: ODE velocities (B, D)
                """

                g_t = self.sde.g(t_).view(-1, 1)

                return self.sde.f(x_, t_) - 0.5 * g_t**2 * self.forward(x_, t_, c_)

            # Euler updates
            if exact and not torch.is_grad_enabled():
                # Inference needs both the field and its divergence at the same
                # point: one network evaluation suffices. Keep the existing
                # gradient-enabled and Hutchinson paths unchanged
                div, velocity = self.divergence(vf, x=x, t=t, cond=cond,
                                                exact=True, return_value=True)
            else:
                div = self.divergence(vf, x=x, t=t, cond=cond, exact=exact,
                                      n_hutchinson=n_hutchinson)
                velocity = vf(x, t, cond)

            div_tot = div_tot + div * dt
            x = x + velocity * dt

        # p(x_data) = p(x_T) + \int divergence (take care with the sign of the divergence here)
        return self.base_log_prob_fn(x) + div_tot

    def sample(self, num_samples: int, cond: torch.Tensor=None, T: float=1.0, steps: int=1000, use_ode: bool=False, EPS: float=None):
        """Generate data from a standard-normal terminal distribution

        Args:
            num_samples (int): Draw count S
            cond (Tensor | None): Conditions (S, C)
            T (float): Terminal time
            steps (int): Euler integration step count
            use_ode (bool): Use Euler ODE sampling instead of Euler-Maruyama
            EPS (float | None): Minimum time, default model EPS

        Returns:
            Tensor: Samples (S, D)
        """

        if use_ode:
            return self.sample_ode(num_samples=num_samples, cond=cond, T=T, steps=steps, EPS=EPS)

        if EPS is None:
            EPS = self.EPS

        reference = next(self.parameters())

        # Match the standard-normal terminal density in base_log_prob_fn
        x = torch.randn((num_samples, self.x_dim), device=reference.device, dtype=reference.dtype)
        time_grid = self.integration_grid(x, T, EPS, steps, reverse=True)

        for i in range(steps):
            dt = time_grid[i + 1] - time_grid[i]
            t = time_grid[i].expand(num_samples, 1)

            g_t   = self.sde.g(t).view(-1, 1)
            drift = self.sde.f(x, t) - g_t**2 * self.forward(x, t, cond)

            # Euler-Murayama update
            x     = x + drift * dt + g_t * torch.sqrt(-dt) * torch.randn_like(x)

        return x

    def sample_ode(self, num_samples: int, cond: torch.Tensor=None, T: float=1.0, steps: int=1000, EPS: float=None):
        """Generate samples with the probability-flow ODE

        Args:
            num_samples (int): Draw count S
            cond (Tensor | None): Conditions (S, C)
            T (float): Terminal time
            steps (int): Euler integration step count
            EPS (float | None): Minimum time, default model EPS

        Returns:
            Tensor: Samples (S, D)
        """

        if EPS is None:
            EPS = self.EPS

        reference = next(self.parameters())

        # Match the standard-normal terminal density in base_log_prob_fn
        x = torch.randn((num_samples, self.x_dim), device=reference.device, dtype=reference.dtype)
        time_grid = self.integration_grid(x, T, EPS, steps, reverse=True)

        for i in range(steps):
            dt = time_grid[i + 1] - time_grid[i]
            t = time_grid[i].expand(num_samples, 1)

            g_t = self.sde.g(t).view(-1, 1)
            v   = self.sde.f(x, t) - 0.5 * g_t**2 * self.forward(x, t, cond)

            # Euler update
            x   = x + v * dt

        return x
