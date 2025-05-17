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
        pass

    # Diffusion coefficient
    @abstractmethod
    def g(self, t): 
        pass

    # Method to sample the forward (noising) process given an initial condition
    @abstractmethod
    def forward_sample(self, x0, t): 
        pass

class VPSDE(BaseSDE):
    """
    Variance Preserving (alpha^2 + sigma^2 = 1) SDE
    """
    def __init__(self, beta_0=0.1, beta_1=20.0):
        self.beta_0, self.beta_1 = beta_0, beta_1

    def beta(self, t):
        # Linear schedule: beta(t) = beta_0 + (beta_1 - beta_0) * t
        return self.beta_0 + (self.beta_1 - self.beta_0) * t

    def integral_beta(self, t):
        # Integral from 0 to t of beta(s) ds
        return self.beta_0 * t + 0.5 * (self.beta_1 - self.beta_0) * t**2
    
    def alpha_sigma(self, t):
        integral = self.integral_beta(t)
        alpha = torch.exp(-0.5 * integral)
        sigma = torch.sqrt(1.0 - alpha**2)
        return alpha, sigma

    def sigma(self, t):
        alpha, sigma = self.alpha_sigma(t)
        return sigma

    def f(self, x, t):
        # Drift: f(x,t) = -0.5 * beta(t) * x
        beta_t = self.beta(t).view(-1, *[1]*(x.ndim-1))
        return -0.5 * beta_t * x
    
    def g(self, t):
        # For VP SDE, we define g(t)=sqrt(beta(t)) for the reverse SDE drift
        beta_t = self.beta(t)
        return torch.sqrt(beta_t)
    
    def forward_sample(self, x0, t):
        # Forward SDE sample: x_t = alpha(t) * x0 + sigma(t) * eps, where eps ~ N(0,I).
        noise = torch.randn_like(x0)
        alpha, sigma = self.alpha_sigma(t)
        
        alpha = alpha.view(-1, *([1] * (x0.ndim - 1)))
        sigma = sigma.view(-1, *([1] * (x0.ndim - 1)))
        
        return alpha * x0 + sigma * noise, noise

    def lambda_weight(self, t):
        # Loss weight term
        return self.g(t)**2


# Unified SDE Model class with conditional support
class SDEModel(nn.Module):
    
    def __init__(self, sde, x_dim, nnet, cond_dim=None, time_embed_dim=8, loss_weighting=False, EPS=1e-5):
        
        super().__init__()
        self.sde = sde
        self.time_embed_dim = time_embed_dim
        self.x_dim = x_dim
        self.cond_dim = cond_dim if cond_dim is not None else 0
        self.EPS     = EPS
        self.score_net = nnet
        self.loss_weighting = loss_weighting
        
    def base_log_prob_fn(self, x: torch.Tensor):
        """ Base gaussian logpdf
        """
        return -0.5 * torch.sum(x**2, dim=1) - 0.5 * self.x_dim * math.log(2 * math.pi)
        
    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor=None):
        """ Hyvärinen score estimator network
        """
        t_emb = sinusoidal_embedding(t, self.time_embed_dim)
        
        if cond is not None:
            x_in = torch.cat([x, t_emb, cond], dim=1)
        else:
            x_in = torch.cat([x, t_emb], dim=1)
        
        score = self.score_net(x_in)
        
        return score
        
    def loss(self, x0: torch.Tensor, cond: torch.Tensor=None):
        """ Denoising score matching MSE loss
        
        score = -noise / sigma <=> noise = -sigma * score
        """
        batch_size = x0.shape[0]
        
        # EPS avoid exploding gradients when t -> 0
        t = self.EPS + (1.0 - self.EPS) * torch.rand(batch_size, 1, device=x0.device)
        
        xt, noise = self.sde.forward_sample(x0, t)
        sigma_t   = self.sde.sigma(t).view(-1, 1)
        score     = self.forward(x=xt, t=t, cond=cond)
        
        lambda_weight = self.sde.lambda_weight(t) if self.loss_weighting else 1.0
        
        # ||predicted - true||^2
        return (lambda_weight * (-sigma_t * score - noise)**2).sum(-1)
        
    def divergence(self, v, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor=None, exact: bool=True, n_hutchinson: int=10):
        """ Vector field divergence
        """
        
        with torch.enable_grad():

            x = x.detach().requires_grad_(True)
            
            # Exact Jacobian trace with autograd
            if exact:
                
                y = v(x,t,cond)
                assert y.shape == x.shape, f"v(x) must return shape {x.shape}, got {y.shape}"
                
                div = torch.zeros(x.shape[0], device=x.device)
                d   = x.shape[1]
                
                for i in range(d):
                    grad = torch.autograd.grad(outputs=y[:, i].sum(), inputs=x, create_graph=False, retain_graph=True)[0]  # shape: [N, D]
                    div += grad[:, i]  # ∂v_i / ∂x_i
                return div
            
            # Hutchinson approximate trace with autograd
            else:
                return div_hutchinson(v=v, x=x, t=t, cond=cond, n_samples=n_hutchinson)
    
    def log_prob(self, x0: torch.Tensor, cond: torch.Tensor=None, T: float=1.0, steps: int=1000, exact: bool=True, n_hutchinson: int=10, EPS: float=None):
        """
        Data log density (likelihood) using the probability flow ODE from t=0 -> t=T
        Conditional variable cond is passed to the score network if provided
        """
        if EPS is None:
            EPS = self.EPS
        
        dt      = (T - EPS) / steps # Positive !
        x       = x0.clone()
        div_tot = torch.zeros(x.shape[0], device=x.device)
        
        batch_size = x.shape[0]
        device     = next(self.parameters()).device
        
        for i in range(steps):
            t = self.EPS + i * dt
            t = torch.full((batch_size, 1), t, device=device, dtype=torch.float32)
            
            # Probability flow (ODE)
            def vf(x_, t_, c_):
                    
                g_t = self.sde.g(t_).view(-1, 1)
                return self.sde.f(x_, t_) - 0.5 * g_t**2 * self.forward(x_, t_, c_)
            
            # Euler updates
            div_tot = div_tot + self.divergence(vf, x=x, t=t, cond=cond, exact=exact, n_hutchinson=n_hutchinson) * dt
            x       = x + vf(x, t, cond) * dt # After divergence update
            
        # p(x_data) = p(x_T) + \int divergence (take care with the sign of the divergence here)
        return self.base_log_prob_fn(x) + div_tot
        
    def sample(self, num_samples: int, cond: torch.Tensor=None, T: float=1.0, steps: int=1000, use_ode: bool=False, EPS: float=None):
        """
        Stochastic Euler-Maruyama sampler using the reverse-time SDE from t=T -> t=0
        """
        if use_ode:
            return self.sample_ode(num_samples=num_samples, cond=cond, T=T, steps=steps, EPS=EPS)
        
        if EPS is None:
            EPS = self.EPS
        
        dt     = -(T - EPS) / steps # Negative!
        device = next(self.parameters()).device
        
        # Initial condition
        sigma_T = self.sde.sigma(torch.tensor([T], device=device)) # approx 1 with typical parameters
        x       = sigma_T * torch.randn((num_samples, self.x_dim), device=device)
        
        for i in range(steps):
            t     = T + i * dt
            t     = torch.full((num_samples, 1), t, device=device, dtype=torch.float32)
            
            g_t   = self.sde.g(t).view(-1, 1)
            drift = self.sde.f(x, t) - g_t**2 * self.forward(x, t, cond)
            
            # Euler-Murayama update
            x     = x + drift * dt + g_t * torch.sqrt(torch.tensor(-dt, device=device)) * torch.randn_like(x)
        
        return x
        
    def sample_ode(self, num_samples: int, cond: torch.Tensor=None, T: float=1.0, steps: int=1000, EPS: float=None):
        """
        Deterministic Euler (1st order) sampler using the probability flow ODE in reverse t=T -> t=0
        """
        if EPS is None:
            EPS = self.EPS
        
        dt     = -(T - EPS) / steps # Negative!
        device = next(self.parameters()).device
        
        # Initial condition
        sigma_T = self.sde.sigma(torch.tensor([T], device=device)) # approx 1 with typical parameters
        x       = sigma_T * torch.randn((num_samples, self.x_dim), device=device)
        
        for i in range(steps):
            t   = T + i * dt
            t   = torch.full((num_samples, 1), t, device=device, dtype=torch.float32)
            
            g_t = self.sde.g(t).view(-1, 1)
            v   = self.sde.f(x, t) - 0.5 * g_t**2 * self.forward(x, t, cond)
            
            # Euler update
            x   = x + v * dt
        
        return x
