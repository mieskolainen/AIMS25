# DDPM discrete-time diffusion (VP-SDE as a Markov process)
#
# Ho et al, https://arxiv.org/abs/2006.11239
#
# m.mieskolainen@imperial.ac.uk, 2025

import torch
import torch.nn as nn
import numpy as np

from .aux import sinusoidal_embedding

class DDPM(nn.Module):
    def __init__(self,
                 x_dim,
                 cond_dim,
                 nnet,
                 beta_start=1e-4,
                 beta_end=0.02,
                 diffusion_steps=1000,
                 time_embed_dim=8
        ):
        
        super().__init__()
        self.x_dim = x_dim
        self.cond_dim = cond_dim
        self.time_embed_dim = time_embed_dim
        self.diffusion_steps = diffusion_steps
        
        self.net = nnet
        
        self.register_buffer("betas", torch.linspace(beta_start, beta_end, diffusion_steps))
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alphas_bar", torch.cumprod(self.alphas, dim=0))
        
    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor=None):
        """ Predict noise
        """
        t_emb = sinusoidal_embedding(t, self.time_embed_dim)
        
        if cond is not None:
            x_in = torch.cat([x, t_emb, cond], dim=1)
        else:
            x_in = torch.cat([x, t_emb], dim=1)
        
        return self.net(x_in)

    def loss_wrapper(self, x: torch.Tensor, cond: torch.Tensor=None):
        """ Returns predicted noise and true noise
        """
        device      = x.device
        batch_dim   = x.shape[0]
        
        t_idx       = torch.randint(0, self.diffusion_steps, (batch_dim,), device=device)
        t_norm      = (t_idx.float() / self.diffusion_steps).unsqueeze(1)  # [batch_dim, 1]
        alpha_bar_t = self.alphas_bar[t_idx].unsqueeze(1)  # [batch_dim, 1]
        
        noise       = torch.randn_like(x)
        xt          = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * noise
        noise_pred  = self.forward(xt, t_norm, cond)
        
        return noise_pred, noise
    
    def loss(self, x: torch.Tensor, cond: torch.Tensor=None):
        """ MSE loss
        """
        noise_pred, noise = self.loss_wrapper(x=x, cond=cond)
        return ((noise_pred - noise)**2).sum(-1)
    
    def sample(self, num_samples: int, cond: torch.Tensor=None):
        """ Sample from the learned distribution
        """
        device = self.betas.device
        x      = torch.randn(num_samples, self.x_dim, device=device)
        
        for t in reversed(range(self.diffusion_steps)):
            
            t_norm     = torch.full((num_samples, 1), t / self.diffusion_steps, device=device)
            noise_pred = self.forward(x=x, t=t_norm, cond=cond)
            
            beta_t      = self.betas[t]
            alpha_t     = self.alphas[t]
            alpha_bar_t = self.alphas_bar[t]
            
            mean = (1 / torch.sqrt(alpha_t)) * (
                x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * noise_pred
            )
            
            noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
            x = mean + torch.sqrt(beta_t) * noise
        
        return x
