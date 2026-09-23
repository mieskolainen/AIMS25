# MLPs, Transformers, Hutchinson etc
#
# m.mieskolainen@imperial.ac.uk, 2025

import torch
import torch.nn as nn
import numpy as np
import tqdm
import math

from torch.utils.data import TensorDataset, random_split, DataLoader

def split_loaders(
    X: torch.Tensor,
    y: torch.Tensor,
    frac: float = 0.9,
    batch_size: int = 128,
    shuffle: bool = True,
    seed: int = 42
):
    """Split paired tensors into training and validation loaders

    Args:
        X (Tensor): Inputs (N, ...)
        y (Tensor): Targets (N, ...)
        frac (float): Training fraction
        batch_size (int): Samples per batch
        shuffle (bool): Shuffle training batches
        seed (int): Split seed

    Returns:
        tuple[DataLoader, DataLoader]: Training and validation loaders
    """

    # 1) Create full dataset
    full_ds = TensorDataset(X, y)

    # 2) Compute split sizes
    n = len(full_ds)
    n_train = int(frac * n)
    n_val   = n - n_train
    
    # 3) Do the split with a fixed seed for reproducibility
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # 4) Wrap in DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def train_wrapper(model, optimizer, scheduler, train_loader, val_loader=None,
                  n_epochs=10, gradient_max=0.5, validation_seed=None, loss_fn=None):
    """Train a model and restore the best validation checkpoint when available

    Args:
        model (nn.Module): Model with loss(targets, conditions)
        optimizer (Optimizer): Parameter optimizer
        scheduler (LRScheduler): Scheduler accepting the epoch loss
        train_loader (DataLoader): Batches of targets and conditions
        val_loader (DataLoader | None): Optional validation batches
        n_epochs (int): Training epochs
        gradient_max (float): Maximum gradient norm
        validation_seed (int | None): Repeat validation draws without changing training RNG
        loss_fn (Callable | None): Per-example loss(targets, conditions), default model.loss

    Returns:
        tuple: ((train_losses, val_losses), model), losses are lists of length n_epochs
    """

    loss_fn = model.loss if loss_fn is None else loss_fn
    best_val_loss = math.inf
    best_state    = None
    
    print(f'Number of model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):0.1E}')
    
    train_losses = []
    val_losses   = []

    epoch_bar = tqdm.trange(n_epochs, desc="Training", unit="epoch")

    for epoch in epoch_bar:
        # --------------------
        # Training phase
        # --------------------
        model.train()
        epoch_loss = 0.0
        total_samples = 0

        for batch_x, batch_c in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(batch_x, batch_c).mean()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_max)
            optimizer.step()

            bsize = batch_x.size(0)
            epoch_loss  += loss.item() * bsize
            total_samples += bsize

        epoch_loss /= total_samples
        train_losses.append(epoch_loss)

        # --------------------
        # Validation phase
        # --------------------
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            total_val_samples = 0

            devices = sorted({p.device.index for p in model.parameters() if p.is_cuda})
            # Fork RNG only when requested, so validation draws repeat across
            # epochs and cannot advance or reset the training RNG sequence
            with torch.random.fork_rng(devices=devices, enabled=validation_seed is not None):
                if validation_seed is not None:
                    torch.random.default_generator.manual_seed(validation_seed)

                    for index in devices:
                        with torch.cuda.device(index):
                            torch.cuda.manual_seed(validation_seed)

                with torch.no_grad():
                    for batch_x, batch_c in val_loader:
                        loss = loss_fn(batch_x, batch_c).mean()
                        bsize = batch_x.size(0)
                        val_loss += loss.item() * bsize
                        total_val_samples += bsize

            val_loss /= total_val_samples
            val_losses.append(val_loss)

            # check for new best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            val_loss = None
            val_losses.append(None)

        # --------------------
        # Scheduler and logging
        # --------------------
        scheduler.step(val_loss if val_loss is not None else epoch_loss)
        current_lr = scheduler.get_last_lr()[0]

        postfix = {
            "train_loss": f"{epoch_loss:.4f}",
            "lr":         f"{current_lr:.6f}",
        }
        if val_loss is not None:
            postfix["val_loss"] = f"{val_loss:.4f}"

        epoch_bar.set_postfix(postfix)
    
    # restore best model (if any validation was done)
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return (train_losses, val_losses), model


def expand_dim(c, num_samples: int, device='cpu'):
    """Broadcast conditions to a float32 sample batch

    Args:
        c (array_like): Scalar, vector (C,), or conditions (S, C)
        num_samples (int): Sample count S
        device (str | torch.device): Output device

    Returns:
        Tensor: Conditions (S, C), with C=1 for scalar input
    """
    
    c_t = torch.as_tensor(c, device=device, dtype=torch.float32)
    
    if c_t.ndim == 0:   # scalar -> (num_samples, 1)
        return c_t.expand(num_samples, 1).clone()
    elif c_t.ndim == 1: # vector -> broadcast to every sample
        return c_t.unsqueeze(0).expand(num_samples, -1).clone()
    elif c_t.ndim == 2:
        if c_t.shape[0] != num_samples:
            raise ValueError(f"number of condition vectors ({c_t.shape[0]}) "
                             f"doesn't match n_samples ({num_samples})")

        return c_t
    else:
        raise ValueError(f"c must be scalar, 1‑D or 2‑D tensor, got ndim={c_t.ndim}")

def div_sandwich(v, x, t, cond, e, create_graph=False):
    """Evaluate one Hutchinson trace probe

    Args:
        v (Callable): Vector field (x, t, cond) -> Tensor (B, D)
        x (Tensor): Differentiable coordinates (B, D)
        t (Tensor): Times (B, 1)
        cond (Tensor | None): Conditions (B, C)
        e (Tensor): Probe vectors (B, D)
        create_graph (bool): Keep the derivative graph

    Returns:
        Tensor: Per-sample trace estimates (B,)
    """

    out = (v(x, t, cond) * e).sum()
    grad = torch.autograd.grad(out, x, create_graph=create_graph, retain_graph=True)[0]

    return (grad * e).sum(dim=1)

def div_hutchinson(v, x, t, cond=None,
                   n_samples=10,
                   noise_type="rademacher",
                   create_graph=False):
    """Estimate vector-field divergence with random trace probes

    Args:
        v (Callable): Vector field (x, t, cond) -> Tensor (B, D)
        x (Tensor): Differentiable coordinates (B, D)
        t (Tensor): Times (B, 1)
        cond (Tensor | None): Conditions (B, C)
        n_samples (int): Probe count
        noise_type (str): rademacher or gaussian
        create_graph (bool): Keep the derivative graph

    Returns:
        Tensor: Mean trace estimates (B,)
    """

    estimates = []
    # enable_grad once
    with torch.enable_grad():
        for _ in range(n_samples):
            if noise_type == "rademacher":
                e = (torch.randint(0, 2, x.shape, device=x.device, 
                                  dtype=x.dtype) * 2 - 1)
            elif noise_type == "gaussian":
                e = torch.randn_like(x)
            else:
                raise ValueError(f"Unsupported noise_type: {noise_type}")

            estimates.append(div_sandwich(v=v, x=x, t=t, cond=cond, e=e,
                                          create_graph=create_graph))

    return torch.stack(estimates, dim=0).mean(dim=0)


def sinusoidal_embedding(timesteps, dim):
    """Encode scalar times or noise levels with sinusoidal features

    Args:
        timesteps (Tensor): Scalar inputs (B, 1)
        dim (int): Embedding width, use an even value

    Returns:
        Tensor: Features (B, 2 * (dim // 2))
    """

    device = timesteps.device
    half_dim = dim // 2
    freqs = torch.exp(
        -np.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=device) / half_dim
    )
    angles = timesteps * freqs  # [batch_size, half_dim]
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

    return emb  # [batch_size, dim]

class GatedMLPBlock(nn.Module):
    """Gated MLP"""

    def __init__(self, in_dim, out_dim, layer_norm=False):
        """Build a sigmoid-gated linear block

        Args:
            in_dim (int): Input width
            out_dim (int): Output width
            layer_norm (bool): Enable layer normalization

        Returns:
            None
        """

        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim * 2)
        self.layer_norm = layer_norm
        if self.layer_norm:
            self.norm = nn.LayerNorm(out_dim * 2)

    def forward(self, x):
        """Apply the gated linear block

        Args:
            x (Tensor): Inputs (..., in_dim)

        Returns:
            Tensor: Outputs (..., out_dim)
        """

        x = self.linear(x)
        if self.layer_norm:
            x = self.norm(x)

        x1, x2 = x.chunk(2, dim=-1)

        return x1 * torch.sigmoid(x2)

def get_activation(act="relu"):
    
    """Construct an activation layer

    Args:
        act (str): relu, silu, tanh, elu, or gelu

    Returns:
        nn.Module: Activation layer
    """

    if   act == 'silu':
        return nn.SiLU()
    elif act == 'tanh':
        return nn.Tanh()
    elif act == 'relu':
        return nn.ReLU()
    elif act == 'elu':
        return nn.ELU()
    elif act == 'gelu':
        return nn.GELU()
    else:
        raise Exception("Unknown activation chosen")

class MLPBlock(nn.Module):
    def __init__(self, in_dim, out_dim, layer_norm=False, act='silu', dropout=0.0):
        """Build a linear block with activation and optional dropout

        Args:
            in_dim (int): Input width
            out_dim (int): Output width
            layer_norm (bool): Enable layer normalization
            act (str): Activation name
            dropout (float): Dropout probability

        Returns:
            None
        """

        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.act = get_activation(act)
        
        self.layer_norm = layer_norm
        if self.layer_norm:
            self.norm = nn.LayerNorm(out_dim)
        
        # Dropout layer (only active if dropout > 0)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    def forward(self, x):
        """Apply the linear block

        Args:
            x (Tensor): Inputs (..., in_dim)

        Returns:
            Tensor: Outputs (..., out_dim)
        """

        x = self.linear(x)
        if self.layer_norm:
            x = self.norm(x)

        x = self.act(x)
        # After activation typical choice
        if self.dropout is not None:
            x = self.dropout(x)

        return x

class ResidualMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=[128], layer_norm=False, act='silu',
                 dropout=0.0, apply_tanh=False, apply_sigmoid=False):
        """Build an MLP with projected residual connections

        Args:
            in_dim (int): Input width
            out_dim (int): Output width
            hidden_dim (list[int]): Hidden widths
            layer_norm (bool): Enable layer normalization
            act (str): Activation name or gated
            dropout (float): Dropout probability
            apply_tanh (bool): Apply tanh to the output
            apply_sigmoid (bool): Apply sigmoid to the output

        Returns:
            None
        """

        super().__init__()
        assert len(hidden_dim) >= 1, "hidden_dim must have at least one element"
        
        self.input_proj = nn.Linear(in_dim, hidden_dim[0])
        self.layer_norm    = layer_norm
        self.dropout       = nn.Dropout(dropout) if dropout > 0.0 else None
        self.apply_tanh    = apply_tanh
        self.apply_sigmoid = apply_sigmoid
        
        if self.layer_norm:
            self.norm_in = nn.LayerNorm(hidden_dim[0])
        
        self.act_fn = get_activation(act) if act != "gated" else nn.Identity()
        
        # Build hidden layers
        self.blocks = nn.ModuleList()
        self.skip_projections = nn.ModuleList()

        for i in range(len(hidden_dim) - 1):
            
            if act != 'gated':
                block = MLPBlock(in_dim=hidden_dim[i], out_dim=hidden_dim[i + 1], layer_norm=layer_norm, act=act, dropout=dropout)
            else:
                block = GatedMLPBlock(in_dim=hidden_dim[i], out_dim=hidden_dim[i + 1], layer_norm=layer_norm)

            self.blocks.append(block)
            # Preserve the identity shortcut unless the hidden width changes
            self.skip_projections.append(
                nn.Identity() if hidden_dim[i] == hidden_dim[i + 1]
                else nn.Linear(hidden_dim[i], hidden_dim[i + 1], bias=False))
        
        self.output_proj = nn.Linear(hidden_dim[-1], out_dim)
        
    def forward(self, x):
        """Evaluate the residual MLP

        Args:
            x (Tensor): Inputs (..., in_dim)

        Returns:
            Tensor: Outputs (..., out_dim)
        """

        x = self.input_proj(x)
        
        if self.layer_norm:
            x = self.norm_in(x)

        x = self.act_fn(x)
        
        for block, skip in zip(self.blocks, self.skip_projections):
            x = skip(x) + block(x)  # residual connection
        
        if self.dropout is not None:
            x = self.dropout(x)
        
        x = self.output_proj(x)
        
        if self.apply_tanh:
            x = torch.tanh(x)
        
        if self.apply_sigmoid:
            x = torch.sigmoid(x)
        
        return x

class VectorTransformer(nn.Module):
    """Encode positioned scalar features or unordered event vectors"""

    def __init__(self, in_dim: int, out_dim: int, d_model: int = 32,
                 nhead: int = 4, num_layers: int = 2,
                 dim_feedforward: int = 128, dropout: float = 0.05,
                 act: str = "relu", token_dim: int | None = None):
        """Build a transformer for scalar features or unordered event tokens

        Args:
            in_dim (int): Scalar feature count in vector mode
            out_dim (int): Summary width
            d_model (int): Token embedding width
            nhead (int): Attention heads
            num_layers (int): Encoder layers
            dim_feedforward (int): Feedforward width
            dropout (float): Dropout probability
            act (str): relu or gelu
            token_dim (int | None): Event width E, None uses positioned scalar tokens

        Returns:
            None
        """

        super().__init__()
        self.token_dim = token_dim
        self.embed = nn.Linear(1 if token_dim is None else token_dim, d_model)
        self.pos_embed = (nn.Parameter(torch.zeros(1, in_dim, d_model))
                          if token_dim is None else None)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation=act, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.fc_out = nn.Linear(d_model + (token_dim is not None), out_dim)

    def encode_tokens(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None):
        """Encode tokens with event permutation equivariance in set mode

        Args:
            x (Tensor): Vectors (B, N) or nonempty event sets (B, N, E)
            padding_mask (Tensor | None): Boolean mask (B, N), True ignores padding including NaN

        Returns:
            Tensor: Token embeddings (B, N, d_model), zero at padding
        """

        if self.token_dim is None:
            if x.ndim != 2 or x.shape[1] != self.pos_embed.shape[1]:
                raise ValueError("Expected (batch, in_dim) vectors.")

            x = x.unsqueeze(-1)
        elif x.ndim != 3 or x.shape[-1] != self.token_dim:
            raise ValueError("Expected (batch, events, token_dim) event sets.")

        if x.shape[0] == 0 or x.shape[1] == 0:
            raise ValueError("Every batch and event set must be nonempty.")

        if padding_mask is not None:
            if padding_mask.dtype != torch.bool or padding_mask.shape != x.shape[:2]:
                raise ValueError("padding_mask must be Boolean with shape (batch, events).")

            if torch.any(padding_mask.all(dim=1)):
                raise ValueError("Every event set must contain an unmasked event.")

            x = x.masked_fill(padding_mask.unsqueeze(-1), 0)

        x = self.embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0)

        return x

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None):
        """Pool token embeddings, including log event count in set mode

        Args:
            x (Tensor): Vectors (B, N) or nonempty event sets (B, N, E)
            padding_mask (Tensor | None): Boolean mask (B, N), True ignores padding including NaN

        Returns:
            Tensor: Summaries (B, out_dim), permutation invariant in set mode
        """

        tokens = self.encode_tokens(x, padding_mask)
        count = (tokens.new_full((tokens.shape[0], 1), tokens.shape[1])
                 if padding_mask is None else (~padding_mask).sum(dim=1, keepdim=True).to(tokens))
        pooled = tokens.sum(dim=1) / count
        if self.token_dim is not None:
            # Mean pooling alone cannot distinguish repeated identical events
            pooled = torch.cat([pooled, count.log()], dim=-1)

        return self.fc_out(pooled)


class SetTransformer(VectorTransformer):
    """Encode sets with equivariant attention and invariant pooling plus log event count

    Use eval() for deterministic permutation invariance with dropout
    Attention costs O(N**2) for N events
    """

    def __init__(self, event_dim: int, out_dim: int, include_moments=False, **kwargs):
        """Build an event-set encoder with invariant pooling

        Args:
            event_dim (int): Features E per event
            out_dim (int): Summary width
            include_moments (bool): Add raw feature means and second moments before projection
            **kwargs (dict): VectorTransformer attention options

        Returns:
            None
        """

        super().__init__(in_dim=event_dim, out_dim=out_dim, token_dim=event_dim, **kwargs)

        self.include_moments = include_moments
        if include_moments:
            self.fc_out = nn.Linear(self.embed.out_features + 2 * event_dim + 1, out_dim)

    def forward(self, x, padding_mask=None):
        """Pool attention and optional raw moments without losing count information

        Args:
            x (Tensor): Event sets (B, N, E)
            padding_mask (Tensor | None): Boolean mask (B, N), True for padding

        Returns:
            Tensor: Invariant context (B, out_dim)
        """

        if not self.include_moments:
            return super().forward(x, padding_mask)

        tokens = self.encode_tokens(x, padding_mask)
        count = (x.new_full((len(x), 1), x.shape[1]) if padding_mask is None
                 else (~padding_mask).sum(1, keepdim=True).to(x))
        valid = x if padding_mask is None else x.masked_fill(padding_mask[..., None], 0)
        pooled = torch.cat([
            tokens.sum(1) / count, valid.sum(1) / count,
            valid.square().sum(1) / count, count.log(),
        ], dim=-1)

        return self.fc_out(pooled)
