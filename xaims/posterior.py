# Bayesian posterior combination and dataset-level neural amortization
#
# m.mieskolainen@imperial.ac.uk, 2025

from functools import partial

import numpy as np
from scipy.integrate import cumulative_trapezoid
import torch
from torch import nn
from torch.utils.data import Dataset


def combine_log_posteriors(log_event_posteriors, log_prior):
    """Combine independent-event posteriors, counting the shared prior once

    Args:
        log_event_posteriors (array_like): Log densities (P, N)
        log_prior (array_like): Shared training-prior log density (P,)

    Returns:
        np.ndarray: Unnormalized log posterior (P,), -inf outside prior support
    """
    
    events = np.asarray(log_event_posteriors, dtype=np.float64)
    prior = np.asarray(log_prior, dtype=np.float64)
    
    if events.ndim != 2 or prior.shape != events.shape[:1] or events.shape[1] < 1:
        raise ValueError("Expected (n_grid, n_events >= 1) and (n_grid,) inputs.")

    if np.any(np.isnan(prior) | np.isposinf(prior)):
        raise ValueError("Prior log densities must be finite or -inf.")

    # Mask unsupported points before subtracting log priors
    supported = np.isfinite(prior)
    values = events[supported]
    
    if np.any(np.isnan(values) | np.isposinf(values)):
        raise ValueError("Event log densities must be finite or -inf on prior support.")
    
    result = np.full(prior.shape, -np.inf)
    result[supported] = values.sum(axis=1) - (events.shape[1] - 1) * prior[supported]
    
    return result


@torch.no_grad()
def sum_event_log_probs(log_prob, parameters, data, batch_size=8192):
    """Sum event log densities in bounded batches, without prior or Jacobian corrections

    Args:
        log_prob (Callable): (parameters=(B, D), cond=(B, E)) -> Tensor (B,)
        parameters (Tensor): Preprocessed parameter vectors (P, D)
        data (Tensor): Preprocessed events (N, E), on parameters.device
        batch_size (int): Maximum parameter/event pairs per call

    Returns:
        Tensor: Float64 log-density sums (P,), on parameters.device
    """

    if (
        parameters.ndim != 2 or data.ndim != 2 or len(data) == 0
        or parameters.shape[1] == 0 or data.shape[1] == 0
    ):
        raise ValueError("Expected (points, parameter_dim) and nonempty (events, event_dim).")

    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")

    if parameters.device != data.device:
        raise ValueError("Parameters and data must be on the same device.")

    result = torch.zeros(len(parameters), device=parameters.device, dtype=torch.float64)
    points_per_batch = max(1, batch_size // min(len(data), batch_size))

    for first in range(0, len(parameters), points_per_batch):
        points = parameters[first:first + points_per_batch]
        events_per_batch = batch_size // len(points)

        for start in range(0, len(data), events_per_batch):
            events = data[start:start + events_per_batch]
            values = log_prob(
                points.repeat_interleave(len(events), dim=0),
                cond=events.repeat(len(points), 1),
            )

            if values.shape != (len(points) * len(events),):
                raise ValueError("log_prob must return one value per parameter/event pair.")

            result[first:first + len(points)] += (
                values.reshape(len(points), len(events)).double().sum(1)
            )

    return result


def normalize_grid_posterior(points, log_posterior):
    """Normalize a scalar-parameter log density by trapezoidal quadrature

    Args:
        points (array_like): Increasing grid (P,) covering the posterior mass
        log_posterior (array_like): Unnormalized log density (P,)

    Returns:
        tuple[np.ndarray, np.ndarray]: Normalized PDF and CDF, each (P,)
    """

    points = np.asarray(points, dtype=np.float64)
    log_posterior = np.asarray(log_posterior, dtype=np.float64)
    
    if (points.ndim != 1 or points.size < 2 or log_posterior.shape != points.shape
            or not np.all(np.isfinite(points)) or np.any(np.diff(points) <= 0)):
        raise ValueError("Expected matching 1D arrays and a finite, increasing grid.")
    
    if (np.any(np.isnan(log_posterior) | np.isposinf(log_posterior))
            or not np.any(np.isfinite(log_posterior))):
        raise ValueError("Log posterior must have finite mass and no NaN or +inf.")
    
    density = np.exp(log_posterior - np.max(log_posterior))
    cdf = cumulative_trapezoid(density, points, initial=0.0)
    integral = cdf[-1]
    
    return density / integral, cdf / integral


def grid_quantiles(points, density, probabilities):
    """Invert a piecewise-linear grid PDF, including zero-density gaps

    Args:
        points (array_like): Increasing parameter grid (P,)
        density (array_like): Nonnegative density values (P,)
        probabilities (array_like): Quantile levels (...) in [0, 1]

    Returns:
        np.ndarray: Parameter quantiles with the same shape as probabilities
    """

    points = np.asarray(points, dtype=np.float64)
    density = np.asarray(density, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    
    if np.any(~np.isfinite(density)) or np.any(density < 0):
        raise ValueError("PDF values must be finite and nonnegative.")

    with np.errstate(divide="ignore"):
        density, cdf = normalize_grid_posterior(points, np.log(density))

    if np.any(~np.isfinite(probabilities)) or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities must lie in [0, 1].")

    index = np.clip(np.searchsorted(cdf, probabilities, side="right") - 1, 0, len(points) - 2)
    width = points[index + 1] - points[index]
    left, right = density[index], density[index + 1]
    
    # Integrated mass inside a cell is quadratic in the fractional position
    mass = (probabilities - cdf[index]) / width
    discriminant = np.maximum(left**2 + 2 * (right - left) * mass, 0)
    denominator = left + np.sqrt(discriminant)
    fraction = np.divide(2 * mass, denominator, out=np.zeros_like(mass), where=denominator > 0)
    result = points[index] + width * np.clip(fraction, 0, 1)
    
    return np.where(probabilities == 1, points[-1], np.where(probabilities == 0, points[0], result))


@torch.no_grad()
def dataset_log_posterior(parameters, events, model, log_prior, *,
                          parameter_transform=None, batch_size=8192,
                          log_prob_kwargs=None):
    """Combine event posteriors with physical-coordinate Jacobians and one shared prior

    Args:
        parameters (Tensor): Physical parameter vectors (P, D), on events.device
        events (Tensor): Preprocessed observations (N, C), with model dtype and device
        model (object): log_prob(parameters, cond) -> joint log densities (B,)
        log_prior (Callable): Physical parameters (P, D) -> joint prior log densities (P,)
        parameter_transform (nn.Module | None): forward(theta, return_logdet=True) returns
            model coordinates (P, D) and log Jacobians (P,) or scalar
        batch_size (int): Maximum parameter/event pairs per density call
        log_prob_kwargs (dict | None): Density options, including diffusion solver settings

    Returns:
        Tensor: Unnormalized float64 log posterior (P,), -inf outside prior support
    """

    if (parameters.ndim != 2 or events.ndim != 2 or len(events) == 0
            or parameters.shape[1] == 0 or events.shape[1] == 0):
        raise ValueError("Expected parameters (P, D) and nonempty events (N, C)")

    if parameters.device != events.device:
        raise ValueError("Parameters and events must be on the same device")

    if not parameters.is_floating_point() or not events.is_floating_point():
        raise ValueError("Parameters and events must be floating tensors")

    prior = torch.as_tensor(log_prior(parameters), device=parameters.device, dtype=torch.float64)
    if prior.shape != (len(parameters),) or torch.any(torch.isnan(prior) | torch.isposinf(prior)):
        raise ValueError("log_prior must return (P,) joint log densities, finite or -inf")

    supported = torch.isfinite(prior)
    result = torch.full_like(prior, -torch.inf)
    if not torch.any(supported):
        return result

    if isinstance(model, nn.Module):
        model.eval()

    points = parameters[supported].to(events)
    ldj = points.new_zeros(len(points))
    if parameter_transform is not None:
        points, ldj = parameter_transform.forward(points, return_logdet=True)

    ldj = torch.as_tensor(ldj, device=events.device, dtype=torch.float64)
    if ldj.shape not in ((), (1,), (len(points),)) or not torch.all(torch.isfinite(ldj)):
        raise ValueError("Parameter log Jacobians must be finite, scalar or shape (P,)")

    log_sum = sum_event_log_probs(
        partial(model.log_prob, **(log_prob_kwargs or {})),
        points, events, batch_size=batch_size,
    )
    if torch.any(torch.isnan(log_sum) | torch.isposinf(log_sum)):
        raise ValueError("Event log densities must be finite or -inf on prior support")

    result[supported] = log_sum + len(events) * ldj - (len(events) - 1) * prior[supported]

    return result


@torch.no_grad()
def grid_posterior_1d(events, model, prior, *, event_transform=None,
                      parameter_transform=None, n_points=101, refinements=2,
                      refinement_points=201, tail_probability=1e-6,
                      credible_level=0.95, batch_size=8192, log_prob_kwargs=None):
    """Refine a scalar posterior grid, retaining coarse tails and checking resolution

    Args:
        events (Tensor): Observed events (N, E), with model dtype and device
        model (object): Single-event log_prob(parameters, cond) interface
        prior (object): Scalar prior with logpdf, support, and ppf methods
        event_transform (Callable | None): Events (N, E) -> model conditions (N, C)
        parameter_transform (nn.Module | None): Physical (P, 1) -> model coordinates and log Jacobians
        n_points (int): Initial grid size >= 2
        refinements (int): Number of peak refinements >= 0
        refinement_points (int): Points per refinement >= 2
        tail_probability (float): Prior tail cutoff in (0, 0.5) for infinite bounds
        credible_level (float): Interval mass in (0, 1) for the resolution check
        batch_size (int): Maximum parameter/event pairs per density call
        log_prob_kwargs (dict | None): Density options, including diffusion solver settings

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Grid, log posterior, PDF, CDF, each (P,)
    """

    if not isinstance(events, torch.Tensor) or events.ndim != 2 or min(events.shape) == 0:
        raise ValueError("Provide a nonempty event tensor (N, E)")

    if not events.is_floating_point() or not torch.all(torch.isfinite(events)):
        raise ValueError("Events must be finite floating values")

    for value, minimum in ((n_points, 2), (refinement_points, 2), (refinements, 0), (batch_size, 1)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError("Grid sizes, refinement count, and batch size must be valid integers")

    if not 0 < tail_probability < 0.5 or not 0 < credible_level < 1:
        raise ValueError("Require 0 < tail_probability < 0.5 and 0 < credible_level < 1")

    features = events if event_transform is None else event_transform(events)
    if (not isinstance(features, torch.Tensor) or features.ndim != 2
            or len(features) != len(events) or features.shape[1] == 0
            or features.device != events.device):
        raise ValueError("Event preprocessing must preserve rows and device")

    lower, upper = prior.support()
    bounds = [lower if np.isfinite(lower) else prior.ppf(tail_probability),
              upper if np.isfinite(upper) else prior.ppf(1 - tail_probability)]
    if not np.all(np.isfinite(bounds)) or bounds[0] >= bounds[1]:
        raise ValueError("The scalar prior must give finite, increasing grid bounds")

    def evaluate(points):
        """Evaluate new scalar grid points using cached event features

        Args:
            points (np.ndarray): Physical grid points (P,)

        Returns:
            np.ndarray: Unnormalized log posterior (P,)
        """

        parameters = torch.as_tensor(points[:, None], dtype=torch.float64, device=events.device)
        values = dataset_log_posterior(
            parameters, features, model,
            log_prior=lambda theta: prior.logpdf(theta[:, 0].cpu().numpy()),
            parameter_transform=parameter_transform, batch_size=int(batch_size),
            log_prob_kwargs=log_prob_kwargs,
        )

        return values.cpu().numpy()

    points = np.linspace(*bounds, n_points)
    log_posterior = evaluate(points)
    if not np.any(np.isfinite(log_posterior)):
        raise ValueError("No finite posterior mass on the grid")

    for _ in range(refinements):
        active = np.flatnonzero(log_posterior >= np.max(log_posterior) - 25.0)
        left, right = max(active[0] - 1, 0), min(active[-1] + 1, len(points) - 1)
        new_points = np.setdiff1d(np.linspace(points[left], points[right], refinement_points), points)
        if len(new_points) == 0:
            break

        new_values = evaluate(new_points)
        order = np.argsort(np.concatenate([points, new_points]))
        points = np.concatenate([points, new_points])[order]
        log_posterior = np.concatenate([log_posterior, new_values])[order]

    density, cdf = normalize_grid_posterior(points, log_posterior)

    for index, bound in [(0, lower), (-1, upper)]:
        if not np.isfinite(bound) and log_posterior[index] > np.max(log_posterior) - 20:
            raise ValueError("Extend the posterior grid into the prior tails")

    low, high = grid_quantiles(points, density, [(1 - credible_level) / 2, (1 + credible_level) / 2])
    if np.count_nonzero((points >= low) & (points <= high)) < 10:
        raise ValueError("The credible interval is unresolved: increase grid resolution")

    return points, log_posterior, density, cdf


# Joint posterior models conditioned on event sets
class EventSetDataset(Dataset):
    """Sample padded event subsets paired with shared parameters"""

    def __init__(self, parameters, events, min_events=1, max_events=None, seed=None):
        """Prepare padded subsets of pre-split event datasets

        Args:
            parameters (Tensor): Shared parameters (B, D)
            events (Tensor): Datasets (B, N, E)
            min_events (int): Minimum subset size
            max_events (int | None): Padded subset size K, default N
            seed (int | None): Fix each dataset subset when provided

        Returns:
            None
        """

        if (parameters.ndim != 2 or events.ndim != 3
                or len(parameters) != len(events) or len(events) == 0
                or parameters.shape[-1] == 0 or events.shape[-1] == 0):
            raise ValueError("Expected (datasets, param_dim) and (datasets, events, event_dim).")

        max_events = events.shape[1] if max_events is None else max_events
        if not 1 <= min_events <= max_events <= events.shape[1]:
            raise ValueError("Require 1 <= min_events <= max_events <= available events.")

        self.parameters, self.events = parameters, events
        self.min_events, self.max_events, self.seed = min_events, max_events, seed

    def __len__(self):
        """Count available datasets

        Args:
            None

        Returns:
            int: Dataset count B
        """

        return len(self.parameters)

    def __getitem__(self, index):
        """Draw one padded event subset without replacement

        Args:
            index (int): Dataset index

        Returns:
            tuple: Parameters (D,), (events (K, E), Boolean mask (K,)), True for padding
        """

        generator = None if self.seed is None else torch.Generator().manual_seed(self.seed + index)
        count = int(torch.randint(self.min_events, self.max_events + 1, (), generator=generator))
        order = torch.randperm(self.events.shape[1], generator=generator)[:self.max_events]
        events = self.events[index, order.to(self.events.device)]
        mask = torch.arange(self.max_events, device=events.device) >= count

        return self.parameters[index], (events.masked_fill(mask[:, None], 0), mask)


class SetPosterior(nn.Module):
    """Combine a set encoder, optional affine coordinates, and a joint density

    Fit optional moments first, then freeze encoder and moment_net during density training
    """


    def __init__(self, encoder, density_model, moment_net=None):
        """Condition a joint density model on an event-set encoder

        Args:
            encoder (nn.Module): Events (B, N, E) and mask (B, N) -> context (B, C)
            density_model (nn.Module): Conditional loss, log_prob, and sample methods
            moment_net (nn.Module | None): Context (B, C) -> location and log scale (B, 2D)

        Returns:
            None
        """

        super().__init__()
        self.encoder = encoder
        self.density_model = density_model
        self.moment_net = moment_net


    def moments(self, context):
        """Predict affine coordinates with log scales bounded to [-7, 5]

        Args:
            context (Tensor): Dataset summaries (B, C)

        Returns:
            tuple: Location and log scale (B, D), or scalar zeros without moment_net
        """

        if self.moment_net is None:
            zero = context.new_zeros(())
            return zero, zero

        values = self.moment_net(context)
        if values.ndim != 2 or len(values) != len(context) or values.shape[1] == 0 or values.shape[1] % 2:
            raise ValueError("moment_net must return (B, 2D)")

        location, log_scale = values.chunk(2, dim=-1)

        return location, log_scale.clamp(-7, 5)


    def standardize(self, parameters, context):
        """Map parameters into learned residual coordinates

        Args:
            parameters (Tensor): Parameter vectors (B, D)
            context (Tensor): Dataset summaries (B, C)

        Returns:
            tuple[Tensor, Tensor]: Residuals (B, D) and log forward Jacobians (B,)
        """

        location, log_scale = self.moments(context)
        if location.ndim and location.shape != parameters.shape:
            raise ValueError("moment_net parameter dimension does not match parameters")

        residuals = (parameters - location) * torch.exp(-log_scale)
        log_jacobian = -log_scale.expand_as(parameters).sum(-1)

        return residuals, log_jacobian


    def moment_loss(self, parameters, events, padding_mask=None, mean_only=False):
        """Fit affine coordinates before freezing the summary for diffusion training

        Args:
            parameters (Tensor): Shared parameters (B, D)
            events (Tensor | tuple): Sets (B, N, E) or (sets, padding_mask)
            padding_mask (Tensor | None): Boolean mask (B, N), True for padding
            mean_only (bool): Fit location by squared error before fitting scales

        Returns:
            Tensor: Squared errors or Gaussian NLLs up to a constant (B,)
        """

        if self.moment_net is None:
            raise ValueError("Moment fitting requires moment_net")

        if isinstance(events, (tuple, list)):
            events, padding_mask = events

        context = self.encoder(events, padding_mask)

        if mean_only:
            location, _ = self.moments(context)
            return (parameters - location).square().sum(-1)

        residuals, log_jacobian = self.standardize(parameters, context)

        return 0.5 * residuals.square().sum(-1) - log_jacobian


    @torch.no_grad()
    def sample_context(self, num_samples, cond, **kwargs):
        """Draw parameters using a previously encoded dataset context

        Args:
            num_samples (int): Draw count B
            cond (Tensor): Repeated contexts (B, C)
            **kwargs (dict): Density sampling options

        Returns:
            Tensor: Parameter draws (B, D)
        """

        location, log_scale = self.moments(cond)
        residuals = self.density_model.sample(num_samples, cond=cond, **kwargs)

        return location + log_scale.exp() * residuals


    def log_prob_context(self, parameters, cond, **kwargs):
        """Evaluate parameter densities with the affine Jacobian

        Args:
            parameters (Tensor): Parameter vectors (B, D)
            cond (Tensor): Dataset contexts (B, C)
            **kwargs (dict): Density evaluation options

        Returns:
            Tensor: Joint log densities (B,)
        """

        residuals, log_jacobian = self.standardize(parameters, cond)

        return self.density_model.log_prob(residuals, cond=cond, **kwargs) + log_jacobian


    def loss(self, parameters, events, padding_mask=None):
        # Compatible with aux.train_wrapper's (target, condition) batches
        """Evaluate the conditional training loss for each dataset

        Args:
            parameters (Tensor): Shared parameters (B, D)
            events (Tensor | tuple): Sets (B, N, E) or (sets, padding_mask)
            padding_mask (Tensor | None): Boolean mask (B, N), True for padding

        Returns:
            Tensor: Losses (B,)
        """

        if isinstance(events, (tuple, list)):
            events, padding_mask = events

        context = self.encoder(events, padding_mask)
        if parameters.ndim != 2 or len(parameters) != len(context):
            raise ValueError("Expected one parameter vector per event set.")

        residuals, _ = self.standardize(parameters, context)

        return self.density_model.loss(residuals, cond=context)


    def log_prob(self, parameters, events, padding_mask=None, **kwargs):
        """Evaluate joint parameter log densities conditioned on event sets

        Args:
            parameters (Tensor): Parameter vectors (B, D) or (B, S, D)
            events (Tensor): Preprocessed event sets (B, N, E)
            padding_mask (Tensor | None): Boolean (B, N), True for padding
            **kwargs (dict): Density-model options

        Returns:
            Tensor: Log densities (B,) or (B, S)
        """

        context = self.encoder(events, padding_mask)
        if parameters.ndim not in (2, 3) or len(parameters) != len(context):
            raise ValueError("Expected parameter shape (B, D) or (B, S, D).")

        shape = parameters.shape[:-1]
        if parameters.ndim == 3:
            context = context.repeat_interleave(parameters.shape[1], dim=0)

        values = self.log_prob_context(
            parameters.reshape(-1, parameters.shape[-1]), cond=context, **kwargs)

        return values.reshape(shape)


    @torch.no_grad()
    def sample(self, num_samples, events, padding_mask=None, batch_size=1024, **kwargs):
        """Draw joint parameters with one encoding per dataset, after eval()

        Args:
            num_samples (int): Draws S per dataset
            events (Tensor): Preprocessed event sets (B, N, E)
            padding_mask (Tensor | None): Boolean (B, N), True for padding
            batch_size (int): Maximum draws per density-model call
            **kwargs (dict): Density-model sampling options

        Returns:
            Tensor: Parameter draws (B, S, D), on the model device
        """

        if num_samples < 1 or batch_size < 1:
            raise ValueError("num_samples and batch_size must be positive.")

        context = self.encoder(events, padding_mask)
        chunks = []

        for start in range(0, len(context) * num_samples, batch_size):
            stop = min(start + batch_size, len(context) * num_samples)
            index = torch.arange(start, stop, device=context.device) // num_samples
            chunks.append(self.sample_context(stop - start, cond=context[index], **kwargs))

        samples = torch.cat(chunks)

        return samples.reshape(len(context), num_samples, samples.shape[-1])


@torch.no_grad()
def importance_sample_posterior(events, num_samples, model, log_prior, *,
                                event_transform=None, parameter_transform=None,
                                batch_size=8192, sample_batch_size=None,
                                sample_kwargs=None, log_prob_kwargs=None):
    """Batch proposals across event posteriors and apply mixture-corrected weights

    Args:
        events (Tensor): Observations (N, E)
        num_samples (int | array_like): Total M >= N, balanced across events, or counts (N,)
        model (object): sample(B, cond) -> (B, D), log_prob(theta, cond) -> (B,)
            SetPosterior uses singleton-event contexts. PyTorch models enter eval mode
        log_prior (Callable): Physical parameters (M, D) -> joint prior log density (M,)
        event_transform (Callable | None): Events (N, E) -> features (N, C)
        parameter_transform (nn.Module | None): reverse(z, return_logdet=True) returns
            physical parameters (M, D) and inverse log Jacobians (M,) or scalar
        batch_size (int): Maximum sample/event pairs per density call
        sample_batch_size (int | None): Maximum draws per call, default batch_size
        sample_kwargs (dict | None): Sampling options. Diffusion uses the ODE
        log_prob_kwargs (dict | None): Density options. Diffusion T, EPS, steps must match

    Returns:
        tuple[Tensor, Tensor]: Physical parameters (M, D) and normalized float64
            weights (M,), on events.device. Weights are zero outside prior support
    """

    from .sde import SDEModel

    if (not isinstance(events, torch.Tensor) or events.ndim != 2
            or not events.is_floating_point() or min(events.shape) == 0
            or not torch.all(torch.isfinite(events))):
        raise ValueError("events must be a finite, nonempty floating (N, event_dim) tensor.")

    if sample_batch_size is None:
        sample_batch_size = batch_size

    if any(isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 1
           for size in (batch_size, sample_batch_size)):
        raise ValueError("Batch sizes must be positive integers.")

    n_events = len(events)
    counts = torch.as_tensor(num_samples, device='cpu')
    if counts.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("Draw counts must be integers.")

    if counts.ndim == 0:
        total = int(counts)
        if total < n_events:
            raise ValueError("The total draw count must be >= the number of events.")

        counts = torch.full((n_events,), total // n_events, dtype=torch.long)
        counts[:total % n_events] += 1
    elif counts.shape != (n_events,) or torch.any(counts < 0) or counts.sum() <= 0:
        raise ValueError("Provide one nonnegative draw count per event, with positive total.")

    total = int(counts.sum())
    counts = counts.to(device=events.device, dtype=torch.long)

    if isinstance(model, nn.Module):
        model.eval()

    features = events if event_transform is None else event_transform(events)
    if (not isinstance(features, torch.Tensor) or features.ndim != 2
            or len(features) != n_events or features.shape[1] == 0
            or features.device != events.device or not torch.all(torch.isfinite(features))):
        raise ValueError("Event preprocessing must return finite (N, condition_dim) features on events.device.")

    density_model = model
    if isinstance(model, SetPosterior):
        features = torch.cat([model.encoder(features[start:start + sample_batch_size, None, :])
                              for start in range(0, n_events, sample_batch_size)])
        density_model = model.density_model

    if not callable(getattr(density_model, 'sample', None)) or not callable(getattr(density_model, 'log_prob', None)):
        raise TypeError("The model must provide both sample(num_samples, cond) and log_prob(parameters, cond).")

    sample = model.sample_context if isinstance(model, SetPosterior) else model.sample
    log_prob = model.log_prob_context if isinstance(model, SetPosterior) else model.log_prob

    sampling = dict(sample_kwargs or {})
    density = dict(log_prob_kwargs or {})
    if isinstance(density_model, SDEModel):
        if not sampling.get('use_ode', True):
            raise ValueError("Importance sampling requires ODE proposals matching SDEModel.log_prob.")

        sampling['use_ode'] = True

        for key in ('T', 'EPS', 'steps'):
            if key in sampling and key in density and sampling[key] != density[key]:
                raise ValueError(f"Use matching {key} in sampling and density evaluation.")

            if key in sampling:
                density.setdefault(key, sampling[key])
            elif key in density:
                sampling[key] = density[key]

    owners = torch.repeat_interleave(torch.arange(n_events, device=events.device), counts)
    chunks = []

    for start in range(0, total, sample_batch_size):
        condition = features[owners[start:start + sample_batch_size]]
        draws = sample(len(condition), cond=condition, **sampling)
        if (not isinstance(draws, torch.Tensor) or draws.ndim != 2 or len(draws) != len(condition)
                or draws.shape[1] == 0 or draws.device != events.device
                or not draws.is_floating_point()):
            raise ValueError("model.sample must return a floating (batch, parameter_dim) tensor on events.device.")

        chunks.append(draws)

    latent = torch.cat(chunks)
    del chunks, owners
    parameters, inverse_ldj = latent, latent.new_zeros(total)
    if parameter_transform is not None:
        parameters, inverse_ldj = parameter_transform.reverse(latent, return_logdet=True)

    if (not isinstance(parameters, torch.Tensor) or parameters.shape != latent.shape
            or parameters.device != events.device or not torch.all(torch.isfinite(latent))
            or not torch.all(torch.isfinite(parameters))):
        raise ValueError("Parameter samples must be finite and preserve shape (M, parameter_dim).")

    inverse_ldj = torch.as_tensor(inverse_ldj, device=events.device, dtype=torch.float64)
    if inverse_ldj.shape not in ((), (1,), (total,)):
        raise ValueError("The inverse-transform log determinant must be scalar or have shape (M,).")

    inverse_ldj = inverse_ldj.expand(total)
    prior = torch.as_tensor(log_prior(parameters), device=events.device, dtype=torch.float64)
    if prior.shape != (total,) or torch.any(torch.isnan(prior) | torch.isposinf(prior)):
        raise ValueError("log_prior must return (M,) joint log densities, finite or -inf.")

    supported = torch.isfinite(prior)
    if not torch.any(supported):
        raise ValueError("Importance weights have no finite mass within the prior support.")

    if not torch.all(torch.isfinite(inverse_ldj[supported])):
        raise ValueError("The parameter Jacobian must be finite on the prior support.")

    if n_events == 1:
        weights = supported.double()

        return parameters, weights / weights.sum()

    points = latent[supported]
    log_product = torch.zeros(len(points), device=events.device, dtype=torch.float64)
    log_mixture = torch.full_like(log_product, -torch.inf)
    log_fractions = (counts.double() / total).log()
    invalid_density = torch.zeros((), dtype=torch.bool, device=events.device)

    for start in range(0, len(points) * n_events, batch_size):
        stop = min(start + batch_size, len(points) * n_events)
        pair_ids = torch.arange(start, stop, device=events.device)
        point_ids, event_ids = pair_ids // n_events, pair_ids % n_events
        values = log_prob(points[point_ids], cond=features[event_ids], **density)
        if not isinstance(values, torch.Tensor) or values.shape != (stop - start,) or values.device != events.device:
            raise ValueError("model.log_prob must return (batch,) joint log densities on events.device.")

        values = values.double()
        invalid_density |= torch.any(torch.isnan(values) | torch.isposinf(values))
        first_point, last_point = start // n_events, (stop - 1) // n_events + 1
        local_ids = point_ids - first_point
        log_product[first_point:last_point].scatter_add_(0, local_ids, values)

        # Accumulate the proposal mixture with streamed log-sum-exp
        weighted = values + log_fractions[event_ids]
        maxima = values.new_full((last_point - first_point,), -torch.inf)
        maxima.scatter_reduce_(0, local_ids, weighted, reduce='amax', include_self=True)
        terms = torch.where(torch.isfinite(weighted), (weighted - maxima[local_ids]).exp(), 0.0)
        sums = torch.zeros_like(maxima).scatter_add_(0, local_ids, terms)
        batch_mixture = maxima + sums.log()
        log_mixture[first_point:last_point] = torch.logaddexp(
            log_mixture[first_point:last_point], batch_mixture)

    if invalid_density:
        raise ValueError("Event log densities must be finite or -inf on prior support.")

    if not torch.all(torch.isfinite(log_mixture)):
        raise ValueError("The proposal mixture must have finite density at its generated samples.")

    supported_weights = (log_product - log_mixture
                         - (n_events - 1) * (prior[supported] + inverse_ldj[supported]))
    if (torch.any(torch.isnan(supported_weights) | torch.isposinf(supported_weights))
            or not torch.any(torch.isfinite(supported_weights))):
        raise ValueError("Importance weights have no finite mass or contain NaN or +inf.")

    log_weights = torch.full((total,), -torch.inf, device=events.device, dtype=torch.float64)
    log_weights[supported] = supported_weights

    return parameters, torch.softmax(log_weights, dim=0)
