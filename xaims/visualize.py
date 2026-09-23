# Visualization functions
#
# m.mieskolainen@imperial.ac.uk, 2025

from itertools import cycle
from time import time

import numpy as np
import numba
from tqdm import tqdm
from scipy.stats import gaussian_kde

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Arc
from matplotlib.colors import ListedColormap
from matplotlib import gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from . import qedgen

from .statistics import (chi2_func, histogram_contributions, covariance_chi2,
                         compute_ratio_uncertainty)

def plot_inversion_comparisons(samples: dict,
                               reference_key: str,
                               extract_observables,
                               nbins: int = 60,
                               figsize: tuple = (8, 6),
                               alpha: float = 0.3):
    """Plot histograms and ratios with parent-event bootstrap bands and discrepancy scores

    Args:
        samples (dict): Named x (N, D), optional w (N,) and shared event_ids (N,)
        reference_key (str): Reference sample name
        extract_observables (Callable): x -> (observable arrays (N,), labels, units)
        nbins (int): Number of histogram edge points
        figsize (tuple[float, float]): Figure size in inches
        alpha (float): Band opacity

    Returns:
        None: Displays one figure per observable, bands condition on trained models
    """

    # 1) Extract observables and weights as numpy
    obs = {}
    weights = {}
    event_ids = {}
    labels = units = None

    for name, sample in samples.items():
        x = sample['x']
        if hasattr(x, 'cpu'): x = x.cpu().numpy()

        obs[name], labels, units = extract_observables(x)
        w = sample.get('w', None)
        if w is None:
            weights[name] = None
        else:
            if hasattr(w, 'cpu'): w = w.cpu().numpy()

            weights[name] = np.asarray(w).reshape(-1)

        ids = sample.get('event_ids')
        if hasattr(ids, 'cpu'): ids = ids.cpu().numpy()

        event_ids[name] = None if ids is None else np.asarray(ids)
    
    # Shared IDs preserve correlations between truth, detector and unfolded samples
    shared_rows = max((int(ids.max()) + 1 for ids in event_ids.values()
                       if ids is not None and len(ids)), default=0)

    # 2) Compute histograms, variances, and chi2 per observable
    for k in range(len(labels)):
        
        # compute bin edges from reference distribution
        data_flat = obs[reference_key][k]
        bins = np.linspace(np.percentile(data_flat, 0.1),
                           np.percentile(data_flat, 99.9),
                           nbins)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        hist, var, contributions, covariances = {}, {}, {}, {}

        for name in samples:
            ids = event_ids[name]
            A = histogram_contributions(obs[name][k], bins, weights[name], ids,
                                        n_events=shared_rows if ids is not None else None)
            contributions[name] = A
            hist[name] = np.asarray(A.sum(axis=0)).ravel()
            covariances[name] = (A.T @ A).toarray()
            var[name] = np.diag(covariances[name])

        chi2, cross_diagonal = {}, {}

        for name in samples:
            if name == reference_key:
                continue

            paired = event_ids[name] is not None and event_ids[reference_key] is not None
            if paired:
                cross = (contributions[name].T @ contributions[reference_key]).toarray()
            else:
                cross = np.zeros_like(covariances[name])

            cross_diagonal[name] = np.diag(cross)
            difference_cov = covariances[name] + covariances[reference_key] - cross - cross.T
            chi2[name] = covariance_chi2(hist[name] - hist[reference_key], difference_cov)

        # 3) Plotting
        # assign colors
        prop_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        color_cycle = cycle(prop_cycle)
        color_map = {
            name: 'black' if name == reference_key else next(color_cycle)
            for name in samples
        }

        fig = plt.figure(figsize=figsize)
        gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0)
        ax_main = fig.add_subplot(gs[0])
        ax_ratio= fig.add_subplot(gs[1], sharex=ax_main)
        ax_main.tick_params(labelbottom=False)
        
        # Main histograms with chi2 in legend
        for name in samples:
            label = name
            if name != reference_key:
                label = rf"{name} ($\tilde{{\chi}}^2/\text{{rank}}$ = {chi2[name]:.1f})"
            
            ax_main.hist(obs[name][k], bins=bins, weights=weights[name],
                         histtype='step', color=color_map[name], label=label)

        ax_main.set_ylabel("Counts")
        ax_main.set_title(f"{labels[k]} ({units[k]})")
        ax_main.legend(fontsize=10)
        #ax_main.grid(True)
        ax_main.set_xlim(bins[0], bins[-1])
        
        # 4) Ratio plots vs reference
        for name in samples:
            if name == reference_key:
                continue

            ratio, sigma = compute_ratio_uncertainty(
                hist[name], var[name], hist[reference_key], var[reference_key],
                covariance=cross_diagonal[name]
            )
            ratio_ext = np.concatenate([ratio, ratio[-1:]])
            sigma_ext = np.concatenate([sigma, sigma[-1:]])
            ax_ratio.step(bins, ratio_ext, where='post',
                          color=color_map[name], label=f"{name}/{reference_key}")
            ax_ratio.fill_between(bins,
                                  ratio_ext - sigma_ext,
                                  ratio_ext + sigma_ext,
                                  step='post', color=color_map[name], alpha=alpha)

        ax_ratio.hlines(1.0, bins[0], bins[-1], color='gray', linestyle='--')
        ax_ratio.set_ylabel("Ratio")
        ax_ratio.set_xlabel(f"{labels[k]} ({units[k]})")
        ax_ratio.grid(True)
        ax_ratio.set_xlim(bins[0], bins[-1])
        ax_ratio.set_ylim(0, 2)
        
        plt.tight_layout()
        plt.show()


def plot_inversion_relative_error(samples: dict,
                                      extract_observables,
                                      reference_key: str = "QED (gen)",
                                      nbins: int = 60,
                                      figsize: tuple = (8, 6),
                                      alpha: float = 0.3):
    """Plot paired relative errors with within-bin standard deviations

    Args:
        samples (dict): Named aligned x (N, D) and optional w (N,)
        extract_observables (Callable): x -> (observable arrays (N,), labels, units)
        reference_key (str): Reference sample name
        nbins (int): Number of histogram edge points
        figsize (tuple[float, float]): Figure size in inches
        alpha (float): Band opacity

    Returns:
        None: Displays one figure per observable
    """

    # 1) Extract observables and weights
    obs = {}
    weights = {}
    labels = units = None

    for name, sample in samples.items():
        x = sample['x']
        if hasattr(x, 'cpu'): x = x.cpu().numpy()

        obs[name], labels, units = extract_observables(x)
        w = sample.get('w', None)
        if w is not None:
            if hasattr(w, 'cpu'): w = w.cpu().numpy()

            w = np.squeeze(w)

        weights[name] = w

    reference_obs = obs[reference_key]

    # 2) For each observable
    for k in range(len(labels)):
        x_ref = reference_obs[k]
        bins = np.linspace(np.percentile(x_ref, 0.1),
                           np.percentile(x_ref, 99.9), nbins)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

        # 3) Plotting
        plt.figure(figsize=figsize)

        for name in samples:
            if name == reference_key:
                continue

            x_est = obs[name][k]
            rel_error = (x_ref - x_est) / x_ref
            w = weights[name]

            bin_indices = np.digitize(x_ref, bins) - 1
            bin_mean = []
            bin_std = []

            for i in range(len(bins) - 1):
                mask = (bin_indices == i)
                if np.any(mask):
                    vals = rel_error[mask]
                    w_bin = None if w is None else w[mask]

                    if w_bin is None:
                        mean = np.mean(vals)
                        std = np.std(vals)
                    else:
                        w_sum = np.sum(w_bin)
                        mean = np.sum(w_bin * vals) / w_sum
                        var = np.sum(w_bin * (vals - mean)**2) / w_sum
                        std = np.sqrt(var)

                    bin_mean.append(mean)
                    bin_std.append(std)
                else:
                    bin_mean.append(np.nan)
                    bin_std.append(np.nan)

            bin_mean = np.array(bin_mean)
            bin_std = np.array(bin_std)
            color = next(color_cycle)

            plt.plot(bin_centers, bin_mean, label=f"{name}", color=color)
            plt.fill_between(bin_centers,
                             bin_mean - bin_std,
                             bin_mean + bin_std,
                             color=color, alpha=alpha)
        
        plt.axhline(0.0, linestyle='--', color='gray')
        plt.xlabel(f"{labels[k]} ({units[k]})")
        plt.ylabel(r"$(x_\mathrm{ref} - x_\mathrm{est}) / x_\mathrm{ref}$")
        plt.grid(True)
        plt.ylim([-0.2, 0.2])
        plt.legend()
        plt.tight_layout()
        plt.show()


        
def plot_losses(train_losses, val_losses=None, 
                          title="Training and Validation Loss", 
                          xlabel="Epoch", ylabel="Loss",
                          figsize=(12, 5)):
    """Plot training and validation losses on linear and logarithmic axes

    Args:
        train_losses (array_like): Training losses (E,)
        val_losses (array_like | None): Validation losses (E,)
        title (str): Figure title
        xlabel (str): Horizontal label
        ylabel (str): Vertical label
        figsize (tuple[float, float]): Figure size in inches

    Returns:
        tuple: Displayed Figure and (linear Axes, logarithmic Axes)
    """

    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=figsize)

    train_losses = np.array(train_losses)
    if val_losses is not None:
        val_losses = np.array(val_losses)
    
    # Linear-linear plot
    ax_lin.plot(train_losses, color='black', label='Train', zorder=10)
    if val_losses is not None:
        ax_lin.plot(val_losses, color='red', label='Validation')

    ax_lin.set_title(f"(Linear Scale)")
    ax_lin.set_xlabel(xlabel)
    ax_lin.set_ylabel(ylabel)
    ax_lin.legend()
    ax_lin.grid(True)
    ax_lin.set_xlim([0,len(train_losses)])
    
    # Log-log plot (take into account if we have negative values)
    min_value = train_losses.min() if (val_losses is None) else min(train_losses.min(), val_losses.min())
    if min_value > 0:
        min_value = 0
    else:
        min_value = np.abs(min_value) + 0.1 # Shift
    
    ax_log.plot(train_losses + min_value, color='black', label='Train', zorder=10)
    
    if val_losses is not None:
        ax_log.plot(val_losses + min_value, color='red', label='Validation')
    
    ax_log.set_xscale('log')
    ax_log.set_yscale('log')
    ax_log.set_title(f"(Log-Log Scale)")
    ax_log.set_xlabel(xlabel)
    ax_log.set_ylabel(f"shift +{min_value:0.1f}" if min_value != 0 else "")
    ax_log.legend()
    ax_log.grid(True, which='both')
    ax_lin.set_xlim([0,len(train_losses)])
    
    plt.tight_layout()
    plt.show()
    
    return fig, (ax_lin, ax_log)

def analyze(
    x_samples: dict[str, np.ndarray],
    s: float,
    t_min: float,
    t_max: float,
    xs_tot: float,
    nbins: int = 150,
    reference_key: str="QED (MC)",
    prc: list = [1, 99]
):
    """Compare generated momenta with analytic Bhabha cross sections

    Args:
        x_samples (dict[str, np.ndarray]): Named momentum samples (N, 3)
        s (float): Squared CM energy in GeV^2
        t_min (float): Lower momentum-transfer bound in GeV^2
        t_max (float): Upper momentum-transfer bound in GeV^2
        xs_tot (float): Total cross section for histogram normalization
        nbins (int): Histogram bin count
        reference_key (str): Reference sample name
        prc (list[float]): Lower and upper plotting percentiles

    Returns:
        None: Displays cross-section and observable comparison figures
    """
    
    # Compute t and costheta for every sample
    t_dict = {}
    cosTheta_dict = {}

    for label, x in x_samples.items():
        t = qedgen.pz_to_t(pz=x[:,2], s=s)
        t_dict[label]        = t
        cosTheta_dict[label] = qedgen.t_to_costheta(t=t, s=s)
    
    # Analytic curves
    t_point = np.linspace(t_min, t_max, 1000)
    cosTheta_point            = qedgen.t_to_costheta(t=t_point, s=s)
    dsigma_dt_analytic        = qedgen.dsigma_dt(s=s,  t=t_point, theory=qedgen.Mode.QED_bhabha) * qedgen.nat2ub
    dsigma_dcosTheta_analytic = qedgen.dsigma_dcostheta(s=s, costheta=cosTheta_point, theory=qedgen.Mode.QED_bhabha) * qedgen.nat2ub
    
    # Title
    theta_min = np.rad2deg(np.arccos(qedgen.t_to_costheta(t=t_max, s=s)))
    theta_max = np.rad2deg(np.arccos(qedgen.t_to_costheta(t=t_min, s=s)))
    title = (
        f"$\\sqrt{{s}}$ = {np.sqrt(s):0.2f} GeV | "
        f"$\\sigma_{{tot}}^{{fid}} \\approx$ {qedgen.nat2ub * xs_tot:0.2f} $\\mu$b | "
        f"$\\Theta$ = ({theta_min:0.1f}, {theta_max:0.1f}) deg"
    )
    
    # Plot dsigma/d(-t)
    fig1,ax1 = plot_1D_xs(
        xs_tot=xs_tot * qedgen.nat2ub,
        x_vals_dict={lbl: np.abs(t_dict[lbl]) for lbl in t_dict},
        reference_key=reference_key,
        analytic_x=np.abs(t_point),
        analytic_y=dsigma_dt_analytic,
        xlabel="Mandelstam $-t$ (GeV$^2$)",
        ylabel="$d\\sigma/d(-t)$ (μb / GeV$^2$)",
        title=title,
        nbins=nbins,
        xlim=[0.0, np.abs(t_min)],
        yscale='log'
    )
    
    # Plot dsigma/dcostheta
    fig2,ax2 = plot_1D_xs(
        xs_tot=xs_tot * qedgen.nat2ub,
        x_vals_dict={lbl: cosTheta_dict[lbl] for lbl in cosTheta_dict},
        reference_key=reference_key,
        analytic_x=cosTheta_point,
        analytic_y=dsigma_dcosTheta_analytic,
        xlabel="scattering angle $\\cos\\Theta$",
        ylabel="$d\\sigma/d\\cos\\Theta$ (μb)",
        title=title,
        nbins=nbins,
        xlim=[-1.0, 1.0],
        yscale='log'
    )
    
    # 2D observables
    obs_dict = {}

    for lbl, x in x_samples.items():
        obs_dict[lbl], labels, units = qedgen.extract_observables(x)
    
    fig3,ax3 = plot_matrix_1D_2D_xs(
        xs_tot=xs_tot,
        x_vals_dict={lbl: obs_dict[lbl] for lbl in obs_dict},
        labels=labels,
        units=units,
        var_indices=[0,1,2],
        nbins=nbins,
        reference_key=reference_key,
        title=f"{title} | cartesian $(p_x,p_y,p_z)$",
        prc=prc
    )
    
    fig4,ax4 = plot_matrix_1D_2D_xs(
        xs_tot=xs_tot,
        x_vals_dict={lbl: obs_dict[lbl] for lbl in obs_dict},
        labels=labels,
        units=units,
        var_indices=[3,4,5],
        nbins=nbins,
        reference_key=reference_key,
        title=f"{title} | collider $(p_T,\\eta,\\phi)$",
        prc=prc
    )

# Example: get N distinct colors from a colormap
def get_distinct_colors(n, cmap_name='tab10'):
    """Select colors evenly across a color map

    Args:
        n (int): Color count
        cmap_name (str): Matplotlib color-map name

    Returns:
        list[tuple]: n RGBA colors
    """

    cmap = plt.get_cmap(cmap_name)

    return [cmap(i) for i in range(n)]

def plot_1d_hist(ax, values, bins, label, color, norm_factor, lw=1.0):
    """Draw a scaled step histogram on an existing axis

    Args:
        ax (Axes): Target axis
        values (array_like): Values (N,)
        bins (int | array_like): Bin count or edges (K + 1,)
        label (str): Legend label
        color (color_like): Line color
        norm_factor (float): Count multiplier
        lw (float): Line width

    Returns:
        None
    """

    h, edges = np.histogram(values, bins=bins)
    h = norm_factor * h
    ax.step(edges, np.append(h, h[-1]), where='post', label=label, color=color, lw=lw)

def plot_2d_contour_kde(ax, x, y, label, color, lw=1.0):
    """Draw Gaussian-KDE contours on an existing axis

    Args:
        ax (Axes): Target axis
        x (array_like): First coordinates (N,)
        y (array_like): Second coordinates (N,)
        label (str): Reserved label, currently unused
        color (color_like): Contour color
        lw (float): Line width

    Returns:
        None
    """

    xy = np.vstack([x, y])
    kde = gaussian_kde(xy)
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 100), np.linspace(ymin, ymax, 100))
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    ax.contour(xx, yy, zz, levels=4, colors=[color], linewidths=lw, linestyles='solid')

def plot_1D_xs(
    xs_tot,
    x_vals_dict,
    analytic_x,
    analytic_y,
    xlabel, ylabel, title,
    reference_key: str = "QED (MC)",
    nbins: int=60,
    xlim=None,
    yscale='log',
    figsize = (8,5)
):
    """Compare sampled differential cross sections with an analytic curve

    Args:
        xs_tot (float): Total cross section for histogram normalization
        x_vals_dict (dict[str, array_like]): Named one-dimensional samples (N,)
        analytic_x (array_like): Analytic curve coordinates (G,)
        analytic_y (array_like): Analytic differential cross section (G,)
        xlabel (str): Horizontal label
        ylabel (str): Vertical label
        title (str): Plot title
        reference_key (str): Reference sample name
        nbins (int): Histogram bin count
        xlim (tuple[float, float] | None): Horizontal limits
        yscale (str): Vertical axis scale
        figsize (tuple[float, float]): Figure size in inches

    Returns:
        tuple[Figure, Axes]: Cross-section plot
    """

    fig,ax = plt.subplots(figsize=figsize)
    
    # Plot analytic
    plt.plot(analytic_x, analytic_y, 'k-', label='QED (analytic)')
    
    # Bin setup
    x_bins = np.linspace(analytic_x.min(), analytic_x.max(), nbins + 1)
    bin_widths = np.diff(x_bins)
    
    colors = get_distinct_colors(len(x_vals_dict), cmap_name='tab10')

    if reference_key in x_vals_dict.keys():
        h_mc, _ = np.histogram(x_vals_dict[reference_key], bins=x_bins)
    else:
        print('plot_1D_xs: No reference histogram given for chi2')
    
    k = 0

    for label, x_vals in x_vals_dict.items():
        
        # Histogram and chi2
        h, _ = np.histogram(x_vals, bins=x_bins)

        if reference_key in x_vals_dict.keys():
            chi2 = chi2_func(h, h_mc) if label != reference_key else None
        else:
            chi2 = -1
        
        # ----------------------------------------------------------
        # Differential normalization
        num_events = len(x_vals)
        norm = 1.0 / (num_events * bin_widths + 1e-15) * xs_tot
        # ----------------------------------------------------------
        
        h = norm * h
        
        label_str = label
        if label != reference_key:
            label_str = rf'{label} ($\tilde{{\chi}}^2/\text{{bin}}$ = {chi2:.1f})'
            zorder = 100-k
        else:
            zorder = 100
        
        plt.step(x_bins, np.append(h, h[-1]), where='post', label=label_str, color=colors[k], zorder=zorder)
        k += 1
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if xlim:
        plt.xlim(xlim)

    plt.yscale(yscale)
    plt.legend()
    plt.grid(True, which='both', linestyle=':', linewidth=0.5)
    plt.show()

    return fig, ax

def plot_matrix_1D_2D_xs(xs_tot, x_vals_dict, labels, units, var_indices, reference_key: str="QED (MC)", nbins: int=50, title: str=None, prc=[1, 99]):
    """Plot marginal cross sections and pairwise scatter plots

    Args:
        xs_tot (float): Total cross section for histogram normalization
        x_vals_dict (dict[str, list]): Named observable arrays, F arrays (N,) per sample
        labels (list[str]): Observable labels (F,)
        units (list[str]): Observable units (F,)
        var_indices (list[int]): Selected observable indices (V,)
        reference_key (str): Reference sample name
        nbins (int): Histogram bin count
        title (str | None): Figure title
        prc (list[float]): Lower and upper plotting percentiles

    Returns:
        tuple[Figure, np.ndarray]: Figure and axes (V, V)
    """
    
    n_vars = len(var_indices)
    fig, axes = plt.subplots(n_vars, n_vars, figsize=(5 * n_vars, 5 * n_vars))
    fig.suptitle(title or '', fontsize=16)
    
    # Other models
    model_names = [k for k in x_vals_dict if k != reference_key]
    
    colors = get_distinct_colors(len(x_vals_dict), cmap_name='tab10')

    # Compute percentiles across all models for each variable
    percentiles = {}

    for idx in var_indices:
        all_vals = np.concatenate([x_vals_dict[name][idx] for name in x_vals_dict])
        p_min, p_max = np.percentile(all_vals, prc)
        percentiles[idx] = (p_min, p_max)
    
    for i in range(n_vars):
        for j in range(n_vars):
            ax = axes[i, j]
            idx_i = var_indices[i]
            idx_j = var_indices[j]

            if i == j:
                
                # Diagonal: 1D histograms
                model_vals = {name: x_vals_dict[name][idx_i] for name in model_names}
                
                # Binning using percentiles
                x_min, x_max = percentiles[idx_i]
                x_bins = np.linspace(x_min, x_max, nbins + 1)
                bin_widths = np.diff(x_bins)
                
                k = 0
                # Reference histogram
                if reference_key in x_vals_dict.keys():
                    ref_vals = x_vals_dict[reference_key][idx_i]
                    
                    # ----------------------------------------------------------
                    # Differential normalization
                    num_events = len(ref_vals)
                    norm = 1.0 / (num_events * bin_widths + 1e-15) * xs_tot
                    # ----------------------------------------------------------
                    
                    h_ref_counts, _ = np.histogram(ref_vals, bins=x_bins)
                    h_ref = norm * h_ref_counts
                    ax.step(x_bins, np.append(h_ref, h_ref[-1]), where='post',
                            label=reference_key, color=colors[k], lw=1.5, zorder=100)
                    k += 1
                else:
                    print('plot_matrix_1D_2D_xs: No reference histogram given for chi2')
                
                # Other models
                for name in model_names:
                    
                    # ----------------------------------------------------------
                    # Differential normalization
                    num_events = len(model_vals[name])
                    norm = 1.0 / (num_events * bin_widths + 1e-15) * xs_tot
                    # ----------------------------------------------------------
                    
                    h_model_counts, _ = np.histogram(model_vals[name], bins=x_bins)
                    
                    if reference_key in x_vals_dict.keys():
                        chi2 = chi2_func(h_model_counts, h_ref_counts) # compare counts
                    else:
                        chi2 = -1
                    
                    h_model = norm * h_model_counts
                    ax.step(x_bins, np.append(h_model, h_model[-1]), where='post',
                            label=rf'{name} ($\tilde{{\chi}}^2/\text{{bin}}$ = {chi2:.1f})', color=colors[k], lw=1.0, zorder=100-k)
                    k += 1
                
                ax.set_xlabel(f"{labels[idx_i]} ({units[idx_i]})")
                ax.set_ylabel(f"$d\\sigma/d${labels[idx_i]} ($\\mu$b / {units[idx_i]})")
                #ax.grid(True)
                ax.set_ylim(0, None)
                ax.set_xlim(x_min, x_max)
                ax.legend(fontsize=8)
                
            elif i > j:
                # Lower triangle: 2D
                k = 0

                for name in x_vals_dict.keys():
                    
                    obs   = x_vals_dict[name]
                    label = name if (i == n_vars - 1 and j == 0) else None  # legend only in bottom-left
                    
                    #plot_2d_contour(ax, obs[idx_j], obs[idx_i], label=label, color=colors[k],
                    #    lw=1.5 if name == reference_key else 1.0, bins=bins)
                    ax.scatter(obs[idx_j], obs[idx_i], s=3, color=colors[k], alpha=0.2, edgecolors='none', zorder=100-k)
                    
                    k += 1
                
                ax.set_xlabel(f"{labels[idx_j]}")
                ax.set_ylabel(f"{labels[idx_i]}")
                
                # Set axis limits based on percentiles
                xlim = percentiles[idx_j]
                ylim = percentiles[idx_i]
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)

            else:
                ax.axis('off')  # Upper triangle is left blank

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    return fig,axes


def plot_gaussian_posterior_comparison(samples, mean, covariance, truth=None,
                                       probabilities=(0.68, 0.95), grid_size=160,
                                       importance_samples=None, importance_weights=None):
    """Compare joint Gaussian credible contours with unweighted and weighted sample KDEs

    Args:
        samples (array_like): Posterior draws (S, 2)
        mean (array_like): Analytic posterior mean (2,)
        covariance (array_like): Analytic posterior covariance (2, 2)
        truth (array_like | None): True parameters (2,)
        probabilities (sequence[float]): Joint credible masses (L,)
        grid_size (int): KDE grid points per axis
        importance_samples (array_like | None): Importance proposals (P, 2)
        importance_weights (array_like | None): Nonnegative proposal weights (P,)

    Returns:
        tuple[Figure, Axes]: Joint contours, weighted scatter if KDE is underdetermined
    """

    from scipy.stats import chi2
    from matplotlib.lines import Line2D

    samples = np.asarray(samples, dtype=float)
    mean, covariance = np.asarray(mean, dtype=float), np.asarray(covariance, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    if (samples.ndim != 2 or samples.shape[1] != 2 or len(samples) < 3
            or mean.shape != (2,) or covariance.shape != (2, 2)
            or not np.all(np.isfinite(samples)) or not np.all(np.isfinite(mean))
            or not np.all(np.isfinite(covariance))):
        raise ValueError("Expected finite (samples >= 3, 2), (2,), and (2, 2) inputs.")

    if (probabilities.ndim != 1 or probabilities.size == 0
            or not np.all(np.isfinite(probabilities))
            or np.any((probabilities <= 0) | (probabilities >= 1)) or grid_size < 16):
        raise ValueError("Contour probabilities must lie in (0, 1). grid_size must be >= 16.")

    datasets = [(samples, None, 'C0', 'Dataset diffusion (KDE)')]
    if (importance_samples is None) != (importance_weights is None):
        raise ValueError("Provide importance samples and weights together.")

    if importance_samples is not None:
        points = np.asarray(importance_samples, dtype=float)
        weights = np.asarray(importance_weights, dtype=float)
        if (points.ndim != 2 or points.shape[1] != 2 or len(points) == 0
                or weights.shape != (len(points),) or not np.all(np.isfinite(points))
                or not np.all(np.isfinite(weights)) or np.any(weights < 0)
                or not 0 < weights.sum() < np.inf):
            raise ValueError("Expected finite (samples, 2) points and nonnegative weights with positive mass.")

        datasets.append((points, weights / weights.sum(), 'C4', 'Importance weighted (KDE)'))

    if truth is not None:
        truth = np.asarray(truth, dtype=float)
        if truth.shape != (2,) or not np.all(np.isfinite(truth)):
            raise ValueError("truth must be a finite two-coordinate vector.")

    root = np.linalg.cholesky(covariance)
    radius = np.sqrt(chi2.ppf(probabilities.max(), df=2))
    extent = 1.2 * radius * np.sqrt(np.diag(covariance))
    fig, ax = plt.subplots(figsize=(6, 5))
    handles = []

    for points, weights, color, label in datasets:
        kde = None
        if weights is None or 1 / np.sum(weights**2) > 3:
            try:
                kde = gaussian_kde(points.T, weights=weights)
            except np.linalg.LinAlgError:
                if weights is None:
                    raise

        if kde is None:
            order = np.argsort(weights)[::-1]
            active = order[:np.searchsorted(np.cumsum(weights[order]), 0.999) + 1]
            handles.append(ax.scatter(
                *points[active].T, s=200 * weights[active] / weights.max(), color=color,
                alpha=0.6, label='Importance weighted samples (KDE unavailable)'))
            continue

        margin = 3 * np.sqrt(np.diag(kde.covariance))
        lower, upper = points.min(axis=0), points.max(axis=0)
        if weights is not None:
            # Near-zero-weight proposal tails must not set the KDE resolution
            for j in range(2):
                order = np.argsort(points[:, j])
                indices = np.searchsorted(np.cumsum(weights[order]), [0.0001, 0.9999])
                lower[j], upper[j] = points[order[indices.clip(max=len(order) - 1)], j]

        lower, upper = np.minimum(lower - margin, mean - extent), np.maximum(upper + margin, mean + extent)
        if truth is not None:
            lower, upper = np.minimum(lower, truth), np.maximum(upper, truth)

        xx, yy = np.meshgrid(np.linspace(lower[0], upper[0], grid_size),
                             np.linspace(lower[1], upper[1], grid_size))
        density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
        ordered = np.sort(density.ravel())[::-1]
        cumulative = np.cumsum(ordered)
        cumulative /= cumulative[-1]  # constant cell area cancels
        levels = ordered[np.searchsorted(cumulative, probabilities).clip(max=len(ordered) - 1)]
        contours = ax.contour(xx, yy, density, levels=np.unique(levels), colors=color)
        ax.clabel(contours, fmt={level: f'{p:.0%}' for level, p in zip(levels, probabilities)}, fontsize=8)
        handles.append(Line2D([], [], color=color, label=label))

    angle = np.linspace(0, 2 * np.pi, 361)
    circle = np.column_stack([np.cos(angle), np.sin(angle)])

    # Parametric ellipses remain resolved even when learned samples are broad
    for probability in probabilities:
        ellipse = mean + np.sqrt(chi2.ppf(probability, df=2)) * (circle @ root.T)
        ax.plot(ellipse[:, 0], ellipse[:, 1], color='C1', linestyle='--')

    handles.append(Line2D([], [], color='C1', linestyle='--', label='Analytic Gaussian'))
    if truth is not None:
        handles.append(ax.scatter(*truth, marker='*', s=90, color='black', label='Simulation truth', zorder=5))

    ax.set(xlabel=r'$\theta_1$', ylabel=r'$\theta_2$',
           title='Posterior contours: ' + ', '.join(f'{p:.0%}' for p in probabilities))
    ax.set_aspect('equal', adjustable='box')
    ax.legend(handles=handles, fontsize=9)
    fig.tight_layout()

    return fig, ax
