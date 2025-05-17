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

@numba.njit
def chi2_func(h: np.ndarray, h_ref: np.ndarray, type: str="symmetric"):
    """ Chi2 comparison metric """
    ind = (h > 0) & (h_ref > 0)
    if np.sum(ind) == 0:
        return -1
    else:
        if   type == "symmetric":
            return np.mean((h[ind] - h_ref[ind]) ** 2 / np.sqrt(h[ind]**2 + h_ref[ind]**2)) 
        elif type == "pearson":
            return np.mean((h[ind] - h_ref[ind]) ** 2 / (h_ref[ind]))
        elif type == "neuman":
            return np.mean((h[ind] - h_ref[ind]) ** 2 / (h[ind]))

def plot_inversion_comparisons(samples: dict,
                               reference_key: str,
                               extract_observables,
                               nbins: int = 60,
                               figsize: tuple = (8, 6),
                               alpha: float = 0.3):
    """
    Compare multiple samples against a reference via histograms, ratio plots,
    and chi2 calculations.
    
    Args:
        samples: dict of name -> {'x': array or tensor (N,D), 'w': array or tensor (N,) or None}
        reference_key: key in `samples` to use as reference for ratios
        extract_observables: function to map x -> (obs_list, labels, units)
        compute_ratio_uncertainty: function(hist_num, var_num, hist_den, var_den) -> (ratio, sigma)
        nbins: number of bins for histograms
        figsize: figure size tuple
        alpha: fill transparency for uncertainty bands
    """
    # 1) Extract observables and weights as numpy
    obs = {}
    weights = {}
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
            weights[name] = w.squeeze()
    
    # 2) Compute histograms, variances, and chi2 per observable
    for k in range(len(labels)):
        
        # compute bin edges from reference distribution
        data_flat = obs[reference_key][k]
        bins = np.linspace(np.percentile(data_flat, 0.1),
                           np.percentile(data_flat, 99.9),
                           nbins)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        # compute histograms and variances per sample
        hist = {}
        var = {}
        for name in samples:
            arr, w_arr = obs[name][k], weights[name]
            hist[name], _ = np.histogram(arr, bins=bins, weights=w_arr)
            if w_arr is None:
                var[name] = hist[name]
            else:
                hist_sq, _ = np.histogram(arr, bins=bins, weights=w_arr**2)
                var[name] = hist_sq

        # compute chi2 vs reference
        chi2 = {}
        
        for name in samples:
            if name == reference_key:
                continue
            chi2[name] = chi2_func(hist[name], hist[reference_key])
        
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
                label = f"{name} ($\\tilde{{\chi}}^2 = {chi2[name]:.1f}$)"
            
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
                hist[name], var[name], hist[reference_key], var[reference_key]
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
    """
    Visualizes (reference(x) - estimate(x)) / reference(x) mean ± std
    as a function of observable x for each sample vs reference.
    
    Args:
        samples: dict of name -> {'x': array or tensor (N,D), 'w': array or tensor (N,) or None}
        extract_observables: function x -> (obs_list, labels, units)
        reference_key: key to use as the reference
        nbins: number of bins for observable axis
        figsize: matplotlib figure size
        alpha: transparency of error band
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
    """
    Plot training and optional validation losses in two subplots:
      - Left: linear-linear scale
      - Right: log-log scale
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

def compute_ratio_uncertainty(hist_num, var_num, hist_den, var_den):
    """
    Ratio uncertainty via Taylor expansion
    """
    ratio = np.zeros_like(hist_den, dtype=float)
    sigma_ratio = np.zeros_like(hist_den, dtype=float)
    
    # Denominator mask
    mask_den = hist_den > 0
    
    # Compute ratio where denominator > 0
    ratio[mask_den] = hist_num[mask_den] / hist_den[mask_den]
    
    # Numerator mask
    mask_num = hist_num > 0
    
    # Valid bins for uncertainty: both numerator and denominator > 0
    valid = mask_den & mask_num
    
    # Calculate uncertainty only on valid bins
    sigma_ratio[valid] = (
        ratio[valid]
        * np.sqrt(
            var_num[valid] / (hist_num[valid] ** 2)
            + var_den[valid] / (hist_den[valid] ** 2)
        )
    )
    return ratio, sigma_ratio

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
    """
    Cross section analysis and comparison
    
    x_samples:     dict[label -> (N_events, 3) array of (px,py,pz)]
                   must include key "QED (MC)" for the ground truth MC
    s:             Mandestam s
    t_min,t_max:   Mandelstam t fiducial limits
    xs_tot:        Total fiducial cross section from the MC generator (in natural units)
    nbins:         Number of histogram bins
    reference_key: Reference histogram key, e.g. "QED (MC)"
    prc:           Percentiles for the matrix of histograms
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
    cmap = plt.get_cmap(cmap_name)
    return [cmap(i) for i in range(n)]

def plot_1d_hist(ax, values, bins, label, color, norm_factor, lw=1.0):
    """
    1D histogram
    """
    h, edges = np.histogram(values, bins=bins)
    h = norm_factor * h
    ax.step(edges, np.append(h, h[-1]), where='post', label=label, color=color, lw=lw)

def plot_2d_contour_kde(ax, x, y, label, color, lw=1.0):
    """
    2D contours with KDE
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
    """
    Plot 1D differential cross section
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
            label_str = fr'{label} ($\tilde{{\chi}}^2$ = {chi2:.1f})'
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
    """
    Create matrix histogram plot (1D on the diagonal, 2D off-diagonal)
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
                            label=fr'{name} ($\tilde{{\chi}}^2$ = {chi2:.1f})', color=colors[k], lw=1.0, zorder=100-k)
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
