# Cool visualizations
#
# m.mieskolainen@imperial.ac.uk, 2025

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Arc
from matplotlib.colors import ListedColormap

import numpy as np
import numba
from scipy.integrate import odeint

from . import qedgen

def scattering_potential_qed(
    m: float = float(1.0),
    b0 = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
    vz0: float = 1.0,
    vy0: float = 0.0,
    potential_k: float = None,
    total_time: float = 400.0,
    sqrts: float = 100.0,
    fontsize: int=12, cmap = plt.cm.pink,
    facecolor: str='black', labelcolor: str='white', spinecolor: str='white'):
    """
    Non-relativistic classic potential V(r) = k/r scattering and QED matrix element squared visualizations.
    
    (automatic kinematic matching between classic and quantum could be added)
    
    Args:
        Classical simulation setup
        
        m:           Particle mass
        b0:          Impact parameters
        vz0:         Initial z-velocity
        vy0:         Initial y-velocity
        potential_k: Potential sign
                     (-1 for attractive, 1 for repulsive, None for both evaluated)
        total_time:  Simulation time
        
        sqrts:       CM energy to evaluate the matrix element
    
    Set potential_k to None for both +-1 trajectories overlaid.
    """
    sqrts = float(sqrts)
    
    # State time derivative
    @numba.njit
    def deriv(state, k, m, eps=1e-8):
        z, y, vz, vy = state
        
        r2 = z*z + y*y + eps*eps
        r3 = r2 * np.sqrt(r2)
        
        az = k * z / (m * r3)
        ay = k * y / (m * r3)
        
        return np.array([vz, vy, az, ay])
    
    # Runge-Kutta 4th order integration step
    @numba.njit
    def rk4_step(state, dt, k, m, eps=1e-8):
        k1 = deriv(state, k, m, eps)
        k2 = deriv(state + 0.5*dt*k1, k, m, eps)
        k3 = deriv(state + 0.5*dt*k2, k, m, eps)
        k4 = deriv(state + dt*k3, k, m, eps)
        
        return state + dt*(k1 + 2*k2 + 2*k3 + k4)/6
    
    @numba.njit
    def simulate_trajectory(b, k, m, total_time, dt=0.02, eps=1e-8):
        
        steps = int(total_time / dt)
        
        # Initial state: particle starts far left at z = -total_time/2, moving along +z
        state = np.array([-total_time/2, b, vz0, vy0])  # [z, y, vz, vy]
        traj  = np.zeros((steps, 2))
        
        for i in range(steps):
            traj[i] = state[:2]
            state = rk4_step(state, dt, k, m, eps)
        
        vz, vy = state[2], state[3]
        theta = np.arctan2(vy, vz)  # scattering angle
        
        return traj, theta
    
    # -----------------------------------------------------------------------
    # Generate classic trajectories & QED amplitude squared weights
    
    if potential_k is None:
        k_values = [-1,1]
    else:
        k_values = [potential_k]
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 14))
    fig.patch.set_facecolor(facecolor)
    ax.set_facecolor(facecolor)
    
    # Different signs
    for k in k_values:

        trajectories, weights, costheta = [], [], []
        
        # Impact parameters
        for b in b0:
            traj, theta = simulate_trajectory(b=b, k=k, m=m, total_time=total_time)
            trajectories.append(traj)
            
            # Compute t
            t = qedgen.costheta_to_t(costheta=np.cos(theta), s=sqrts**2)
            
            if k < 0:
                # Attractive potential -> e+e- -> Bhabha scattering
                amp2 = qedgen.bhabha_amplitude2(s=sqrts**2, t=t)
            
            elif k > 0:
                # Repulsive potential  -> e-e- -> Moeller scattering
                amp2 = qedgen.moeller_amplitude2(s=sqrts**2, t=t)
            
            weights.append(amp2)
            costheta.append(np.cos(theta))
        
        print(f'costheta = {costheta}')
        weights = np.array(weights)
        
        # Log-scale for the visualization
        weights = np.log(weights)
        
        # Normalize color scheme
        norm = plt.Normalize(vmin=weights.min(), vmax=weights.max())
        
        arrow_len = 2.0
        
        for traj, w in zip(trajectories, weights):
            z_vals, y_vals = traj[:,0], traj[:,1]
            col = cmap(norm(w))
            ax.plot(z_vals, y_vals, color=col, lw=1.5)
            
            # compute the direction vector at the midpoint
            mid = len(traj) // 2
            p_prev, p_next = traj[mid-1], traj[mid+1]
            dz, dy = p_next - p_prev
            mag = np.hypot(dz, dy)
            if mag > 0:
                # scale the step
                dz, dy = dz/mag * arrow_len, dy/mag * arrow_len
                
                # draw an arrow with a visible head
                ax.annotate(
                    "",
                    xy=(traj[mid,0] + dz, traj[mid,1] + dy),
                    xytext=(traj[mid,0], traj[mid,1]),
                    arrowprops=dict(
                        arrowstyle="->",        # simple arrow
                        color=col,
                        lw=1.5,
                        mutation_scale=12       # controls head size
                    )
                )
        
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        
    #cbar = fig.colorbar(sm, ax=ax, shrink=1.0, pad=0.02)
    #cbar.ax.set_yticks([])       # remove ticks
    #cbar.ax.set_ylabel('')       # remove label
    #for spine in cbar.ax.spines.values():
    #    spine.set_visible(True)
    
    ax.set_xlim(-40, 40)
    ax.set_ylim(0, 25)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['bottom'].set_color(spinecolor)
    ax.spines['left'].set_color(spinecolor)
    
    ax.set_xlabel('beam axis ($z$)', color=labelcolor, fontsize=fontsize)
    ax.set_ylabel('transverse impact parameter ($b$)', color=labelcolor, fontsize=fontsize)
    
    if potential_k is None:
        att_txt = "repulsive and attractive" # Both
    elif potential_k > 0:
        att_txt = "repulsive"
    else:
        att_txt = "attractive"
    
    sgn_txt = f"{potential_k:0.0f}" if (potential_k is not None) else f"\\pm 1"
    
    ax.set_title(f'Classic {att_txt} potential $V(r)={sgn_txt}/r$ scattering trajectories \n' + 
                 f'QED Scattering Amplitude Squared $|\\mathcal{{M(s,\\Theta}})|^2$ $\\sim$ log color',
                 color=labelcolor, fontsize=fontsize)
    
    ax.set_aspect('equal', 'box')
    plt.tight_layout()
    
    # -----------------------------------------------
    # Create scattering angle visualization
    
    inset_ax = inset_axes(ax, width="20%", height="25%", loc='lower right', borderpad=1.1)
    inset_ax.set_facecolor("black")
    
    # Draw initial beam
    inset_ax.annotate('', xy=(0.09, 0), xytext=(-1.5, 0),
                      arrowprops=dict(arrowstyle='->', color='grey', lw=1.5))
    
    # Outgoing trajectory
    theta = np.pi / 4
    r = 1.5
    x_end = r * np.cos(theta)
    y_end = r * np.sin(theta)
    inset_ax.annotate('', xy=(x_end, y_end), xytext=(0, 0),
                      arrowprops=dict(arrowstyle='->', color='grey', lw=1.5))
    
    # Draw angle arc
    arc = Arc((0, 0), 1.6, 1.6, angle=0,
              theta1=0, theta2=np.degrees(theta),
              color='grey', lw=1)
    inset_ax.add_patch(arc)
    
    # Insert theta
    inset_ax.text(0.35, 0.05, r'$\Theta$', color='grey', fontsize=14)
    
    inset_ax.set_xlim(-1.6, 1.6)
    inset_ax.set_ylim(-0.2, 1.6)
    inset_ax.set_xticks([]); inset_ax.set_yticks([])
    for spine in inset_ax.spines.values():
        spine.set_color(spinecolor)
    inset_ax.set_aspect('equal', 'box')
    # -----------------------------------------------
    
    plt.show()

    return fig, ax

def plot_mandelstam(m=0.5):
    """
    Mandelstam plane
    """
    
    # === Create a grid in (s,t) ===
    # Choose ranges wide enough to capture the regions of interest.
    s_vals = np.linspace(-6, 10, 400)
    t_vals = np.linspace(-6, 10, 400)
    S, T = np.meshgrid(s_vals, t_vals)
    U = 4*m*m - S - T   # Using s+t+u = 4m^2  =>  u = 4m^2 - s - t
    
    # === Define masks for the three channels ===
    # s-channel: s > 4m^2, t < 0, and (s + t) >= 4m^2 (which ensures u <= 0)
    mask_s = (S > 4*m*m) & (T < 0) & ((S + T) >= 4*m*m)
    
    # t-channel: t > 4m^2, s < 0, and (s + T) >= 4m^2 (which ensures u <= 0)
    mask_t = (T > 4*m*m) & (S < 0) & ((S + T) >= 4*m*m)
    
    # u-channel: u > 4m^2, i.e. 4m^2 - s - t > 4m^2, which implies s+t < 0 (with s < 0 and t < 0)
    mask_u = (S < 0) & (T < 0) & ((S + T) < 0)
    
    # Create an array to hold region labels:
    # We'll assign integers: 1 = s-channel, 2 = t-channel, 3 = u-channel.
    region = np.full(S.shape, np.nan)
    region[mask_s] = 1   # s-channel
    region[mask_t] = 2   # t-channel
    region[mask_u] = 3   # u-channel
    
    # Define a discrete colormap for the channels.
    cmap = ListedColormap(['limegreen', 'lightblue', 'violet'])
    
    # === Plotting the Mandelstam Diagram ===
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot the regions using pcolormesh.
    mesh = ax.pcolormesh(S, T, region, cmap=cmap, shading='auto', vmin=1, vmax=3, alpha=0.6)
    
    # Plot the boundary lines:
    
    # Vertical line: s = 4m^2
    ax.plot(np.full_like(t_vals, 4*m*m), t_vals, 'k--', lw=1, label=rf'$s=4m^2$')
    # Horizontal line: t = 4m^2
    ax.plot(s_vals, np.full_like(s_vals, 4*m*m), 'k--', lw=1, label=rf'$t=4m^2$')
    # Diagonal line: u = 4m^2  <=>  s+t = 4m^2  =>  t = 4m^2 - s
    ax.plot(s_vals, 4*m*m - s_vals, 'k--', lw=1, label=rf'$u=4m^2$')
    
    ax.text(6, -1.5, 
            "$s$-domain (experiment)\n($p_1 + p_2 \\to p_3 + p_4$)", 
            fontsize=13, ha='center', va='center', color='k')
    
    ax.text(-2.5, 7, 
            "$t$-domain (crossed)\n($p_1 + \\bar{p}_3 \\to \\bar{p}_2 + p_4$)", 
            fontsize=13, ha='center', va='center', color='k')
    
    ax.text(-3, -3, 
            "$u$-domain (crossed)\n($p_1 + \\bar{p}_4 \\to p_3 + \\bar{p}_2$)", 
            fontsize=13, ha='center', va='center', color='k')

    ax.plot(s_vals, np.zeros_like(s_vals), 'gray')
    ax.plot(np.zeros_like(t_vals), t_vals, 'gray')
    
    # Set axis labels and title
    ax.set_xlabel('$s$', fontsize=14)
    ax.set_ylabel('$t$', fontsize=14)
    ax.set_title(f'The mathematical landscape of Mandelstam with $m = {m}$', fontsize=16)
    
    # Add explanatory text at the bottom
    explanation_text = (
        "For a $2 \\rightarrow 2$ with equal masses: $s+t+u=4m^2$. The 'physical' domains are: \n"
        "$s>4m^2$ ($t, u<0$); "
        "$t>4m^2$ ($s, u<0$); "
        "$u>4m^2$ ($s, t <0$). \n\n"
        "Real world collider experiments have direct access only to the physical $s$-domain. \n\n"
        "The physical $t$- and $u$-domains have two particles crossed (flip momentum and charge). \n\n"
        "Some parts of the unphysical domain (white area) can be used to constrain the amplitude e.g. via analyticity (aka classic and modern Bootstrap method, very advanced topic)"
    )
    ax.text(-6, -8, explanation_text, fontsize=10, va='top', wrap=True)
    
    # Set plot limits
    ax.set_xlim(-6, 10)
    ax.set_ylim(-6, 10)
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax

def OU_simulation(alpha=3.0, sigma=2.5, N=25, T=1.0, mu0=4.0, std0=0.75, plot_title=True, plot_hist=False):
    """
    Visualization using drift+diffusion Ornstein–Uhlenbeck process with both SDE and ODE
    
    Args:
        alpha, sigma: OU process parameters
        N:            Number of SDE trajectories
        T:            Evolution time endpoint
        mu0, std0:    Data distribution, mimodal gaussian (+- mean, std)
    """
    
    sigma2 = sigma**2
    v0     = std0**2

    # Bi-modal Gaussian data density
    def pdf_data(x, mean, std):
        return 0.5 * (np.exp(-0.5*((x - mean)/std)**2)/(std*np.sqrt(2*np.pi)) + np.exp(-0.5*((x+mean)/std)**2)/(std*np.sqrt(2*np.pi)))
    
    # Analytical process mean and variance when X_0 ~ N(mu0, v0)
    def m(t): return mu0 * np.exp(-alpha*t)
    def v(t): return v0 * np.exp(-2*alpha*t) + (sigma2/(2*alpha))*(1 - np.exp(-2*alpha*t))
    
    def score(x, t):
        mt = m(t)
        vt = v(t)
        st = np.sqrt(vt)
    
        # Full normalized Gaussian components
        p1 = np.exp(-0.5 * ((x - mt) / st)**2) / (st * np.sqrt(2 * np.pi))
        p2 = np.exp(-0.5 * ((x + mt) / st)**2) / (st * np.sqrt(2 * np.pi))
    
        # Component score functions (derivative of log Gaussian)
        g1 = -(x - mt) / vt
        g2 = -(x + mt) / vt
    
        # Mixture score
        return (p1 * g1 + p2 * g2) / (p1 + p2)
    
    # ODE
    def forward_ode(x, t):
        return -alpha*x + 0.5*sigma2*score(x, t)
    
    def reverse_ode(x, s):
        t = T - s
        return -forward_ode(x, t)
    
    # ------------------------------------------------------------------
    ## ODE simulation
    # ------------------------------------------------------------------
    
    # Time grids
    n_steps_ode = 500
    t_fwd_ode   = np.linspace(0.0, T, n_steps_ode)
    s_rev_ode   = np.linspace(0.0, T, n_steps_ode)
    t_rev_ode   = T - s_rev_ode
    
    # ODE trajectories with numerical integration
    x_plus   = odeint(forward_ode,  mu0,   t_fwd_ode).flatten()
    x_minus  = odeint(forward_ode, -mu0,   t_fwd_ode).flatten()
    
    # Start from where the forward ended
    x0p      = x_plus[-1]
    x0m      = x_minus[-1]
    xr_plus  = odeint(reverse_ode, x0p, s_rev_ode, rtol=1e-5, atol=1e-8).flatten()
    xr_minus = odeint(reverse_ode, x0m, s_rev_ode, rtol=1e-5, atol=1e-8).flatten()
    
    # ODE density heatmap obtained analytically
    x_vals  = np.linspace(-6, 6, 400)
    density = np.zeros((n_steps_ode, x_vals.size))
    
    for i, tt in enumerate(t_fwd_ode):
        
        # Analytical mean and variance at time t
        mt, vt = m(tt), v(tt)
        st     = np.sqrt(vt)
        
        # Evaluate bi-modal gaussian density
        density[i] = pdf_data(x=x_vals, mean=mt, std=st)
    
    density_rev = density[::-1]
    
    # ------------------------------------------------------------------
    ## SDE simulation
    # ------------------------------------------------------------------
    
    # Forward SDE simulation
    dt      = 5e-3
    n_steps = int(T / dt)
    kappa   = np.exp(-alpha * dt)
    var_noise = (sigma2 / (2 * alpha)) * (1 - np.exp(-2 * alpha * dt))
    
    t_fwd = np.linspace(0, T, n_steps + 1)
    t_rev = t_fwd[::-1]
    
    # Initial samples from symmetric Gaussian mixture
    x0 = np.random.choice([-mu0, mu0], size=N) + np.sqrt(v0) * np.random.randn(N)
    
    # Simulate forward SDE
    x_fwd = np.empty((n_steps + 1, N))
    x_fwd[0] = x0
    for i in range(n_steps):
        noise = np.random.randn(N)
        x_fwd[i + 1] = kappa * x_fwd[i] + np.sqrt(var_noise) * noise
    
    # Reverse SDE sampling (Gaussian mixture kernel)
    m_rev = m(t_rev)      # Mean at reverse times
    v_rev = v(t_rev)      # Variance at reverse times
    
    x_rev = np.empty_like(x_fwd)
    x_rev[0] = x_fwd[-1]  # Start from last forward sample
    
    for i in range(n_steps):
        xt = x_rev[i]
        mt, vt = m_rev[i], v_rev[i]
        s_t = np.sqrt(vt)
    
        # Density values for two modes
        p1 = np.exp(-0.5 * ((xt - mt) / s_t) ** 2)
        p2 = np.exp(-0.5 * ((xt + mt) / s_t) ** 2)
        w1 = p1 / (p1 + p2)
        
        # Conditional means and shared std for the reverse kernel
        cov   = kappa * vt
        denom = kappa**2 * vt + var_noise
        std   = np.sqrt(vt - cov**2 / denom)
    
        mu1 =  mt + cov * (xt - kappa * mt) / denom
        mu2 = -mt + cov * (xt + kappa * mt) / denom
        
        # Sample from weighted Gaussian mixture
        pick = np.random.rand(N) < w1
        mu   = np.where(pick, mu1, mu2)
        x_rev[i + 1] = mu + std * np.random.randn(N)
    
    # ------------------------------------------------------------------
    ## Plots
    # ------------------------------------------------------------------
    
    fig, (ax_f, ax_r) = plt.subplots(1, 2, figsize=(12, 5), facecolor='black', sharey=True, gridspec_kw={'wspace': 0})
    
    for ax in (ax_f, ax_r):
        ax.set_facecolor('black')
        ax.tick_params(colors='white')
        ax.set_ylim(-6, 6)
        ax.axvspan(1 - dt/2, 1 + dt/2, color='red', alpha=0.15)
    
    colors = plt.cm.viridis(np.linspace(0, 1, N))
    
    # Forward panel
    ax_f.imshow(density.T, origin='lower', extent=[0, T, -6, 6], aspect='auto', cmap='hot', alpha=0.3)
    ax_f.plot(t_fwd_ode, x_plus,  color='white', lw=2)
    ax_f.plot(t_fwd_ode, x_minus, color='white', lw=2)
    
    for i in range(N):
        ax_f.plot(t_fwd, x_fwd[:, i], color=colors[i], alpha=0.6)
    
    # Forward marginals
    if plot_hist:
        for data, xpos in [(x_fwd[0], 0.0), (x_fwd[-1], T)]:
            c, edges = np.histogram(data, bins=40, density=True)
            cen = (edges[:-1] + edges[1:]) / 2
            h = edges[1] - edges[0]
            ax_f.barh(cen, c, height=h, left=xpos, color='white', alpha=0.3)
    else:
        for tt, xpos in [(0.0, 0.0), (T, T)]:
            mt, vt = m(tt), v(tt)
            st = np.sqrt(vt)
            for sign in [-1, 1]:
                xplot = np.linspace(-6, 6, 300)
                p = 0.5 * np.exp(-0.5*((xplot - sign * mt)/st)**2) / (st * np.sqrt(2 * np.pi))
                ax_f.fill_betweenx(xplot, xpos, xpos + p, color='white', alpha=0.10)
    
    ax_f.set_xlim(0, T)
    ax_f.set_xlabel('$t$', color='white')
    ax_f.set_ylabel('$x$', color='white')
    ax_f.spines['left'].set_color('white')
    ax_f.spines['right'].set_color('none')
    ax_f.spines['bottom'].set_color('white')
    
    # Reverse panel
    ax_r.imshow(density_rev.T, origin='lower', extent=[T, 0, -6, 6], aspect='auto', cmap='hot', alpha=0.3)
    ax_r.plot(t_rev_ode, xr_plus,  color='white', lw=2)
    ax_r.plot(t_rev_ode, xr_minus, color='white', lw=2)
    
    for i in range(N):
        ax_r.plot(t_rev, x_rev[:, i], color=colors[i], alpha=0.6)
    
    # Reverse marginals
    if plot_hist:
        for data, xpos in [(x_rev[0], T), (x_rev[-1], 0.0)]:
            c, edges = np.histogram(data, bins=40, density=True)
            cen = (edges[:-1] + edges[1:]) / 2
            h = edges[1] - edges[0]
            ax_r.barh(cen, c, height=h, left=xpos, color='white', alpha=0.3)
    else:
        for tt, xpos in [(T, T), (0.0, 0.0)]:
            mt, vt = m(tt), v(tt)
            st = np.sqrt(vt)
            for sign in [-1, 1]:
                xplot = np.linspace(-6, 6, 300)
                p = 0.5 * np.exp(-0.5*((xplot - sign * mt)/st)**2) / (st * np.sqrt(2 * np.pi))
                ax_r.fill_betweenx(xplot, xpos, xpos + p, color='white', alpha=0.10)
    
    ax_r.set_xlim(T, 0)
    ax_r.set_xlabel('$t$', color='white')
    ax_r.spines['left'].set_color('none')
    ax_r.spines['right'].set_color('white')
    ax_r.spines['bottom'].set_color('white')
    ax_r.yaxis.set_label_position('right')
    ax_r.yaxis.tick_right()
    ax_r.set_ylabel('$x$', color='white')

    if plot_title:
        
        # Titles for individual panels
        ax_f.set_title(r'SDE: $dx = -\alpha x\,dt + \sigma\,dw$', color='white')
        ax_r.set_title(r'SDE$^{-1}$: $dx = (\alpha x + \sigma^2 \nabla_x \log p_t(x))\,dt + \sigma\,d\widebar{w}$', color='white')
        
        # Shared ODE title in the center
        fig.suptitle(r'$\frac{dx}{dt} = -\alpha x + \frac{\sigma^2}{2} \nabla_x \log p_t(x)$', color='white', fontsize=14)
        
    plt.tight_layout(pad=0)
    plt.show()
    
    return fig, (ax_f, ax_r)
