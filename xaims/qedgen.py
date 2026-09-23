# MC generator for 2 -> 2 QED process (Bhabha and Moeller scattering at tree-level)
#
# Values provided in natural units (GeV scale)
#
# m.mieskolainen@imperial.ac.uk, 2025

import numpy as np
import numba
from enum import IntEnum

# --------------------------------------------------------
# Scattering amplitude modes as enum (due to numba)

class Mode(IntEnum):
    QED_bhabha      = 0
    QED_bhabha_t    = 1
    QED_bhabha_s    = 2
    QED_bhabha_int  = 3
    
    QED_moeller     = 4
    QED_moeller_t   = 5
    QED_moeller_u   = 6
    QED_moeller_int = 7

@numba.njit
def amplitude2_wrapper(s: float, t: np.ndarray, theory: Mode):
    
    """Select the QED squared scattering amplitude

    Args:
        s (float): Squared CM energy in GeV^2
        t (float | np.ndarray): Momentum transfer (...) in GeV^2
        theory (Mode): Scattering theory selector

    Returns:
        float | np.ndarray: Spin-averaged squared amplitude, matching t
    """

    if theory == Mode.QED_bhabha:
       return bhabha_amplitude2(s=s, t=t)

    if theory == Mode.QED_bhabha_t:   # diagnostics
        return bhabha_amplitude2_t(s=s, t=t)

    if theory == Mode.QED_bhabha_s:   # diagnostics
        return bhabha_amplitude2_s(s=s, t=t)

    if theory == Mode.QED_bhabha_int: # diagnostics
        return bhabha_amplitude2_int(s=s, t=t)
    
    if theory == Mode.QED_moeller:
        return moeller_amplitude2(s=s, t=t)

    if theory == Mode.QED_moeller_t:   # diagnostics
        return moeller_amplitude2_t(s=s, t=t)

    if theory == Mode.QED_moeller_u:   # diagnostics
        return moeller_amplitude2_u(s=s, t=t)

    if theory == Mode.QED_moeller_int: # diagnostics
        return moeller_amplitude2_int(s=s, t=t)

# --------------------------------------------------------

# --------------------------------------------------------
# Constants

# Conversion from barns from GeV^-2
nat2mb = 0.389379
nat2ub = nat2mb * 1e3
nat2nb = nat2mb * 1e6
nat2pb = nat2mb * 1e9

# Electron mass [GeV]
m_e = 5.11E-4

# Coupling
alpha_qed = 1.0 / 137.035999 # no running
e_qed = np.sqrt(4.0 * np.pi * alpha_qed)
# --------------------------------------------------------

def extract_observables(x: np.ndarray):
    """Extract momentum, pseudorapidity, and azimuth observables

    Args:
        x (np.ndarray): Three-momenta (N, 3) in GeV

    Returns:
        tuple: Six arrays (N,) for px, py, pz, pt, eta, phi, plus labels and units
    """

    px, py, pz = x[:, 0], x[:, 1], x[:, 2]
    
    pt   = np.sqrt(px**2 + py**2)
    eta  = np.arcsinh(pz / np.clip(pt, 1e-6, None))
    phi  = np.arctan2(py, px)
    labels = ["$p_x$", "$p_y$", "$p_z$", "$p_T$", "$\\eta$", "$\\phi$"]
    units  = ["GeV", "GeV", "GeV", "GeV", "unit", "rad"]
    
    return [px, py, pz, pt, eta, phi], labels, units

@numba.njit
def pz_to_t(pz: np.ndarray, s: float, m: float=5.11e-4):
    """Convert longitudinal CM momentum to momentum transfer

    Args:
        pz (float | np.ndarray): Longitudinal momentum (...) in GeV
        s (float): Squared CM energy in GeV^2
        m (float): Particle mass in GeV

    Returns:
        float | np.ndarray: Momentum transfer in GeV^2, matching pz
    """

    p_star = np.sqrt(0.25*s - m**2)

    return 2.0*m**2 - 0.5*s + 2.0*p_star*pz

@numba.njit
def t_to_costheta(t: np.ndarray, s: float, m: float=5.11e-4):
    """Convert momentum transfer to the CM scattering cosine

    Args:
        t (float | np.ndarray): Momentum transfer (...) in GeV^2
        s (float): Squared CM energy in GeV^2
        m (float): Particle mass in GeV

    Returns:
        float | np.ndarray: cos(theta), matching t
    """

    p_star2 = 0.25*s - m**2

    return (t - 2.0*m**2 + 0.5*s) / (2.0*p_star2)

@numba.njit
def costheta_to_t(costheta: np.ndarray, s: float, m: float=5.11e-4):
    """Convert the CM scattering cosine to momentum transfer

    Args:
        costheta (float | np.ndarray): Scattering cosines (...)
        s (float): Squared CM energy in GeV^2
        m (float): Particle mass in GeV

    Returns:
        float | np.ndarray: Momentum transfer in GeV^2, matching costheta
    """

    p_star2 = 0.25*s - m**2

    return 2.0*m**2 - 0.5*s + 2.0*p_star2*costheta

@numba.njit
def kallen_lambda(s: float, m1: float, m2: float):
    """Evaluate the Kallen function lambda(s, m1^2, m2^2)

    Args:
        s (float): Squared CM energy in GeV^2
        m1 (float): First particle mass in GeV
        m2 (float): Second particle mass in GeV

    Returns:
        float: Kallen function in GeV^4
    """

    m1_sq = m1**2
    m2_sq = m2**2

    return s**2 + m1_sq**2 + m2_sq**2 - 2*s*m1_sq - 2*s*m2_sq - 2*m1_sq*m2_sq

@numba.njit
def kinematics(s: float, m1: float=5.11e-4, m2: float=5.11e-4):
    """Evaluate the two-body flux and phase-space factor

    Args:
        s (float): Squared CM energy in GeV^2
        m1 (float): First particle mass in GeV
        m2 (float): Second particle mass in GeV

    Returns:
        float: 1 / (16 * pi * lambda) in GeV^-4
    """

    return 1.0 / (16.0 * np.pi * kallen_lambda(s=s, m1=m1, m2=m2))

# ------------------------------------------------------------------
# FeynCalc Mathematica code for Bhabha spin-average amplitude squared

"""
<< FeynCalc`

Remove["Global`*"]

prop[p_, m_] := GS[p] + m;
PR = (1 + GA[5])/2;
PL = (1 - GA[5])/2;

Line1 := prop[p3, m] . GA[mu] . prop[p1, m] . GA[nu]  
Line2 := prop[p2, -m] . GA[mu] . prop[p4, -m] . GA[nu]
Line3 := prop[p2, -m] . GA[mu] . prop[p1, m] . GA[nu]
Line4 := prop[p3, m] . GA[mu] . prop[p4, -m] . GA[nu]
Line5 := prop[p3, m] . GA[mu] . prop[p1, m] . GA[nu] . prop[p2, -m] . GA[mu] . prop[p4, -m] . GA[nu] 

M2t  = Simplify[Contract[TR[Line1] TR[Line2]]]
M2s  = Simplify[Contract[TR[Line3] TR[Line4]]]
M2st = -2*Simplify[Contract[TR[Line5]]]

kin = {
   SP[p1, p1] -> m^2,
   SP[p2, p2] -> m^2,
   SP[p3, p3] -> m^2,
   SP[p4, p4] -> m^2,
   SP[p1, p2] -> (s - 2 m^2)/2,
   SP[p3, p4] -> (s - 2 m^2)/2,
   SP[p1, p3] -> (2 m^2 - t)/2,
   SP[p2, p4] -> (2 m^2 - t)/2,
   SP[p1, p4] -> (2 m^2 - u)/2,
   SP[p2, p3] -> (2 m^2 - u)/2
};

spinavg = 1/4;

(* t-channel term *)
res1 = e^4 * M2t / t^2 // FCE;
res1 = spinavg * res1 /. kin 

(* s-channel term *)
res2 = e^4 * M2s / s^2 // FCE;
res2 = spinavg * res2 /. kin 

(* interference term *)
res3 = e^4 * M2st / (t*s) // FCE;
res3 = spinavg * res3 /. kin // Simplify
"""
# ------------------------------------------------------------------

@numba.njit
def bhabha_amplitude2_massless(s: float, t: np.ndarray):
    """Evaluate the massless spin-averaged Bhabha squared amplitude

    Args:
        s (float): Squared CM energy in GeV^2
        t (float | np.ndarray): Momentum transfer (...) in GeV^2

    Returns:
        float | np.ndarray: Dimensionless contribution, matching t
    """

    u = - s - t    

    return 2*e_qed**4 * ((s**2 + u**2) / (t*t) + (t**2 + u**2) / (s*s) + (2*u**2) / (t*s))

@numba.njit
def bhabha_amplitude2(s: float, t: np.ndarray, m=5.11e-4):
    """Evaluate the massive spin-averaged Bhabha squared amplitude

    Args:
        s (float): Squared CM energy in GeV^2
        t (float | np.ndarray): Momentum transfer (...) in GeV^2
        m (float): Particle mass in GeV

    Returns:
        float | np.ndarray: Dimensionless contribution, matching t
    """

    u = 4*m**2 - s - t
    
    num_t   = (s - 2*m**2)**2 + (u - 2*m**2)**2 + 4*m**2*t
    num_s   = (t - 2*m**2)**2 + (u - 2*m**2)**2 + 4*m**2*s
    num_int = 2*(u**2 - 8*m**2*u + 12*m**4)
    
    return 2*e_qed**4 * (num_t/(t*t) + num_s/(s*s) + num_int/(s*t))

@numba.njit
def bhabha_amplitude2_t(s: float, t: np.ndarray, m=5.11e-4):
    """Evaluate the Bhabha t-channel squared contribution

    Args:
        s (float): Squared CM energy in GeV^2
        t (float | np.ndarray): Momentum transfer (...) in GeV^2
        m (float): Particle mass in GeV

    Returns:
        float | np.ndarray: Dimensionless contribution, matching t
    """

    u     = 4*m**2 - s - t
    num_t = (s - 2*m**2)**2 + (u - 2*m**2)**2 + 4*m**2*t
    
    return 2*e_qed**4 * (num_t/(t*t))

@numba.njit
def bhabha_amplitude2_s(s: float, t: np.ndarray, m=5.11e-4):
    """Evaluate the Bhabha s-channel squared contribution

    Args:
        s (float): Squared CM energy in GeV^2
        t (float | np.ndarray): Momentum transfer (...) in GeV^2
        m (float): Particle mass in GeV

    Returns:
        float | np.ndarray: Dimensionless contribution, matching t
    """

    u     = 4*m**2 - s - t
    num_s = (t - 2*m**2)**2 + (u - 2*m**2)**2 + 4*m**2*s
    
    return 2*e_qed**4 * (num_s/(s*s))

@numba.njit
def bhabha_amplitude2_int(s: float, t: np.ndarray, m=5.11e-4):
    """Evaluate the Bhabha s/t interference contribution

    Args:
        s (float): Squared CM energy in GeV^2
        t (float | np.ndarray): Momentum transfer (...) in GeV^2
        m (float): Particle mass in GeV

    Returns:
        float | np.ndarray: Dimensionless contribution, matching t
    """

    u       = 4*m**2 - s - t
    num_int = 2*(u**2 - 8*m**2*u + 12*m**4)
    
    return 2*e_qed**4 * (num_int/(t*s))

# ------------------------------------------------------------------

@numba.njit
def moeller_amplitude2(s: float, t: np.ndarray, m=5.11e-4):
    """Evaluate the Moeller squared amplitude with the 1/2 symmetry factor

    Args:
        s (float): Squared CM energy in GeV^2
        t (float | np.ndarray): Momentum transfer (...) in GeV^2
        m (float): Particle mass in GeV

    Returns:
        float | np.ndarray: Dimensionless contribution, matching t
    """

    u = 4*m**2 - s - t
    
    num_t = (s - 2*m**2)**2 + (u - 2*m**2)**2 + 4*m**2*t
    num_u = (t - 2*m**2)**2 + (s - 2*m**2)**2 + 4*m**2*u
    num_int = 2*(s**2 - 8*m**2*s + 12*m**4)
    
    S = 0.5 # Cross-section symmetry factor (identical final state particles)
    
    return S * 2*e_qed**4 * (num_t/(t*t) + num_u/(u*u) + num_int/(t*u))

@numba.njit
def moeller_amplitude2_t(s: float, t: np.ndarray, m=5.11e-4):
    """Evaluate the Moeller t-channel contribution with the 1/2 symmetry factor

    Args:
        s (float): Squared CM energy in GeV^2
        t (float | np.ndarray): Momentum transfer (...) in GeV^2
        m (float): Particle mass in GeV

    Returns:
        float | np.ndarray: Dimensionless contribution, matching t
    """

    u = 4*m**2 - s - t
    num_t = (s - 2*m**2)**2 + (u - 2*m**2)**2 + 4*m**2*t
    
    S = 0.5 # Cross-section symmetry factor (identical final state particles)
    
    return S * 2*e_qed**4 * (num_t/(t*t))

@numba.njit
def moeller_amplitude2_u(s: float, t: np.ndarray, m=5.11e-4):
    """Evaluate the Moeller u-channel contribution with the 1/2 symmetry factor

    Args:
        s (float): Squared CM energy in GeV^2
        t (float | np.ndarray): Momentum transfer (...) in GeV^2
        m (float): Particle mass in GeV

    Returns:
        float | np.ndarray: Dimensionless contribution, matching t
    """

    u = 4*m**2 - s - t
    num_u = (t - 2*m**2)**2 + (s - 2*m**2)**2 + 4*m**2*u
    
    S = 0.5 # Cross-section symmetry factor (identical final state particles)
    
    return S * 2*e_qed**4 * (num_u/(u*u))

@numba.njit
def moeller_amplitude2_int(s: float, t: np.ndarray, m=5.11e-4):
    """Evaluate Moeller t/u interference with the 1/2 symmetry factor

    Args:
        s (float): Squared CM energy in GeV^2
        t (float | np.ndarray): Momentum transfer (...) in GeV^2
        m (float): Particle mass in GeV

    Returns:
        float | np.ndarray: Dimensionless contribution, matching t
    """

    u = 4*m**2 - s - t
    num_int = 2*(s**2 - 8*m**2*s + 12*m**4)
    
    S = 0.5 # Cross-section symmetry factor (identical final state particles)
    
    return S * 2*e_qed**4 * (num_int/(t*u))

# ------------------------------------------------------------------

@numba.njit
def alpha_eff(Q2):
    # Effective running coupling (QED 1-loop)
    """Evaluate the one-loop running QED fine-structure constant

    Args:
        Q2 (float): Momentum scale squared in GeV^2

    Returns:
        float: Effective fine-structure constant
    """

    Q2 = max(Q2, m_e**2 + 1e-10)

    return alpha_qed / (1 - alpha_qed / (3*np.pi) * np.log(Q2 / m_e**2))

@numba.njit
def e_qed_eff(Q2):
    """Evaluate the running QED charge

    Args:
        Q2 (float): Momentum scale squared in GeV^2

    Returns:
        float: sqrt(4 * pi * alpha_eff(Q2))
    """

    return np.sqrt(4.0*np.pi*alpha_eff(Q2))

@numba.njit
def dsigma_dt(s: float, t: np.ndarray, theory: Mode):
    """Evaluate the differential cross section in momentum transfer

    Args:
        s (float): Squared CM energy in GeV^2
        t (float | np.ndarray): Momentum transfer (...) in GeV^2
        theory (Mode): Scattering theory selector

    Returns:
        float | np.ndarray: dsigma/dt in GeV^-4, matching t
    """

    return kinematics(s=s) * amplitude2_wrapper(s=s, t=t, theory=theory)

@numba.njit
def dsigma_dcostheta(s: float, costheta: np.ndarray, theory: Mode):
    """Evaluate the differential cross section in scattering cosine

    Args:
        s (float): Squared CM energy in GeV^2
        costheta (float | np.ndarray): Scattering cosines (...)
        theory (Mode): Scattering theory selector

    Returns:
        float | np.ndarray: dsigma/dcos(theta) in GeV^-2, matching costheta
    """

    t   = costheta_to_t(costheta=costheta, s=s)
    jac = 2 * (s/4 - m_e**2) # Jacobian dt / dcos(theta)
    
    return dsigma_dt(s=s, t=t, theory=theory) * jac

@numba.njit
def mc_cross_section(wsum: float, w2sum: float, N: int):
    """Estimate a Monte Carlo integral and its standard error

    Args:
        wsum (float): Sum of trial weights
        w2sum (float): Sum of squared trial weights
        N (int): Positive trial count

    Returns:
        tuple[float, float]: Cross section and standard error in GeV^-2, NaN error for one trial
    """
    
    if N < 1:
        raise ValueError("At least one trial is required")

    I = wsum / N
    if N == 1:
        return I, np.nan

    variance = (w2sum - (wsum**2) / N) / (N * (N - 1))
    sigma = np.sqrt(max(variance, 0.0))

    return I, sigma

@numba.njit
def generate_event(s: float, t_min: float, t_max: float, theory: Mode):
    """Generate one outgoing electron by rejection sampling

    Args:
        s (float): Squared CM energy in GeV^2
        t_min (float): Lower transfer bound in GeV^2, above -(s - 4m^2)
        t_max (float): Upper transfer bound in GeV^2, between t_min and zero
        theory (Mode): Scattering theory selector

    Returns:
        tuple: Momentum array (3,) in GeV, weight sum, squared-weight sum, trial count
    """
    
    if t_min <= -(s - 4*m_e**2):
        raise ValueError("t_min must be greater than -(s - 4*m_e**2) = ", -(s - 4*m_e**2))
    
    if t_max >= 0.0:
        raise ValueError("t_max must be smaller than 0")
    
    if s <= 4 * m_e**2:
        raise ValueError("s must be greater than 4*m_e^2 = ", 4*m_e**2)
    
    # Kinematic quantities in the CM frame
    E_star = np.sqrt(s) / 2
    p_star = np.sqrt(E_star**2 - m_e**2)

    # Integration volume
    V = np.abs(t_max - t_min)
    
    # Precompute maximum weight for acceptance-rejection
    # In general, this needs to be explored by sampling, but here
    # we know that the maximum amplitude squared is obtained when t -> 0
    
    w_max = V * kinematics(s) * max(amplitude2_wrapper(s=s, t=t_max, theory=theory),
                                    amplitude2_wrapper(s=s, t=t_min, theory=theory))
    
    trials = 0
    wsum   = 0
    w2sum  = 0
    
    while True:
        
        r = np.random.rand()
        
        # Convert [0,1] -> t scale
        t = t_min + (t_max - t_min) * r
        
        # -----------------------------------
        # Total weight
        w = V * kinematics(s=s) * amplitude2_wrapper(s=s, t=t, theory=theory)
        # -----------------------------------
        
        trials += 1
        wsum   += w
        w2sum  += w**2
        
        if np.random.uniform(0, w_max) < w:
            break
    
    # Construct 3-momentum of the outgoing electron
    costheta = t_to_costheta(t=t, s=s)
    theta    = np.arccos(costheta)
    phi      = np.random.uniform(0, 2*np.pi)
    
    p3x = p_star * np.sin(theta) * np.cos(phi)
    p3y = p_star * np.sin(theta) * np.sin(phi)
    p3z = p_star * costheta
    
    x = np.array([p3x, p3y, p3z])
    
    return x, wsum, w2sum, trials

@numba.njit
def generator(num_events: int, sqrts: float, t_min: float, t_max: float, theory: Mode):
    """Generate unweighted two-body scattering events and estimate the cross section

    Args:
        num_events (int): Event count N
        sqrts (float): CM energy in GeV
        t_min (float): Lower transfer bound in GeV^2, above -(s - 4m^2)
        t_max (float): Upper transfer bound in GeV^2, between t_min and zero
        theory (Mode): Scattering theory selector

    Returns:
        tuple: Momenta (N, 3) in GeV, cross section and error in GeV^-2, acceptance fraction
    """
    
    s = sqrts**2
    d = int(3)
    x = np.zeros((int(num_events), d), dtype=np.float64)
    
    wsum   = 0
    w2sum  = 0
    trials = 0
    
    for i in range(num_events):
        
        x[i,:], wsum_evt, w2sum_evt, trials_evt = \
            generate_event(s=s, t_min=t_min, t_max=t_max, theory=theory)
        
        wsum   += wsum_evt
        w2sum  += w2sum_evt
        trials += trials_evt
    
    xs_tot, xs_tot_err = mc_cross_section(wsum=wsum, w2sum=w2sum, N=trials)
    
    # Acceptance-Rejection efficiency
    eff = num_events / trials
    
    return x, xs_tot, xs_tot_err, eff


def detector_log_prob(sqrts, momenta, resolution, theta_min, theta_max,
                      theory=Mode.QED_bhabha, n_angles=256):
    """Integrate the QED angular density against isotropic Gaussian detector noise

    Args:
        sqrts (array_like): CM energies in GeV, scalar or (B,), above threshold
        momenta (array_like): Observed Cartesian momenta (B, 3) in GeV
        resolution (float): Gaussian sigma per component in GeV, positive
        theta_min (float): Lower true polar-angle cut in radians, above zero
        theta_max (float): Upper true polar-angle cut in radians, below pi
        theory (Mode): Nonnegative QED differential cross section
        n_angles (int): Polar quadrature nodes, increase to check convergence

    Returns:
        np.ndarray: Cartesian log densities (B,), conditional on accepted event count
    """

    from scipy.special import i0e, logsumexp, roots_legendre

    momenta = np.asarray(momenta, dtype=np.float64)
    energy = np.asarray(sqrts, dtype=np.float64)
    if momenta.ndim != 2 or momenta.shape[1] != 3 or not np.all(np.isfinite(momenta)):
        raise ValueError("Expected finite momenta (B, 3)")
    if energy.shape not in ((), (len(momenta),)) or np.any(~np.isfinite(energy) | (energy <= 2 * m_e)):
        raise ValueError("Expected scalar or (B,) finite CM energies above threshold")
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError("Detector resolution must be finite and positive")
    if not 0 < theta_min < theta_max < np.pi:
        raise ValueError("Require 0 < theta_min < theta_max < pi")
    if not isinstance(n_angles, (int, np.integer)) or n_angles < 2:
        raise ValueError("n_angles must be an integer >= 2")

    energy = np.broadcast_to(energy, (len(momenta),))
    nodes, weights = roots_legendre(n_angles)
    angles = theta_min + (nodes + 1) * (theta_max - theta_min) / 2
    cosine, sine = np.cos(angles), np.sin(angles)
    angular = dsigma_dcostheta(energy[:, None]**2, cosine[None], theory) * sine * weights
    if np.any(~np.isfinite(angular) | (angular < 0)) or np.any(angular.sum(1) <= 0):
        raise ValueError("Theory must define a finite nonnegative angular density")
    angular /= angular.sum(1, keepdims=True)

    radius = np.sqrt(energy**2 / 4 - m_e**2)[:, None]
    pt = np.linalg.norm(momenta[:, :2], axis=1)[:, None]
    pz = momenta[:, 2, None]
    bessel_argument = pt * radius * sine / resolution**2
    # Integrate azimuth analytically, using the scaled Bessel function for stability
    exponent = -0.5 * ((pt - radius * sine)**2 + (pz - radius * cosine)**2) / resolution**2
    with np.errstate(divide="ignore"):
        terms = exponent + np.log(i0e(bessel_argument)) + np.log(angular)

    return logsumexp(terms, axis=1) - 1.5 * np.log(2 * np.pi * resolution**2)
