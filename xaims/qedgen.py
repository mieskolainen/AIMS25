# MC generator for 2 -> 2 QED process (Bhabha scattering at tree-level)
#
# Values provided in natural units (GeV scale).
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
    """
    Compute observables given an array (N,3) of 3-momentum (px, py, pz)
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
    """ Map final state leg pz and Mandelstam s to Mandelstam t
    """
    p_star = np.sqrt(0.25*s - m**2)
    return 2.0*m**2 - 0.5*s + 2.0*p_star*pz

@numba.njit
def t_to_costheta(t: np.ndarray, s: float, m: float=5.11e-4):
    """ Mandelstam t -> cos(theta) scattering angle """
    p_star2 = 0.25*s - m**2
    return (t - 2.0*m**2 + 0.5*s) / (2.0*p_star2)

@numba.njit
def costheta_to_t(costheta: np.ndarray, s: float, m: float=5.11e-4):
    """ Scattering angle cos(theta) -> Mandelstam t """
    p_star2 = 0.25*s - m**2
    return 2.0*m**2 - 0.5*s + 2.0*p_star2*costheta

@numba.njit
def kallen_lambda(s: float, m1: float, m2: float):
    """ Källen kinematic function lambda(s, m1, m2), where m1 and m2 are particle masses """
    m1_sq = m1**2
    m2_sq = m2**2
    return s**2 + m1_sq**2 + m2_sq**2 - 2*s*m1_sq - 2*s*m2_sq - 2*m1_sq*m2_sq

@numba.njit
def kinematics(s: float, m1: float=5.11e-4, m2: float=5.11e-4):
    """ Kinematic part of the cross-section """
    return 1.0 / (16.0 * np.pi * kallen_lambda(s=s, m1=m1, m2=m2))

# ------------------------------------------------------------------
# FeynCalc Mathematica code for Bhabha spin-average amplitude squared

"""
<< FeynCalc`

Remove["Global`*"]

prop[p_, m_] := GS[p] + m;
PR = (1 + GA[5])/2;
PL = (1 - GA[5])/2;

Line1 := GS[p3] . GA[mu] . GS[p1] . GA[nu]  
Line2 := GS[p2] . GA[mu] . GS[p4] . GA[nu]
Line3 := GS[p2] . GA[mu] . GS[p1] . GA[nu]
Line4 := GS[p3] . GA[mu] . GS[p4] . GA[nu]
Line5 := GS[p3] . GA[mu] . GS[p1] . GA[nu] . GS[p2] . GA[mu] . GS[p4] . GA[nu] 

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
    """
    Spin averaged LO-QED scattering amplitude |M(s,t)|^2 in the massless limit
    """
    u = - s - t    
    return 2*e_qed**4 * ((s**2 + u**2) / (t*t) + (t**2 + u**2) / (s*s) + (2*u**2) / (t*s))

@numba.njit
def bhabha_amplitude2(s: float, t: np.ndarray, m=5.11e-4):
    """
    Spin averaged LO-QED scattering amplitude |M(s,t)|^2 for e+e- -> e+e-
    """
    u = 4*m**2 - s - t
    
    num_t   = (s - 2*m**2)**2 + (u - 2*m**2)**2
    num_s   = (t - 2*m**2)**2 + (u - 2*m**2)**2
    num_int = 2*(u - 2*m**2)**2
    
    return 2*e_qed**4 * (num_t/(t*t) + num_s/(s*s) + num_int/(s*t))

@numba.njit
def bhabha_amplitude2_t(s: float, t: np.ndarray, m=5.11e-4):
    """ (t-channel term only -- for diagnostics) """
    u     = 4*m**2 - s - t
    num_t = (s - 2*m**2)**2 + (u - 2*m**2)**2
    
    return 2*e_qed**4 * (num_t/(t*t))

@numba.njit
def bhabha_amplitude2_s(s: float, t: np.ndarray, m=5.11e-4):
    """ (s-channel term only -- for diagnostics) """
    u     = 4*m**2 - s - t
    num_s = (t - 2*m**2)**2 + (u - 2*m**2)**2
    
    return 2*e_qed**4 * (num_s/(s*s))

@numba.njit
def bhabha_amplitude2_int(s: float, t: np.ndarray, m=5.11e-4):
    """ (interference term only -- for diagnostics) """
    u       = 4*m**2 - s - t
    num_int = 2*(u - 2*m**2)**2
    
    return 2*e_qed**4 * (num_int/(t*s))

# ------------------------------------------------------------------

@numba.njit
def moeller_amplitude2(s: float, t: np.ndarray, m=5.11e-4):
    """
    Spin-averaged LO-QED Moeller scattering amplitude |M(s,t)|^2 for e-e- -> e-e-
    """
    u = 4*m**2 - s - t
    
    num_t = (s - 2*m**2)**2 + (u - 2*m**2)**2
    num_u = (t - 2*m**2)**2 + (s - 2*m**2)**2
    num_int = 2*(s - 2*m**2)**2
    
    S = 0.5 # Cross-section symmetry factor (identical final state particles)
    
    return S * 2*e_qed**4 * (num_t/(t*t) + num_u/(u*u) + num_int/(t*u))

@numba.njit
def moeller_amplitude2_t(s: float, t: np.ndarray, m=5.11e-4):
    """
    (t-channel term only -- for diagnostics)
    """
    u = 4*m**2 - s - t
    num_t = (s - 2*m**2)**2 + (u - 2*m**2)**2
    
    S = 0.5 # Cross-section symmetry factor (identical final state particles)
    
    return S * 2*e_qed**4 * (num_t/(t*t))

@numba.njit
def moeller_amplitude2_u(s: float, t: np.ndarray, m=5.11e-4):
    """
    (u-channel term only -- for diagnostics)
    """
    u = 4*m**2 - s - t
    num_u = (t - 2*m**2)**2 + (s - 2*m**2)**2
    
    S = 0.5 # Cross-section symmetry factor (identical final state particles)
    
    return S * 2*e_qed**4 * (num_u/(u*u))

@numba.njit
def moeller_amplitude2_int(s: float, t: np.ndarray, m=5.11e-4):
    """
    (interference term only -- for diagnostics)
    """
    u = 4*m**2 - s - t
    num_int = 2*(s - 2*m**2)**2
    
    S = 0.5 # Cross-section symmetry factor (identical final state particles)
    
    return S * 2*e_qed**4 * (num_int/(t*u))

# ------------------------------------------------------------------

@numba.njit
def alpha_eff(Q2):
    # Effective running coupling (QED 1-loop)
    Q2 = max(Q2, m_e**2 + 1e-10)
    return alpha_qed / (1 - alpha_qed / (3*np.pi) * np.log(Q2 / m_e**2))

@numba.njit
def e_qed_eff(Q2):
    return np.sqrt(4.0*np.pi*alpha_eff(Q2))

@numba.njit
def dsigma_dt(s: float, t: np.ndarray, theory: Mode):
    """ For analytic comparisons """
    return kinematics(s=s) * amplitude2_wrapper(s=s, t=t, theory=theory)

@numba.njit
def dsigma_dcostheta(s: float, costheta: np.ndarray, theory: Mode):
    """ For analytic comparisons """
    t   = costheta_to_t(costheta=costheta, s=s)
    jac = 2 * (s/4 - m_e**2) # Jacobian dt / dcos(theta)
    
    return dsigma_dt(s=s, t=t, theory=theory) * jac

@numba.njit
def mc_cross_section(wsum: float, w2sum: float, N: int):
    """ Compute integrated cross section and its uncertainty with MC """
    
    I        = wsum / N
    variance = (w2sum - (wsum**2) / N) / (N * (N - 1))
    sigma    = np.sqrt(variance)
    return I, sigma

@numba.njit
def generate_event(s: float, t_min: float, t_max: float, theory: Mode):
    """
    Generate one event with von Neumann Acceptance-Rejection.

    No importance sampling.
    
    Make sure s is a float !
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
    """
    Main event generator routine of 2 -> 2 scattering
    
    p1 + p2 -> p3 + p4
    
    Returns only the final state 3-momentum (px,py,pz) of the p3 leg.
    The rest can be obtained via 4-momentum conservation.

    Make sure s is a float !
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
