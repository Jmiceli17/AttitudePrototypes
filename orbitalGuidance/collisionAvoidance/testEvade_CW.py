import numpy as np
from scipy.optimize import minimize

def cw_phi(n, tau):
    """Returns the STM blocks for the CW equations for time interval tau and mean motion n."""
    # Position-to-position
    phi_rr = np.array([
        [4-3*np.cos(n*tau), 0, 0],
        [6*(np.sin(n*tau)-n*tau), 1, 0],
        [0, 0, np.cos(n*tau)]
    ])
    # Position-to-velocity
    phi_rv = np.array([
        [np.sin(n*tau)/n, 2*(1-np.cos(n*tau))/n, 0],
        [-2*(1-np.cos(n*tau))/n, (4*np.sin(n*tau)-3*n*tau)/n, 0],
        [0, 0, np.sin(n*tau)/n]
    ])
    # Velocity-to-position
    phi_vr = np.array([
        [3*n*np.sin(n*tau), 0, 0],
        [6*n*(np.cos(n*tau)-1), 0, 0],
        [0, 0, -n*np.sin(n*tau)]
    ])
    # Velocity-to-velocity
    phi_vv = np.array([
        [np.cos(n*tau), 2*np.sin(n*tau), 0],
        [-2*np.sin(n*tau), 4*np.cos(n*tau)-3, 0],
        [0, 0, np.cos(n*tau)]
    ])
    return phi_rr, phi_rv, phi_vr, phi_vv

def propagate_triple_impulse(r0, v0, t0, t1, t2, t3, t_target, dv1, dv2, dv3, n):
    """
    Propagate the relative state from t0 to t_target, applying up to three impulses at t1, t2, t3.
    Only impulses occurring at or before t_target are applied.
    
    Args:
        r0, v0: Initial relative position and velocity (Hill frame, km and km/s)
        t0: Initial time (s)
        t1, t2, t3: Impulse times (s)
        t_target: Target time to propagate to (s)
        dv1, dv2, dv3: Delta-v vectors for impulses (km/s)
        n: Mean motion (rad/s)
    Returns:
        r_target: Relative position at t_target (km)
    """
    # First segment: t0 to t1
    tau1 = t1 - t0
    phi_rr, phi_rv, _, _ = cw_phi(n, tau1)
    r = phi_rr @ r0 + phi_rv @ v0
    v = v0
    v += dv1

    if t_target <= t2:
        tau = t_target - t1
        phi_rr, phi_rv, _, _ = cw_phi(n, tau)
        return phi_rr @ r + phi_rv @ v

    # Second segment: t1 to t2
    tau2 = t2 - t1
    phi_rr, phi_rv, _, _ = cw_phi(n, tau2)
    r = phi_rr @ r + phi_rv @ v
    v = v
    v += dv2

    if t_target <= t3:
        tau = t_target - t2
        phi_rr, phi_rv, _, _ = cw_phi(n, tau)
        return phi_rr @ r + phi_rv @ v

    # Third segment: t2 to t3
    tau3 = t3 - t2
    phi_rr, phi_rv, _, _ = cw_phi(n, tau3)
    r = phi_rr @ r + phi_rv @ v
    v = v
    v += dv3

    # Final segment: t3 to t_target (if t_target > t3)
    tau = t_target - t3
    phi_rr, phi_rv, _, _ = cw_phi(n, tau)
    return phi_rr @ r + phi_rv @ v


def optimize_maneuvers_cw(
    r0, v0, t0, t1, t2_guess, t3_guess, t_ca, d_miss, n,
    dv1_guess, dv2_guess, dv3_guess
):
    def cost(x):
        dv1 = x[0:3]
        dv2 = x[3:6]
        dv3 = x[6:9]
        t2 = x[9]
        t3 = x[10]
        return np.linalg.norm(dv1) + np.linalg.norm(dv2) + np.linalg.norm(dv3)

    def constraint(x):
        dv1 = x[0:3]
        dv2 = x[3:6]
        dv3 = x[6:9]
        t2 = x[9]
        t3 = x[10]
        r_ca = propagate_triple_impulse(r0, v0, t0, t1, t2, t3, t_ca, dv1, dv2, dv3, n)
        # Threat at origin in Hill frame
        return np.linalg.norm(r_ca) - d_miss

    x0 = np.concatenate((dv1_guess, dv2_guess, dv3_guess, [t2_guess, t3_guess]))
    bounds = [(-1, 1)]*9 + [(t1+1, t_ca-2), (t1+2, t_ca-1)]
    cons = [
        {'type': 'ineq', 'fun': constraint},
        {'type': 'ineq', 'fun': lambda x: x[9] - t1 - 1},
        {'type': 'ineq', 'fun': lambda x: x[10] - x[9] - 1},
        {'type': 'ineq', 'fun': lambda x: t_ca - x[10] - 1},
    ]
    result = minimize(cost, x0, method='SLSQP', bounds=bounds, constraints=cons)
    if not result.success:
        print("Optimization failed:", result.message)
        return None
    dv1 = result.x[0:3]
    dv2 = result.x[3:6]
    dv3 = result.x[6:9]
    t2 = result.x[9]
    t3 = result.x[10]
    return dv1, dv2, dv3, t1, t2, t3

# Example usage
if __name__ == "__main__":
    n = 0.001027  # mean motion of chief orbit [rad/s] for ~7000 km orbit
    r0 = [0., 0., 0.5]  # initial relative position in Hill frame [km]
    v0 = [0., 0., 0.]  # initial relative velocity in Hill frame [km/s]
    t0 = 0
    t1 = 600
    t2_guess = 700
    t3_guess = 3000
    t_ca = 2500
    d_miss = 1.0
    dv1_guess = [0, 0.01, 0]
    dv2_guess = [0, 0.01, 0]
    dv3_guess = [0, 0.01, 0]

    result = optimize_maneuvers_cw(
        r0, v0, t0, t1, t2_guess, t3_guess, t_ca, d_miss, n,
        dv1_guess, dv2_guess, dv3_guess
    )
    if result:
        dv1, dv2, dv3, t1, t2, t3 = result
        print("\nOptimized Maneuvers (Hill frame):")
        print(f"1st maneuver at t={t1:.1f}s: Δv = {dv1} km/s")
        print(f"2nd maneuver at t={t2:.1f}s: Δv = {dv2} km/s")
        print(f"3rd maneuver at t={t3:.1f}s: Δv = {dv3} km/s")


    # Propagate your satellite to t_CA using the STM-based method
    sat_pos_ca = propagate_triple_impulse(r0, v0, t0, t1, t2, t3, t_ca, dv1, dv2, dv3, n)

    # Compute relative position at t_CA
    relative_pos = sat_pos_ca 

    print("Relative position at t_CA:", relative_pos)
    print("Miss distance at t_CA:", np.linalg.norm(relative_pos))