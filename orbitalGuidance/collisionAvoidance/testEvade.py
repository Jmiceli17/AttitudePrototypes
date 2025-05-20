import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# Constants
MU = 398600441.8  # Earth's gravitational parameter, m^3/s^2

# Orbital propagation (two-body)
def propagate_orbit(r0, v0, t0, t1):
    def two_body(t, y):
        r = y[:3]
        v = y[3:]
        norm_r = np.linalg.norm(r)
        a = -MU * r / norm_r**3
        return np.concatenate((v, a))
    y0 = np.concatenate((r0, v0))
    sol = solve_ivp(two_body, [t0, t1], y0, rtol=1e-9, atol=1e-9)
    return sol.y[:,-1][:3], sol.y[:,-1][3:]

# Main optimization function
def optimize_maneuvers(
    r0, v0, t0,
    threat_pos_ca, t_ca,
    d_miss,
    t1,  # fixed time for first maneuver
    t2_guess, t3_guess,
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
        # First segment: t0 to t1
        r, v = propagate_orbit(r0, v0, t0, t1)
        # First impulse
        v += dv1
        # Second segment: t1 to t2
        r, v = propagate_orbit(r, v, t1, t2)
        # Second impulse
        v += dv2
        # Third segment: t2 to t3
        r, v = propagate_orbit(r, v, t2, t3)
        # Third impulse
        v += dv3
        # Final segment: t3 to t_ca
        r, v = propagate_orbit(r, v, t3, t_ca)
        # Miss distance constraint (must be >= d_miss)
        return np.linalg.norm(r - threat_pos_ca) - d_miss

    # Initial guess vector
    x0 = np.concatenate((dv1_guess, dv2_guess, dv3_guess, [t2_guess, t3_guess]))

    # Constraints: miss distance at CA, and time ordering
    cons = [
        {'type': 'ineq', 'fun': constraint},  # miss distance
        {'type': 'ineq', 'fun': lambda x: x[9] - t1 - 1},  # t2 > t1
        {'type': 'ineq', 'fun': lambda x: x[10] - x[9] - 1},  # t3 > t2
        {'type': 'ineq', 'fun': lambda x: t_ca - x[10] - 1},  # t_ca > t3
    ]

    # Bounds for the optimizer:
    # - First 9 values: delta-v components for the 3 maneuvers (x, y, z for each), limited to [-1, 1] km/s.
    # - Next 2 values: times of the 2nd and 3rd maneuvers.
    #     - 2nd maneuver time: must be after the 1st (t1+1) and before the 3rd/CA (t_ca-2).
    #     - 3rd maneuver time: must be after the 2nd (t1+2) and before closest approach (t_ca-1).
    # This ensures the optimizer only explores feasible values for all variables.
    bounds = [(-1, 1)]*9 + [(t1+1, t_ca-2), (t1+2, t_ca-1)]


    # Run optimizer
    result = minimize(
        cost, x0, method='SLSQP', constraints=cons, bounds=bounds,
        options={'ftol': 1e-5, 'disp': True, 'maxiter': 100}
    )

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
    # Example initial satellite state (circular LEO) [ECI]
    r0 = np.array([7000000, 0, 0])  # m
    v0 = np.array([0, 7546.0, 0])  # m/s
    t0 = 0  # seconds

    # Threat at closest approach [ECI]
    threat_pos_ca = np.array([7000000, 500, 0])  # 500m
    t_ca = 3600  # 1 hour after t0

    d_miss = 1000  # 1 km minimum miss distance

    # Maneuver times (first is fixed)
    t1 = 600  # 10 minutes after t0 (fixed)
    t2_guess = 1800  # 30 minutes
    t3_guess = 3000  # 50 minutes

    # Initial guess for delta-v's (km/s)
    dv1_guess = np.array([0.0, 0.01, 0.0])
    dv2_guess = np.array([0.0, 0.01, 0.0])
    dv3_guess = np.array([0.0, 0.01, 0.0])

    result = optimize_maneuvers(
        r0, v0, t0,
        threat_pos_ca, t_ca,
        d_miss,
        t1,
        t2_guess, t3_guess,
        dv1_guess, dv2_guess, dv3_guess
    )

    if result:
        dv1, dv2, dv3, t1, t2, t3 = result
        print("\nOptimized Maneuvers:")
        print(f"1st maneuver at t={t1:.1f}s: Δv = {dv1} [km/s]")
        print(f"2nd maneuver at t={t2:.1f}s: Δv = {dv2} [km/s]")
        print(f"3rd maneuver at t={t3:.1f}s: Δv = {dv3} [km/s]")
