# -*- coding: utf-8 -*-
"""
Created on Tue May 12 12:56:41 2026

@author: student
"""

# -*- coding: utf-8 -*-
"""
1D TRANSIENT RADIAL HEAT CONDUCTION IN A HOLLOW CYLINDER

Coolant model:
    Tw(t) = T_cold + (T_hot - T_cold) exp(-t/tau)

Saved files:
    r_grid.npy
    time_vector.npy
    T_hist.npy
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Literal, Tuple

# ============================================================
# USER SETTINGS
# ============================================================

# Geometry
a = 1.994       # [m] inner radius
b = 2.2015      # [m] outer radius

# Material properties
lam = 45.0      # [W/(m*K)] #thermal conductivity, heat conduction inside the material
rho = 7800.0    # [kg/m^3]
cp = 500.0      # [J/(kg*K)] #specipic heat capacity, how much heat energy is neede to raise the temperature of the material
kappa = lam / (rho * cp) # thermal diffusivity, how fast heat spreads through the material.

# Initial condition
T0 = 290.0      # [°C]

# Coolant transient
T_hot = 290.0
T_cold = 40.0
tau = 2.0

# Heat transfer coefficients
h_values = [5000.0, 10000.0, 20000.0] #heat transfer coefficient, how fast heat transfer from fluid to surface, convection

# Outer boundary condition
OuterBC = Literal["adiabatic", "fixed_temperature", "convection"]
outer_bc: OuterBC = "adiabatic"

T_outer_fixed = 290.0
h_outer = 10.0
T_inf_outer = 290.0

# Numerics
N = 101 # does vary much when n=501, 1001.
t_end = 2000.0
dt = 1 
theta = 1 #backward or implicit euler as its unconditionally stable


# Plot snapshots
profile_times = [0, 0.5, 1, 2, 5, 10, 20, 100, 500, 2000]

# ============================================================
# TRIDIAGONAL SOLVER
# ============================================================

def thomas(a_sub, b_diag, c_sup, d_rhs): #fast tridiagonal matrix solver
    n = len(b_diag)

    a_sub = a_sub.astype(float).copy()
    b_diag = b_diag.astype(float).copy()
    c_sup = c_sup.astype(float).copy()
    d_rhs = d_rhs.astype(float).copy()

    for i in range(1, n): # forward elimination
        w = a_sub[i - 1] / b_diag[i - 1]
        b_diag[i] -= w * c_sup[i - 1]
        d_rhs[i] -= w * d_rhs[i - 1]

    x = np.zeros(n)
    x[-1] = d_rhs[-1] / b_diag[-1] # backward substitution

    for i in range(n - 2, -1, -1): 
        x[i] = (d_rhs[i] - c_sup[i] * x[i + 1]) / b_diag[i] # solving the unknown using the already known value

    return x

# ============================================================
# COOLANT TEMPERATURE MODEL
# ============================================================

def Tw_exponential(t, Thot, Tcold, tau):
    if t <= 0.0:
        return Thot
    return Tcold + (Thot - Tcold) * np.exp(-t / tau) #coolant temp changing over time

# ============================================================
# TRANSIENT HEAT CONDUCTION SOLVER
# ============================================================

def solve_cylinder_transient(
    a, b, lam, kappa, h_inner, Tw, outer_bc,
    dt, t_end, N, T0, theta,
    h_outer=0.0,
    T_inf_outer=None,
    T_outer_fixed=None
):

    r = np.linspace(a, b, N) # specifying the node for space
    dr = r[1] - r[0]

    times = np.arange(0.0, t_end + dt, dt) # specifying the timestep

    T = np.full(N, T0)
    T_hist = np.zeros((len(times), N)) #creates a matrix no of points in the simulation X N
    T_hist[0, :] = T.copy()

    for nstep in range(1, len(times)):
        t_next = times[nstep]

        L_old = np.zeros_like(T) # store the heat conduction operator , hows the shape of the temp field

        for i in range(1, N - 1):
            ri = r[i]

            d2 = (T[i + 1] - 2.0 * T[i] + T[i - 1]) / dr**2 #curvature of the temp through the wall, whether the heat is accumulating, spreading or flattening
            d1 = (T[i + 1] - T[i - 1]) / (2.0 * dr) #temp gradient, how fast temp change with position

            L_old[i] = d2 + (1.0 / ri) * d1 # how heat is spreading through the wall at current timestep

        A = np.zeros(N - 1)
        B = np.zeros(N)
        C = np.zeros(N - 1)
        rhs = np.zeros(N)

        # Inner convection boundary:
        # lambda dT/dr = h(T - Tw) # convection between the coolant and the wall
        B[0] = (-lam / dr - h_inner) # coefficient of To
        C[0] = (lam / dr) # coefficient of T1
        rhs[0] = -h_inner * Tw(t_next) # right hand side

        # Interior nodes
        for i in range(1, N - 1):
            ri = r[i]

            aL = (1.0 / dr**2) - (1.0 / (2.0 * ri * dr)) #cofficient from finite difference form
            bL = -2.0 / dr**2
            cL = (1.0 / dr**2) + (1.0 / (2.0 * ri * dr))

            A[i - 1] = -dt * kappa * theta * aL
            B[i] = 1.0 - dt * kappa * theta * bL
            C[i] = -dt * kappa * theta * cL

            rhs[i] = T[i] + dt * kappa * (1.0 - theta) * L_old[i]

        # Outer boundary
        if outer_bc == "adiabatic":
            A[N - 2] = -1.0 #coeff from Finite Difference Form
            B[N - 1] = 1.0
            rhs[N - 1] = 0.0

        elif outer_bc == "fixed_temperature":
            B[N - 1] = 1.0
            rhs[N - 1] = float(T_outer_fixed)

        elif outer_bc == "convection":
            A[N - 2] = (lam / dr)
            B[N - 1] = (-lam / dr - h_outer)
            rhs[N - 1] = -h_outer * T_inf_outer(t_next)

        else:
            raise ValueError(f"Unknown outer_bc = {outer_bc}")

        T = thomas(A, B, C, rhs)
        T_hist[nstep, :] = T

    return r, times, T_hist

# ============================================================
# RUN AND PLOT
# ============================================================

def run_demo():
    plt.figure(figsize=(8, 5))

    for h in h_values:
        Tw = lambda t: Tw_exponential(t, T_hot, T_cold, tau)

        r, times, T_hist = solve_cylinder_transient(
            a=a,
            b=b,
            lam=lam,
            kappa=kappa,
            h_inner=h,
            Tw=Tw,
            outer_bc=outer_bc,
            dt=dt,
            t_end=t_end,
            N=N,
            T0=T0,
            theta=theta,
            h_outer=h_outer,
            T_inf_outer=(lambda t: T_inf_outer) if outer_bc == "convection" else None,
            T_outer_fixed=T_outer_fixed if outer_bc == "fixed_temperature" else None
        )

        plt.plot(times, T_hist[:, 0], label=f"h={h:.0f} W/m²K")

    plt.xlabel("Time [s]")
    plt.ylabel(r"Inner wall temperature $T(a,t)$ [°C]")
    plt.title(r"Inner wall cooling response, $T_w(t)=40+250e^{-t/1}$")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("inner_wall_temperature_exponential.png", dpi=300)

    # One selected case for temperature profiles
    h0 = h_values[0]
    Tw0 = lambda t: Tw_exponential(t, T_hot, T_cold, tau)

    r, times, T_hist = solve_cylinder_transient(
        a=a,
        b=b,
        lam=lam,
        kappa=kappa,
        h_inner=h0,
        Tw=Tw0,
        outer_bc=outer_bc,
        dt=dt,
        t_end=t_end,
        N=N,
        T0=T0,
        theta=theta,
        h_outer=h_outer,
        T_inf_outer=(lambda t: T_inf_outer) if outer_bc == "convection" else None,
        T_outer_fixed=T_outer_fixed if outer_bc == "fixed_temperature" else None
    )

    x_mm = (r - a) * 1e3

    plt.figure(figsize=(8, 5))

    for t_plot in profile_times:
        idx = int(round(t_plot / dt))
        if 0 <= idx < len(times):
            plt.plot(x_mm, T_hist[idx, :], label=f"t={times[idx]:.1f}s")

    plt.xlabel("Distance from inner surface [mm]")
    plt.ylabel("Temperature [°C]")
    plt.title(f"Temperature profiles, h={h0:.0f} W/m²K")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("temperature_profiles_exponential.png", dpi=300)

    # Save arrays for stress/fracture codes
    np.save("r_grid.npy", r)
    np.save("time_vector.npy", times)
    np.save("T_hist.npy", T_hist)

    print("Saved files:")
    print(" - r_grid.npy")
    print(" - time_vector.npy")
    print(" - T_hist.npy")
    print(" - inner_wall_temperature_exponential.png")
    print(" - temperature_profiles_exponential.png")

    print("\nModel summary:")
    print(f"Tw(t) = {T_cold:.1f} + {T_hot - T_cold:.1f} exp(-t/{tau:.1f})")
    print(f"kappa = {kappa:.6e} m^2/s")
    print(f"dt = {dt:.3f} s")
    print(f"t_end = {t_end:.1f} s")
    print(f"tau = {tau:.3f} s")
    print(f"theta = {theta:.2f}")
    print(f"N = {N}")

    plt.show()

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    run_demo()