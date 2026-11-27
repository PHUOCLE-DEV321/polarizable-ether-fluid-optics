"""1D Numerical Verification of Chapter III Extended Maxwell-van Vlaenderen Equations:
    d_t(E_x) = c^2 * d_x(S) - J_x / eps0        (since curl B = 0 in 1D longitudinal mode)
    d_t(S)   = -d_x(E_x) + rho / eps0           (Eq. 13a: div E + d_t S = rho / eps0)
    B = 0 identically (pure longitudinal electro-scalar pressure wave!)

Verifies:
1) Pure longitudinal S-wave propagates at speed c with B = 0.
2) Exact numerical energy balance Eq. (18a):
       d_t u_tot + d_x S_tot = -J_x * E_x + c^2 * rho * S
   where u_tot = 0.5 * eps0 * E_x^2 + 0.5 / mu0 * S^2 and S_tot = (S / mu0) * E_x.
3) Charge-conserving transverse/standard source (d_t rho + d_x J_x = 0) with S(0)=0
   excites zero S-wave (recovering standard Maxwell limit).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def simulate_longitudinal_s_wave(
    source_conserves_charge: bool = False,
) -> dict:
    """Staggered Yee-like leapfrog solver for 1D Extended Maxwell longitudinal equations."""
    c = 1.0
    eps0 = 1.0
    mu0 = 1.0 / (eps0 * c**2)

    nx = 800
    x_min, x_max = -20.0, 20.0
    dx = (x_max - x_min) / nx
    x = np.linspace(x_min + 0.5 * dx, x_max - 0.5 * dx, nx)
    dt = 0.4 * dx / c  # Stable RK4 CFL
    n_steps = 750

    E = np.zeros(nx, dtype=float)
    S = np.zeros(nx, dtype=float)

    # Localized Gaussian source at x = 0 oscillating over finite pulse window
    sigma = 0.6
    g_x = np.exp(-0.5 * (x / sigma) ** 2)
    dg_dx = -(x / sigma**2) * g_x

    def pulse_env(t: float) -> tuple[float, float]:
        t0, tw = 3.0, 0.8
        arg = (t - t0) / tw
        env = np.exp(-0.5 * arg**2)
        denv_dt = -(arg / tw) * env
        return env, denv_dt

    if source_conserves_charge:
        # Satisfy Gauss's law d_x(E) = rho/eps0 at t = 0 so S(0) = d_t S(0) = 0 identically
        E = -g_x * pulse_env(0.0)[0] / eps0

    def d_dx(arr: np.ndarray) -> np.ndarray:
        a = np.pad(arr, 2, mode="constant")
        return (a[:-4] - 8.0 * a[1:-3] + 8.0 * a[3:-1] - a[4:]) / (12.0 * dx)

    def sources(t_val: float) -> tuple[np.ndarray, np.ndarray]:
        env, denv = pulse_env(t_val)
        if source_conserves_charge:
            return -dg_dx * env, g_x * denv
        return g_x * env, np.zeros_like(x)

    def rhs(t_val: float, e_arr: np.ndarray, s_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rho_val, j_val = sources(t_val)
        return -c**2 * d_dx(s_arr) - j_val / eps0, -d_dx(e_arr) + rho_val / eps0

    times = []
    u_hist = []
    work_cum = 0.0
    work_hist = []
    E_snaps = {}
    S_snaps = {}

    for step in range(n_steps + 1):
        t_cur = step * dt
        u_cur = float(np.sum(0.5 * eps0 * E**2 + 0.5 / mu0 * S**2) * dx)
        times.append(t_cur)
        u_hist.append(u_cur)
        work_hist.append(work_cum)
        if step in (200, 400, 700):
            E_snaps[t_cur] = E.copy()
            S_snaps[t_cur] = S.copy()

        if step == n_steps:
            break

        # RK4 integration for (E, S) and Simpson quadrature for source power
        k1_e, k1_s = rhs(t_cur, E, S)
        k2_e, k2_s = rhs(t_cur + 0.5 * dt, E + 0.5 * dt * k1_e, S + 0.5 * dt * k1_s)
        k3_e, k3_s = rhs(t_cur + 0.5 * dt, E + 0.5 * dt * k2_e, S + 0.5 * dt * k2_s)
        k4_e, k4_s = rhs(t_cur + dt, E + dt * k3_e, S + dt * k3_s)

        E_next = E + (dt / 6.0) * (k1_e + 2.0 * k2_e + 2.0 * k3_e + k4_e)
        S_next = S + (dt / 6.0) * (k1_s + 2.0 * k2_s + 2.0 * k3_s + k4_s)

        rho0, j0 = sources(t_cur)
        rho_h, j_h = sources(t_cur + 0.5 * dt)
        rho1, j1 = sources(t_cur + dt)
        p0 = float(np.sum(-j0 * E + c**2 * rho0 * S) * dx)
        E_h = E + 0.5 * dt * (k1_e + k2_e) * 0.5
        S_h = S + 0.5 * dt * (k1_s + k2_s) * 0.5
        p_h = float(np.sum(-j_h * E_h + c**2 * rho_h * S_h) * dx)
        p1 = float(np.sum(-j1 * E_next + c**2 * rho1 * S_next) * dx)
        work_cum += (dt / 6.0) * (p0 + 4.0 * p_h + p1)

        E, S = E_next, S_next

    times_arr = np.array(times)
    u_arr = np.array(u_hist)
    w_arr = np.array(work_hist)
    # Boundary flux S_tot is ~0 because pulse hasn't reached |x|=20 by t=15
    energy_residual = float(np.max(np.abs(u_arr - w_arr)) / max(np.max(u_arr), 1e-12))

    return {
        "x": x,
        "times": times_arr,
        "u_tot": u_arr,
        "work_cum": w_arr,
        "energy_rel_err": energy_residual,
        "E_snaps": E_snaps,
        "S_snaps": S_snaps,
        "max_S": float(np.max(np.abs(S))),
    }


def make_plot(res_wave: dict, res_maxwell: dict, out_path: str = "longitudinal_S_wave.png") -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.3), dpi=160)

    x = res_wave["x"]
    colors = ["#1f77b4", "#2ca02c", "crimson"]
    for (t_val, s_arr), col in zip(res_wave["S_snaps"].items(), colors):
        ax1.plot(x, s_arr, color=col, lw=2.0, label=rf"$S(x, t={t_val:.1f})$ (lan truyền với $c=1$)")

    ax1.set_xlim(-16.0, 16.0)
    ax1.set_ylim(-0.03, 0.76)
    ax1.set_xlabel(r"Tọa độ $x$")
    ax1.set_ylabel(r"Biên độ trường nén dọc $S(x,t)$")
    ax1.set_title(r"Lan truyền Sóng dọc Tesla $S$ ($\mathbf{B} = \mathbf{0}$)")
    ax1.legend(loc="upper right", fontsize=8.2, framealpha=0.93)
    ax1.grid(alpha=0.25, ls="--")

    # Panel 2: Energy balance verification Eq. (18a)
    t = res_wave["times"]
    ax2.plot(t, res_wave["u_tot"], color="#1f77b4", lw=2.4, label=r"Tổng năng lượng trường $\int u_{\rm total}\,dx$")
    ax2.plot(
        t,
        res_wave["work_cum"],
        color="crimson",
        lw=2.0,
        ls="--",
        label=rf"Công nguồn $\int_0^t \int (-\mathbf{{J}}\cdot\mathbf{{E}} + c^2\rho S)\,dx\,dt'$ (sai lệch {res_wave['energy_rel_err']:.2e})",
    )
    ax2.set_xlim(0.0, t[-1])
    ax2.set_xlabel("Thời gian $t$")
    ax2.set_ylabel("Năng lượng tổng hợp")
    ax2.set_title("Kiểm chứng Định lý Poynting mở rộng Eq. (18)–(18a)")
    ax2.legend(loc="lower right", fontsize=8.0, framealpha=0.93)
    ax2.grid(alpha=0.25, ls="--")

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    res_wave = simulate_longitudinal_s_wave(source_conserves_charge=False)
    res_maxwell = simulate_longitudinal_s_wave(source_conserves_charge=True)

    print("=== KIỂM CHỨNG SỐ CHƯƠNG III: SÓNG DỌC MỞ RỘNG & ĐỊNH LÝ POYNTING ===")
    print(f"1) Nguồn bảo toàn điện tích (d_t rho + div J = 0): max |S| = {res_maxwell['max_S']:.3e} (quy về Maxwell chuẩn)")
    print(f"2) Nguồn bức xạ dọc (d_t rho + div J != 0)       : max |S| = {res_wave['max_S']:.4f}")
    print(f"   Kiểm chứng bảo toàn năng lượng Eq. (18a)      : sai số tương đối = {res_wave['energy_rel_err']:.3e}")

    make_plot(res_wave, res_maxwell, "longitudinal_S_wave.png")
    print("Đã xuất đồ thị tại: longitudinal_S_wave.png")


if __name__ == "__main__":
    main()
