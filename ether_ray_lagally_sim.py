"""Numerical solver for the Eikonal ray ODE and Lagally sink-flow visualization.

Solves the ray trajectory equation in flat 3D Euclidean space:
    d/dl (n(r) * dr/dl) = grad(n(r))
using the Hamiltonian-like first-order system (r, p = n * dr/dl), and verifies
the Lagally hydrodynamic force theorem + visualizes the sink velocity field.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


# Dimensionless gravitational scale r_g = G*M/c^2 (set r_g = 1 in geometric units)
def refractive_index_and_grad(
    pos: np.ndarray, r_g: float = 1.0, model: str = "nonlinear"
) -> tuple[float, np.ndarray]:
    """Return n(r) and grad(n)(r) for a spherical mass at the origin.

    Models:
        - "nonlinear": n(r) = (1 - r_g / r)^(-2) from Eq. (30) (valid for r > r_g)
        - "weak":      n(r) = 1 + 2 * r_g / r
    """
    r = float(np.linalg.norm(pos))
    if r <= 0.0:
        raise ValueError("Ray hit the origin singularity r = 0.")

    if model == "nonlinear":
        if r <= r_g:
            raise ValueError(f"Ray entered singular region r={r:.4g} <= r_g={r_g:.4g}.")
        factor = 1.0 - r_g / r
        n_val = factor ** (-2)
        # dn/dr = -2 * r_g / (r^2 * (1 - r_g/r)^3)
        dn_dr = -2.0 * r_g / (r**2 * factor**3)
    elif model == "weak":
        n_val = 1.0 + 2.0 * r_g / r
        dn_dr = -2.0 * r_g / (r**2)
    else:
        raise ValueError(f"Unknown model: {model}")

    grad_n = (dn_dr / r) * pos
    return n_val, grad_n


def solve_ray(
    impact_b: float,
    x_extent: float = 150.0,
    r_g: float = 1.0,
    model: str = "nonlinear",
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> dict:
    """Integrate d/dl(n dr/dl) = grad(n) in dimensionless units r_tilde = r / r_g."""
    b_dim = impact_b / r_g
    x_dim = x_extent / r_g

    r0 = np.array([-x_dim, b_dim], dtype=float)
    n0, _ = refractive_index_and_grad(r0, r_g=1.0, model=model)
    # Initial tangent along +x: p = n * dr/dl
    p0 = np.array([n0, 0.0], dtype=float)
    y0 = np.concatenate([r0, p0])

    def rhs(_l: float, state: np.ndarray) -> np.ndarray:
        pos = state[:2]
        mom = state[2:]
        n_val, grad_n = refractive_index_and_grad(pos, r_g=1.0, model=model)
        dr_dl = mom / n_val
        dp_dl = grad_n
        return np.concatenate([dr_dl, dp_dl])

    def reach_right_boundary(_l: float, state: np.ndarray) -> float:
        return state[0] - x_dim

    reach_right_boundary.terminal = True
    reach_right_boundary.direction = 1

    def capture_horizon(_l: float, state: np.ndarray) -> float:
        return np.linalg.norm(state[:2]) - 1.05

    capture_horizon.terminal = True
    capture_horizon.direction = -1

    l_max = 2.5 * x_dim + 30.0
    sol = solve_ivp(
        rhs,
        (0.0, l_max),
        y0,
        method="DOP853",
        events=[reach_right_boundary, capture_horizon],
        rtol=rtol,
        atol=atol,
        max_step=max(0.25, 0.05 * b_dim),
    )
    if not sol.success:
        raise RuntimeError(f"ODE solver failed for b={impact_b}: {sol.message}")

    captured = len(sol.t_events[1]) > 0
    traj = sol.y[:2].T * r_g
    moms = sol.y[2:].T

    # Check exact ray-invariant |p| / n(r) == 1
    inv_err = max(
        abs(np.linalg.norm(m) / refractive_index_and_grad(p, r_g=r_g, model=model)[0] - 1.0)
        for p, m in zip(traj, moms)
    )

    p_in = moms[0]
    p_out = moms[-1]
    theta_in = np.arctan2(p_in[1], p_in[0])
    theta_out = np.arctan2(p_out[1], p_out[0])
    deflection_rad = abs(theta_out - theta_in)

    # Weak-field analytical formula (finite-x correction approaches 4*r_g/b as x_extent -> inf)
    weak_analytic_inf = 4.0 * r_g / impact_b
    weak_analytic_finite = weak_analytic_inf * (x_extent / np.sqrt(x_extent**2 + impact_b**2))

    return {
        "b": impact_b,
        "traj": traj,
        "captured": captured,
        "deflection_rad": deflection_rad,
        "weak_analytic_inf": weak_analytic_inf,
        "weak_analytic_finite": weak_analytic_finite,
        "invariant_max_err": inv_err,
    }


def verify_lagally_surface_integral(
    r_pos: np.ndarray,
    Q_M: float = 1.0,
    Q_m: float = 0.05,
    rho_a0: float = 1.0,
    eps: float = 1e-3,
    n_theta: int = 80,
    n_phi: int = 160,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Numerically verify Eq. (25): F_g = -oint P n dS - oint rho v (v.n) dS = rho_a0 * Q_m * v_M."""
    # Midpoint spherical quadrature on a small control sphere of radius eps centered at r_pos
    d_th = np.pi / n_theta
    d_ph = 2.0 * np.pi / n_phi
    th = (np.arange(n_theta) + 0.5) * d_th
    ph = (np.arange(n_phi) + 0.5) * d_ph
    TH, PH = np.meshgrid(th, ph, indexing="ij")

    n_vec = np.stack(
        [np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)], axis=-1
    )
    dS = (eps**2) * np.sin(TH) * d_th * d_ph

    # Field points on control sphere around r_pos: r_field = r_pos + eps * n_vec
    r_field = r_pos[None, None, :] + eps * n_vec
    r_mag = np.linalg.norm(r_field, axis=-1, keepdims=True)
    v_M_field = -(Q_M / (4.0 * np.pi * r_mag**3)) * r_field
    v_sink = -(Q_m / (4.0 * np.pi * eps**2)) * n_vec
    v_tot = v_M_field + v_sink

    v2 = np.sum(v_tot**2, axis=-1, keepdims=True)
    # Bernoulli pressure P = P0 - 0.5 * rho_a0 * |v|^2 (constant P0 integrates to 0 on closed surface)
    P_dyn = -0.5 * rho_a0 * v2
    v_dot_n = np.sum(v_tot * n_vec, axis=-1, keepdims=True)

    integrand = -P_dyn * n_vec - rho_a0 * v_tot * v_dot_n
    F_numeric = np.sum(integrand * dS[..., None], axis=(0, 1))

    r_center = np.linalg.norm(r_pos)
    v_M_center = -(Q_M / (4.0 * np.pi * r_center**3)) * r_pos
    F_analytic = rho_a0 * Q_m * v_M_center
    rel_err = float(np.linalg.norm(F_numeric - F_analytic) / np.linalg.norm(F_analytic))
    return F_numeric, F_analytic, rel_err


def make_plots(
    ray_results: list[dict],
    r_g: float = 1.0,
    Q_M: float = 4.0 * np.pi,
    rho_a0: float = 1.0,
    out_path: str = "ether_simulation.png",
) -> None:
    """Create a 2-panel figure: (1) bent ray trajectories & n(r), (2) Lagally sink flow."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.8), dpi=160)

    # --- Panel 1: Ray trajectories from direct Eikonal ODE integration ---
    lim_x = 30.0
    y_min, y_max = -22.0, 24.0
    gx = np.linspace(-lim_x, lim_x, 360)
    gy = np.linspace(y_min, y_max, 280)
    GX, GY = np.meshgrid(gx, gy)
    GR = np.hypot(GX, GY)
    # Mask interior singular zone for clean color mapping
    n_excess = np.where(GR > 2.2 * r_g, (1.0 - r_g / GR) ** (-2) - 1.0, np.nan)

    im1 = ax1.pcolormesh(
        GX,
        GY,
        n_excess,
        shading="auto",
        cmap="magma_r",
        norm=LogNorm(vmin=1e-2, vmax=1.5),
        alpha=0.65,
    )
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.03)
    cbar1.set_label(r"$n(r) - 1 = (1 - r_g/r)^{-2} - 1$")

    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(ray_results)))
    for res, col in zip(ray_results, colors):
        tr = res["traj"]
        if res["captured"]:
            label = rf"$b={res['b']:.1f}\,r_g$ (bị hút, $<b_{{\rm crit}}$)"
            ax1.plot(tr[:, 0], tr[:, 1], color="crimson", lw=1.9, ls="--", label=label)
        else:
            deg = np.degrees(res["deflection_rad"])
            label = rf"$b={res['b']:.1f}\,r_g,\ |\Delta\varphi|={deg:.2f}^\circ$"
            ax1.plot(tr[:, 0], tr[:, 1], color=col, lw=1.9, label=label)
        ax1.axhline(res["b"], color=col, ls=":", lw=0.7, alpha=0.4)

    # Draw photon sphere (r = 3 r_g) and singular horizon (r = r_g)
    photon_ring = plt.Circle((0, 0), 3.0 * r_g, color="orange", fill=False, ls="--", lw=1.2, zorder=5, label=r"Quả cầu photon $r=3\,r_g$")
    mass_circle = plt.Circle((0, 0), 1.05 * r_g, color="black", zorder=6)
    ax1.add_patch(photon_ring)
    ax1.add_patch(mass_circle)
    ax1.set_xlim(-lim_x, lim_x)
    ax1.set_ylim(y_min, y_max)
    ax1.set_aspect("equal")
    ax1.set_xlabel(r"$x / r_g$")
    ax1.set_ylabel(r"$y / r_g$")
    ax1.set_title(r"Quỹ đạo tia sáng từ ODE $\frac{d}{dl}\left(n\frac{d\mathbf{r}}{dl}\right) = \nabla n$")
    ax1.legend(loc="lower left", fontsize=7.8, framealpha=0.92)
    ax1.grid(alpha=0.25, ls="--")

    # --- Panel 2: Lagally sink velocity field & Bernoulli pressure drop ---
    lim = 10.0
    vx_g = np.linspace(-lim, lim, 240)
    vy_g = np.linspace(-lim, lim, 240)
    VXG, VYG = np.meshgrid(vx_g, vy_g)
    RR = np.hypot(VXG, VYG)
    r_min = 0.85
    RR_safe = np.maximum(RR, r_min)
    U = -(Q_M / (4.0 * np.pi * RR_safe**3)) * VXG
    V = -(Q_M / (4.0 * np.pi * RR_safe**3)) * VYG
    speed = np.hypot(U, V)
    # Bernoulli pressure depression |Delta P| = 0.5 * rho_a0 * |v|^2
    dP_mag = 0.5 * rho_a0 * speed**2
    dP_plot = np.where(RR >= r_min, dP_mag, np.nan)

    im2 = ax2.pcolormesh(
        VXG,
        VYG,
        dP_plot,
        shading="auto",
        cmap="Blues",
        norm=LogNorm(vmin=1e-4, vmax=1.0),
        alpha=0.75,
    )
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.03)
    cbar2.set_label(r"Độ sụt áp suất Bernoulli $|\Delta P(r)| = \frac{1}{2}\rho_{a0}|\mathbf{v}_M|^2$")

    ax2.streamplot(
        VXG,
        VYG,
        U,
        V,
        color="#1f4e78",
        density=1.35,
        linewidth=1.0,
        arrowsize=1.1,
    )

    # Overlay sparse normalized-direction quiver arrows outside core
    step = 22
    mask_q = RR[::step, ::step] > 1.5
    xq = VXG[::step, ::step][mask_q]
    yq = VYG[::step, ::step][mask_q]
    uq = U[::step, ::step][mask_q] / speed[::step, ::step][mask_q]
    vq = V[::step, ::step][mask_q] / speed[::step, ::step][mask_q]
    ax2.quiver(xq, yq, uq, vq, color="#0d2b45", alpha=0.55, scale=28, width=0.003)

    # Primary sink M and test sink m with Lagally force vector
    ax2.scatter([0], [0], s=220, color="crimson", edgecolor="black", zorder=6, label=r"Bể hút $M$ ($Q_M$)")
    test_pos = np.array([5.5, 3.2])
    r_tp = np.linalg.norm(test_pos)
    f_dir = -test_pos / r_tp
    ax2.scatter(
        [test_pos[0]],
        [test_pos[1]],
        s=90,
        color="gold",
        edgecolor="black",
        zorder=6,
        label=r"Bể hút thử $m$ ($Q_m$)",
    )
    ax2.annotate(
        "",
        xy=test_pos + 2.4 * f_dir,
        xytext=test_pos,
        arrowprops=dict(arrowstyle="-|>", color="crimson", lw=2.4, mutation_scale=16),
        zorder=7,
    )
    ax2.text(
        1.1,
        5.0,
        r"$\mathbf{F}_g = \rho_{a0} Q_m \mathbf{v}_M = -\frac{GMm}{r^2}\hat{\mathbf{r}}$",
        color="darkred",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="crimson", alpha=0.92),
        zorder=8,
    )

    ax2.set_xlim(-lim, lim)
    ax2.set_ylim(-lim, lim)
    ax2.set_aspect("equal")
    ax2.set_xlabel(r"$x$")
    ax2.set_ylabel(r"$y$")
    ax2.set_title(r"Trường vận tốc bể hút thế $\mathbf{v}_M = -\frac{Q_M}{4\pi r^2}\hat{\mathbf{r}}$ & Lực Lagally")
    ax2.legend(loc="lower left", fontsize=8.5, framealpha=0.9)
    ax2.grid(alpha=0.25, ls="--")

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    r_g = 1.0
    print("=== 1. KIỂM TRA ODE QUỸ ĐẠO TIA SÁNG (n(r) = (1 - r_g/r)^(-2)) ===")
    print(f"{'b/r_g':>8} | {'ODE |Δφ| (rad)':>15} | {'4r_g/b (rad)':>14} | {'Sai khác tương đối':>18} | {'Max ||p|/n - 1|':>16}")
    print("-" * 82)

    # Strong-to-medium rays for visualization (b_crit = 6.75 r_g) + weak-field validation rays
    vis_bs = [6.2, 7.2, 9.0, 13.0, 18.0]
    check_bs = [6.2, 7.2, 9.0, 13.0, 18.0, 50.0, 200.0]

    vis_results = []
    for b in check_bs:
        res = solve_ray(impact_b=b, x_extent=max(200.0, 25.0 * b), r_g=r_g, model="nonlinear")
        if b in vis_bs:
            vis_results.append(res)
        if res["captured"]:
            print(
                f"{b:8.1f} | {'CAPTURED':>15} | {res['weak_analytic_inf']:14.8e} | "
                f"{'N/A (b < 6.75)':>18} | {res['invariant_max_err']:16.3e}"
            )
        else:
            rel_diff = abs(res["deflection_rad"] - res["weak_analytic_finite"]) / res["weak_analytic_finite"]
            print(
                f"{b:8.1f} | {res['deflection_rad']:15.8e} | {res['weak_analytic_inf']:14.8e} | "
                f"{rel_diff:18.3e} | {res['invariant_max_err']:16.3e}"
            )

    # Real Solar limb benchmark (weak-field regime: r_g = GM_sun/c^2 ~ 1476.625 m, R_sun ~ 6.957e8 m)
    r_g_sun = 1476.62504
    r_sun = 6.957e8
    sun_res = solve_ray(impact_b=r_sun, x_extent=300.0 * r_sun, r_g=r_g_sun, model="nonlinear")
    arcsec_num = np.degrees(sun_res["deflection_rad"]) * 3600.0
    arcsec_ana = np.degrees(sun_res["weak_analytic_inf"]) * 3600.0
    print(
        f"\nBenchmark Mặt Trời (b = R_sun): ODE = {arcsec_num:.6f} arcsec, "
        f"Giải tích 4GM/(Rc^2) = {arcsec_ana:.6f} arcsec"
    )

    print("\n=== 2. KIỂM TRA TÍCH PHÂN MẶT CẦU ĐỊNH LÝ LAGALLY (Eq. 25) ===")
    r_test = np.array([5.5, 3.2, 0.0])
    F_num, F_ana, lag_err = verify_lagally_surface_integral(r_test, Q_M=4.0 * np.pi, Q_m=0.05, rho_a0=1.0)
    print(f"F_numeric (tích phân mặt cầu): {F_num}")
    print(f"F_analytic (rho_a0*Q_m*v_M)  : {F_ana}")
    print(f"Sai số tương đối             : {lag_err:.3e}")

    out_png = "ether_simulation.png"
    make_plots(vis_results, r_g=r_g, out_path=out_png)
    print(f"\nĐã xuất đồ thị trực quan hóa tại: {out_png}")


if __name__ == "__main__":
    main()
