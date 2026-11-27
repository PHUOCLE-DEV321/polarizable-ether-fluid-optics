"""Unified Polarizable-Ether Verification:
1. All 4 Solar System tests from a single Lagrangian L = -m*c^2 * sqrt(e^{-2u} - e^{2u} v^2/c^2)
   and refractive index n(r) = exp(2*r_g/r), u = r_g/r = GM/(r*c^2):
   - Light bending
   - Shapiro time delay (direct integration along bent ODE ray)
   - Gravitational redshift
   - Mercury perihelion precession (exact Binet ODE integration)
2. Strong-field Black Hole Shadow comparison:
   - n_orig(r) = (1 - r_g/r)^(-2) -> b_crit = 6.7500 r_g
   - n_PV(r)   = exp(2*r_g/r)     -> b_crit = 2*e = 5.4366 r_g
   - GR (Schwarzschild)           -> b_crit = 3*sqrt(3) = 5.1962 r_g
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


# Physical constants (IAU / SI)
C = 299_792_458.0
GM_SUN = 1.3271244e20  # m^3 / s^2
R_SUN = 6.957e8  # m
R_G_SUN = GM_SUN / (C**2)  # ~1476.625 m


def n_and_grad(pos: np.ndarray, r_g: float = 1.0, model: str = "pv") -> tuple[float, np.ndarray]:
    """Refractive index n(r) and grad(n) for:
    - 'pv':   n(r) = exp(2*r_g/r)
    - 'orig': n(r) = (1 - r_g/r)^(-2)
    - 'gr':   isotropic Schwarzschild optical metric n(r) = (1 + r_g/(2r))^3 / (1 - r_g/(2r))
    """
    r = float(np.linalg.norm(pos))
    if model == "pv":
        n_val = np.exp(2.0 * r_g / r)
        dn_dr = -2.0 * r_g / (r**2) * n_val
    elif model == "orig":
        if r <= r_g:
            raise ValueError("Inside r <= r_g singularity")
        fac = 1.0 - r_g / r
        n_val = fac ** (-2)
        dn_dr = -2.0 * r_g / (r**2 * fac**3)
    elif model == "gr":
        if r <= 0.5 * r_g:
            raise ValueError("Inside isotropic horizon r <= 0.5 r_g")
        u2 = 0.5 * r_g / r
        n_val = ((1.0 + u2) ** 3) / (1.0 - u2)
        # d/du2 [ (1+u2)^3 / (1-u2) ] = (1+u2)^2 * (4 - 2*u2) / (1-u2)^2
        dn_du2 = ((1.0 + u2) ** 2) * (4.0 - 2.0 * u2) / ((1.0 - u2) ** 2)
        dn_dr = dn_du2 * (-0.5 * r_g / (r**2))
    else:
        raise ValueError(f"Unknown model: {model}")
    return n_val, (dn_dr / r) * pos


def trace_ray_and_shapiro(
    b: float,
    x_start: float,
    x_end: float,
    r_g: float = 1.0,
    model: str = "pv",
) -> dict:
    """Integrate Eikonal ray + optical path delay int (n(r) - 1)/c dl."""
    b_dim = b / r_g
    xs_dim = x_start / r_g
    xe_dim = x_end / r_g

    r0 = np.array([xs_dim, b_dim], dtype=float)
    n0, _ = n_and_grad(r0, r_g=1.0, model=model)
    p0 = np.array([n0, 0.0], dtype=float)
    # State: [x, y, px, py, excess_optical_length_in_rg]
    y0 = np.array([r0[0], r0[1], p0[0], p0[1], 0.0], dtype=float)

    def rhs(_l: float, state: np.ndarray) -> np.ndarray:
        pos = state[:2]
        mom = state[2:4]
        n_val, grad_n = n_and_grad(pos, r_g=1.0, model=model)
        return np.array([mom[0] / n_val, mom[1] / n_val, grad_n[0], grad_n[1], n_val - 1.0])

    def reach_end(_l: float, state: np.ndarray) -> float:
        return state[0] - xe_dim

    reach_end.terminal = True
    reach_end.direction = 1

    r_cap = 1.05 if model == "orig" else (0.55 if model == "gr" else 0.35)

    def capture(_l: float, state: np.ndarray) -> float:
        return np.linalg.norm(state[:2]) - r_cap

    capture.terminal = True
    capture.direction = -1

    l_max = 2.0 * (xe_dim - xs_dim) + 40.0
    sol = solve_ivp(
        rhs,
        (0.0, l_max),
        y0,
        method="DOP853",
        events=[reach_end, capture],
        rtol=1e-11,
        atol=1e-13,
        max_step=max(0.25, 0.25 * b_dim) if b_dim < 100.0 else np.inf,
    )
    if not sol.success:
        raise RuntimeError(sol.message)

    captured = len(sol.t_events[1]) > 0
    traj = sol.y[:2].T * r_g
    moms = sol.y[2:4].T
    theta_in = np.arctan2(moms[0, 1], moms[0, 0])
    theta_out = np.arctan2(moms[-1, 1], moms[-1, 0])

    # Excess coordinate travel time due to n(r) > 1 AND slight path lengthening
    dl_total = sol.t[-1] * r_g
    straight_dist = np.hypot(traj[-1, 0] - traj[0, 0], traj[-1, 1] - traj[0, 1])
    excess_opt_len = sol.y[4, -1] * r_g + (dl_total - straight_dist)

    return {
        "b": b,
        "traj": traj,
        "captured": captured,
        "deflection_rad": abs(theta_out - theta_in),
        "shapiro_oneway_s": excess_opt_len / C,
    }


def solve_mercury_precession() -> dict:
    """Solve exact Polarizable-Ether Binet ODE for Mercury's orbit over 30 revolutions.

    From L = -m c^2 sqrt(e^{-2u} - e^{2u} v^2/c^2), exact radial orbit in u = r_g / r is:
        d^2 u / dphi^2 + u = (r_g / h_bar)^2 * (2 * E_bar^2 * e^{4u} - e^{2u})
    where E_bar = E/(m c^2) and h_bar = J/(m c).
    """
    a = 5.790905e10  # m (Mercury semi-major axis)
    ecc = 0.205630  # eccentricity
    period_s = 87.9691 * 86400.0
    sec_per_century = 100.0 * 365.25 * 86400.0
    orbits_per_century = sec_per_century / period_s

    r_p = a * (1.0 - ecc)
    r_a = a * (1.0 + ecc)
    u_p = R_G_SUN / r_p
    u_a = R_G_SUN / r_a

    # Exact conserved E_bar^2 and h_bar from turning points (du/dphi = 0 at u_p and u_a):
    # u^2 = (r_g/h_bar)^2 * (E_bar^2 * e^{4u} - e^{2u}) at u_p and u_a
    # => u_p^2 / u_a^2 = (E_bar^2 * e^{4u_p} - e^{2u_p}) / (E_bar^2 * e^{4u_a} - e^{2u_a})
    ratio2 = (u_p / u_a) ** 2
    E_bar2 = (ratio2 * np.exp(2.0 * u_a) - np.exp(2.0 * u_p)) / (
        ratio2 * np.exp(4.0 * u_a) - np.exp(4.0 * u_p)
    )
    inv_h2 = (u_p**2) / (E_bar2 * np.exp(4.0 * u_p) - np.exp(2.0 * u_p))

    # Solve for perturbation du(phi) = u(phi) - u_kepler(phi) or full u(phi) with scaled u_0
    # Using exact RHS in u directly with tight tolerances over 25 revolutions:
    n_orbits = 25
    phi_max = 2.0 * np.pi * n_orbits

    def rhs(_phi: float, state: np.ndarray) -> np.ndarray:
        u_val, du_val = state
        # Use expm1-stable form near u -> 0: 2*E^2*e^{4u} - e^{2u}
        rhs_force = inv_h2 * (2.0 * E_bar2 * np.exp(4.0 * u_val) - np.exp(2.0 * u_val))
        return np.array([du_val, rhs_force - u_val])

    phi_eval = np.linspace(0.0, phi_max, n_orbits * 2000 + 1)
    sol = solve_ivp(
        rhs,
        (0.0, phi_max),
        np.array([u_p, 0.0]),
        method="DOP853",
        t_eval=phi_eval,
        rtol=1e-13,
        atol=1e-21,
    )
    if not sol.success:
        raise RuntimeError(sol.message)

    # Find perihelion maxima of u(phi)
    peaks, _ = find_peaks(sol.y[0], distance=1500)
    # Parabolic sub-grid refinement around each peak for machine-precision peak angle
    peak_phis = [0.0]
    for idx in peaks:
        y1, y2, y3 = sol.y[0, idx - 1], sol.y[0, idx], sol.y[0, idx + 1]
        dphi = phi_eval[1] - phi_eval[0]
        delta = 0.5 * (y1 - y3) / (y1 - 2.0 * y2 + y3)
        peak_phis.append(phi_eval[idx] + delta * dphi)

    peak_phis = np.array(peak_phis)
    # Linear fit peak_phis[k] = k * (2*pi + domega)
    k = np.arange(len(peak_phis), dtype=float)
    domega_per_orbit = float(np.polyfit(k, peak_phis - 2.0 * np.pi * k, 1)[0])
    arcsec_per_century_num = np.degrees(domega_per_orbit) * 3600.0 * orbits_per_century

    # Analytical 1PN formula: 6 * pi * GM / (c^2 * a * (1 - e^2))
    domega_ana = 6.0 * np.pi * R_G_SUN / (a * (1.0 - ecc**2))
    arcsec_per_century_ana = np.degrees(domega_ana) * 3600.0 * orbits_per_century

    return {
        "arcsec_century_num": arcsec_per_century_num,
        "arcsec_century_ana": arcsec_per_century_ana,
        "rel_err": abs(arcsec_per_century_num - arcsec_per_century_ana) / arcsec_per_century_ana,
        "E_bar": np.sqrt(E_bar2),
    }


def make_comparison_plot(out_path: str = "solar_strongfield_tests.png") -> None:
    """Visualize strong-field shadow differences & Shapiro delay profile."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.5), dpi=160)

    # --- Panel 1: Critical Photon Shadow Comparison (Strong-field discriminator) ---
    models = [
        ("orig", r"Ê-te Eq. (30) $(1 - r_g/r)^{-2}$ ($b_{\rm crit}=6.750\,r_g$)", "crimson", "-."),
        ("pv", r"Ê-te Thống nhất $e^{2r_g/r}$ ($b_{\rm crit}=5.437\,r_g$)", "#1f77b4", "-"),
        ("gr", r"Tương đối rộng GR ($b_{\rm crit}=5.196\,r_g$)", "#2e7d32", "--"),
    ]
    # Trace grazing rays just above GR shadow (b = 5.5 r_g and 6.9 r_g)
    for mod, label, col, ls in models:
        res = trace_ray_and_shapiro(b=6.9, x_start=-30.0, x_end=30.0, r_g=1.0, model=mod)
        tr = res["traj"]
        ax1.plot(tr[:, 0], tr[:, 1], color=col, ls=ls, lw=2.1, label=rf"{label}: $|\Delta\varphi|={np.degrees(res['deflection_rad']):.1f}^\circ$")

    # Draw shadow disks as semi-transparent circles centered at origin for visual scale
    for r_b, col, ls in [(6.75, "crimson", "-."), (2.0 * np.e, "#1f77b4", "-"), (3.0 * np.sqrt(3.0), "#2e7d32", "--")]:
        ax1.add_patch(plt.Circle((0, 0), r_b, color=col, fill=False, ls=ls, lw=1.2, alpha=0.65))

    ax1.add_patch(plt.Circle((0, 0), 1.0, color="black", zorder=6))
    ax1.set_xlim(-22, 22)
    ax1.set_ylim(-14, 14)
    ax1.set_aspect("equal")
    ax1.set_xlabel(r"$x / r_g$")
    ax1.set_ylabel(r"$y / r_g$")
    ax1.set_title(r"Phân biệt trường mạnh ($b = 6.9\,r_g$) & Vòng bóng tới hạn $b_{\rm crit}$")
    ax1.legend(loc="lower left", fontsize=7.6, framealpha=0.93)
    ax1.grid(alpha=0.25, ls="--")

    # --- Panel 2: Shapiro Round-Trip Time Delay vs Impact Parameter (Cassini/Earth-Planet) ---
    # Earth at r_E = 1 AU, target spacecraft (Cassini at Saturn ~ 8.43 AU, or Mars ~ 1.524 AU)
    AU = 1.495978707e11
    x_earth = 1.0 * AU
    x_cassini = 8.43 * AU  # Saturn conjunction geometry
    b_vals_rsun = np.linspace(1.6, 20.0, 25)
    dt_num_us = []
    dt_ana_us = []
    for b_rs in b_vals_rsun:
        b_m = b_rs * R_SUN
        res = trace_ray_and_shapiro(b=b_m, x_start=-x_earth, x_end=x_cassini, r_g=R_G_SUN, model="pv")
        # Round-trip delay in microseconds
        dt_num_us.append(2.0 * res["shapiro_oneway_s"] * 1e6)
        dt_ana_us.append(2.0 * (2.0 * R_G_SUN / C) * np.log(4.0 * x_earth * x_cassini / (b_m**2)) * 1e6)

    ax2.plot(b_vals_rsun, dt_ana_us, color="#1f77b4", lw=2.3, label=r"Giải tích 1PN: $\Delta t_{\rm rt} = \frac{4GM_\odot}{c^3}\ln\left(\frac{4 r_E r_S}{b^2}\right)$")
    ax2.scatter(b_vals_rsun, dt_num_us, color="crimson", s=26, zorder=5, label="Tích phân ODE trực tiếp trên tia cong")
    ax2.axvline(1.6, color="gray", ls=":", lw=1.2)
    ax2.annotate(
        rf"Cận nhật Cassini ($b \approx 1.6\,R_\odot$)" + "\n" + rf"$\Delta t_{{\rm rt}} \approx {dt_num_us[0]:.1f}\ \mu\text{{s}}$",
        xy=(1.6, dt_num_us[0]),
        xytext=(4.5, dt_num_us[0] - 28.0),
        fontsize=8.4,
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.92),
    )
    ax2.set_xlabel(r"Tham số va chạm $b / R_\odot$")
    ax2.set_ylabel(r"Độ trễ khứ hồi Shapiro $\Delta t_{\rm rt}$ ($\mu$s)")
    ax2.set_title("Kiểm chứng Độ trễ Shapiro (Hình học Trái Đất – Mặt Trời – Cassini)")
    ax2.legend(loc="upper right", fontsize=8.0, framealpha=0.93)
    ax2.grid(alpha=0.25, ls="--")

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print("=== KIỂM CHỨNG 4 BÀI TEST HỆ MẶT TRỜI TỪ LAGRANGIAN THỐNG NHẤT ===")
    print("L = -m*c^2 * sqrt(e^{-2u} - e^{2u}*v^2/c^2), n(r) = e^{2u}, u = GM/(r*c^2)\n")

    # 1. Light bending at Solar limb
    ray_sun = trace_ray_and_shapiro(b=R_SUN, x_start=-300.0 * R_SUN, x_end=300.0 * R_SUN, r_g=R_G_SUN, model="pv")
    arcsec_bend = np.degrees(ray_sun["deflection_rad"]) * 3600.0
    print(f"1) Bẻ cong ánh sáng (b = R_sun): ODE = {arcsec_bend:.6f} arcsec | Lý thuyết = {np.degrees(4*R_G_SUN/R_SUN)*3600:.6f} arcsec")

    # 2. Shapiro round-trip time delay (Cassini superior conjunction at b = 1.6 R_sun)
    AU = 1.495978707e11
    b_cassini = 1.6 * R_SUN
    shapiro = trace_ray_and_shapiro(b=b_cassini, x_start=-1.0 * AU, x_end=8.43 * AU, r_g=R_G_SUN, model="pv")
    dt_rt_num = 2.0 * shapiro["shapiro_oneway_s"] * 1e6
    dt_rt_ana = 2.0 * (2.0 * R_G_SUN / C) * np.log(4.0 * AU * 8.43 * AU / (b_cassini**2)) * 1e6
    print(f"2) Độ trễ Shapiro khứ hồi (b = 1.6 R_sun): ODE = {dt_rt_num:.4f} us | Giải tích = {dt_rt_ana:.4f} us")

    # 3. Gravitational redshift at Solar surface
    z_exact = np.exp(-R_G_SUN / R_SUN) - 1.0
    z_weak = -R_G_SUN / R_SUN
    print(f"3) Dịch chuyển đỏ hấp dẫn (bề mặt Mặt Trời): e^(-u)-1 = {z_exact:.8e} | -GM/(Rc^2) = {z_weak:.8e}")

    # 4. Mercury perihelion precession from exact Binet ODE
    merc = solve_mercury_precession()
    print(
        f"4) Tiến động cận điểm Sao Thủy: ODE = {merc['arcsec_century_num']:.4f} ''/thế kỷ | "
        f"Giải tích 1PN = {merc['arcsec_century_ana']:.4f} ''/thế kỷ (sai số tương đối {merc['rel_err']:.3e})"
    )

    # 5. Strong-field shadow discriminator
    print("\n=== ĐỐI CHIẾU BÓNG LỖ ĐEN TRƯỜNG MẠNH (b_crit / r_g) ===")
    print(f"- Mô hình cũ (1 - r_g/r)^(-2): b_crit = {6.75:.4f} r_g")
    print(f"- Ê-te thống nhất exp(2r_g/r) : b_crit = {2.0*np.e:.4f} r_g")
    print(f"- Tương đối rộng GR (EHT)    : b_crit = {3.0*np.sqrt(3.0):.4f} r_g")

    make_comparison_plot("solar_strongfield_tests.png")
    print("\nĐã xuất đồ thị tại: solar_strongfield_tests.png")


if __name__ == "__main__":
    main()
