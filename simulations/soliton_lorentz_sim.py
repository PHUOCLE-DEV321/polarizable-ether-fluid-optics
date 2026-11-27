"""Direct nonlinear PDE simulation of Sine-Gordon standing-wave solitons in a
stationary ether (c = 1):
    d^2(psi)/dt^2 - c^2 * d^2(psi)/dx^2 + sin(psi) = 0

Demonstrates that wave-structured matter propagating at velocity v = beta * c:
1) Dynamically exhibits spatial contraction L(beta)/L(0) = sqrt(1 - beta^2) (Chapter I).
2) Dynamically exhibits internal oscillation slowing T(beta)/T(0) = 1 / sqrt(1 - beta^2)
   (kinetic time dilation required for Kennedy-Thorndike & Ives-Stilwell).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


def laplacian_4th_order(f: np.ndarray, dx: float) -> np.ndarray:
    """4th-order central finite-difference Laplacian with clamped/Neumann ghost boundaries."""
    f_pad = np.pad(f, 2, mode="edge")
    return (-f_pad[:-4] + 16.0 * f_pad[1:-3] - 30.0 * f_pad[2:-2] + 16.0 * f_pad[3:-1] - f_pad[4:]) / (
        12.0 * dx**2
    )


def evolve_sine_gordon(
    x: np.ndarray,
    psi0: np.ndarray,
    vel0: np.ndarray,
    t_end: float,
    dt: float,
    sample_every: int = 20,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Symplectic Stormer-Verlet integration of d^2 psi/dt^2 = d^2 psi/dx^2 - sin(psi)."""
    dx = float(x[1] - x[0])
    n_steps = int(round(t_end / dt))
    psi = psi0.copy()
    vel = vel0.copy()

    acc = laplacian_4th_order(psi, dx) - np.sin(psi)
    times = [0.0]
    snaps = [psi.copy()]

    for step in range(1, n_steps + 1):
        vel += 0.5 * dt * acc
        psi += dt * vel
        acc = laplacian_4th_order(psi, dx) - np.sin(psi)
        vel += 0.5 * dt * acc
        if step % sample_every == 0:
            times.append(step * dt)
            snaps.append(psi.copy())

    return np.array(times), snaps


def kink_ic(x: np.ndarray, x0: float, beta: float) -> tuple[np.ndarray, np.ndarray]:
    """Moving Sine-Gordon kink initial condition at t = 0."""
    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    xi = gamma * (x - x0)
    psi0 = 4.0 * np.arctan(np.exp(xi))
    # d(psi)/dt = -beta * gamma * sech(xi) * 2
    vel0 = -2.0 * beta * gamma / np.cosh(xi)
    return psi0, vel0


def breather_ic(x: np.ndarray, x0: float, beta: float, omega0: float = 0.6) -> tuple[np.ndarray, np.ndarray]:
    """Moving Sine-Gordon breather (localized oscillating particle) at t = 0."""
    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    sq = np.sqrt(1.0 - omega0**2)
    # Exact moving breather evaluated analytically at t = 0 and dt_eps for clean d_t(psi)
    def field(t_val: float) -> np.ndarray:
        xp = gamma * (x - x0 - beta * t_val)
        tp = gamma * (t_val - beta * (x - x0))
        arg = (sq / omega0) * np.sin(omega0 * tp) / np.cosh(sq * xp)
        return 4.0 * np.arctan(arg)

    eps = 1e-6
    psi0 = field(0.0)
    vel0 = (field(eps) - field(-eps)) / (2.0 * eps)
    return psi0, vel0


def measure_kink_fwhm(x: np.ndarray, psi: np.ndarray) -> float:
    """Measure FWHM of spatial gradient d(psi)/dx using sub-grid linear interpolation."""
    grad = np.gradient(psi, x)
    half_max = 0.5 * np.max(grad)
    above = np.where(grad >= half_max)[0]
    i_l, i_r = above[0], above[-1]

    # Sub-grid interpolation on left and right crossings
    x_l = x[i_l - 1] + (half_max - grad[i_l - 1]) / (grad[i_l] - grad[i_l - 1]) * (x[i_l] - x[i_l - 1])
    x_r = x[i_r] + (half_max - grad[i_r]) / (grad[i_r + 1] - grad[i_r]) * (x[i_r + 1] - x[i_r])
    return float(x_r - x_l)


def main() -> None:
    dx = 0.04
    dt = 0.01  # CFL = dt/dx = 0.25
    x = np.arange(-55.0, 55.0 + 0.5 * dx, dx)
    t_end_kink = 24.0

    betas = [0.0, 0.50, 0.75, 0.85]
    kink_results = []

    print("=== 1. MÔ PHỎNG PDE SINE-GORDON: ĐO BỀ RỘNG CO ĐỘ DÀI CỦA KINK SOLITON ===")
    print(f"{'beta':>6} | {'FWHM đo (t=24)':>15} | {'W(beta)/W(0)':>14} | {'sqrt(1-beta^2)':>15} | {'Sai khác':>10}")
    print("-" * 72)

    w0 = None
    for beta in betas:
        psi0, vel0 = kink_ic(x, x0=-10.0 * beta, beta=beta)
        times, snaps = evolve_sine_gordon(x, psi0, vel0, t_end=t_end_kink, dt=dt, sample_every=200)
        psi_final = snaps[-1]
        w_meas = measure_kink_fwhm(x, psi_final)
        if w0 is None:
            w0 = w_meas
        ratio = w_meas / w0
        exact = np.sqrt(1.0 - beta**2)
        kink_results.append((beta, psi_final, w_meas, ratio, exact))
        print(f"{beta:6.2f} | {w_meas:15.6f} | {ratio:14.6f} | {exact:15.6f} | {abs(ratio - exact):10.3e}")

    # 2. Breather oscillation period (time dilation) at beta = 0.0 vs beta = 0.75
    omega0 = 0.6
    t_end_br = 45.0
    x_br = np.arange(-45.0, 55.0 + 0.5 * dx, dx)
    br_data = {}
    for beta in [0.0, 0.75]:
        psi0, vel0 = breather_ic(x_br, x0=-12.0 * beta, beta=beta, omega0=omega0)
        times, snaps = evolve_sine_gordon(x_br, psi0, vel0, t_end=t_end_br, dt=dt, sample_every=5)
        # Track core amplitude psi(x_peak(t), t) along moving breather center
        core_vals = np.array([np.max(np.abs(s)) for s in snaps])
        # Each full breather period has 2 positive amplitude maxima (|sin(omega0*t')|)
        pks, _ = find_peaks(core_vals, height=1.5, distance=int(2.0 / (times[1] - times[0])))
        half_periods = np.diff(times[pks])
        period_meas = 2.0 * float(np.mean(half_periods))
        br_data[beta] = (times, core_vals, period_meas)

    T0 = br_data[0.0][2]
    T_mov = br_data[0.75][2]
    gamma_exact = 1.0 / np.sqrt(1.0 - 0.75**2)
    print("\n=== 2. MÔ PHỎNG PDE BREATHER: GIÃN CHU KỲ DAO ĐỘNG (ĐỒNG HỒ SÓNG DỪNG) ===")
    print(f"Chu kỳ đứng yên T(0.00)  = {T0:.5f} (lý thuyết 2*pi/omega0 = {2*np.pi/omega0:.5f})")
    print(f"Chu kỳ chuyển động T(0.75) = {T_mov:.5f}")
    print(f"Tỷ số T(0.75)/T(0.00)    = {T_mov/T0:.5f} vs gamma = {gamma_exact:.5f} (sai khác {abs(T_mov/T0 - gamma_exact):.3e})")

    # --- Visualization ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.4), dpi=160)

    # Panel 1: Centered gradient profiles showing contraction at t = 24
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "crimson"]
    for (beta, psi_fin, w_meas, ratio, exact), col in zip(kink_results, colors):
        grad = np.gradient(psi_fin, x)
        x_center = x[np.argmax(grad)]
        ax1.plot(
            x - x_center,
            grad / np.max(grad),
            color=col,
            lw=2.1,
            label=rf"$\beta={beta:.2f}$: $W/W_0={ratio:.3f}$ ($\sqrt{{1-\beta^2}}={exact:.3f}$)",
        )
    ax1.set_xlim(-6.0, 6.0)
    ax1.set_xlabel(r"$x - x_{\rm center}$ (trên lưới Ê-te đứng yên)")
    ax1.set_ylabel(r"Mật độ biến dạng chuẩn hóa $(\partial_x\psi)/(\partial_x\psi)_{\max}$")
    ax1.set_title("Co độ dài tự phát của Soliton chuyển động trên lưới Ê-te ($t=24$)")
    ax1.legend(loc="upper right", fontsize=8.0, framealpha=0.93)
    ax1.grid(alpha=0.25, ls="--")

    # Panel 2: Breather internal oscillation (time dilation)
    t0_arr, amp0, _ = br_data[0.0]
    tm_arr, ampm, _ = br_data[0.75]
    ax2.plot(t0_arr, amp0, color="#1f77b4", lw=1.9, label=rf"Đứng yên ($\beta=0$): $T_0={T0:.2f}$")
    ax2.plot(
        tm_arr,
        ampm,
        color="crimson",
        lw=2.0,
        ls="--",
        label=rf"Chuyển động ($\beta=0.75$): $T={T_mov:.2f} \approx \gamma T_0$ ($\gamma={gamma_exact:.3f}$)",
    )
    ax2.set_xlim(0.0, t_end_br)
    ax2.set_xlabel("Thời gian tuyệt đối $t$ của lưới Ê-te")
    ax2.set_ylabel(r"Biên độ cực đại hạt sóng dừng $\max_x|\psi(x,t)|$")
    ax2.set_title("Chậm nhịp dao động nội tại (Giãn thời gian động học) của Breather")
    ax2.legend(loc="lower right", fontsize=8.0, framealpha=0.93)
    ax2.grid(alpha=0.25, ls="--")

    plt.tight_layout()
    out_png = "soliton_lorentz.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"\nĐã xuất đồ thị tại: {out_png}")


if __name__ == "__main__":
    main()
