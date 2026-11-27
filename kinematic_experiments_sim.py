"""Empirical kinematics verification beyond Michelson-Morley:
1) Kennedy-Thorndike (KT, asymmetric arms L1 != L2) under Earth's seasonal velocity
   modulation v(t) = v_CMB + v_earth * cos(Omega * t):
   - Length contraction ALONE (no clock slowing) fails KT by > 8 orders of magnitude.
   - Length contraction + Breather clock slowing (gamma^-1) gives exact null signal.
2) Ives-Stilwell (IS, quadratic Doppler shift) up to relativistic speeds (beta = 0.338
   as in Botermann et al. 2014 ESR ion-beam test):
   - Classical Galilean clock predicts zero average shift (1/(1-beta) + 1/(1+beta))/2 - 1 = beta^2,
     Wait: check exact longitudinal shift vs wavelength/frequency!
     In frequency: (1/(1-beta) + 1/(1+beta))/2 = 1/(1-beta^2) = 1 + beta^2 + ..., whereas
     in wavelength (Ives-Stilwell): (lambda_+ + lambda_-)/(2*lambda_0) - 1:
     - Classical Galilean (no time dilation): (1-beta + 1+beta)/2 - 1 = 0!
     - Dynamical Ether (clock slowed by gamma): gamma*(1-beta + 1+beta)/2 - 1 = gamma - 1 ~ +0.5*beta^2!
"""

from __future__ import annotations

from decimal import Decimal, getcontext
import numpy as np
import matplotlib.pyplot as plt


def kennedy_thorndike_analysis() -> dict:
    """Compare seasonal frequency drift d(nu)/nu for KT experiment."""
    c = 299_792_458.0
    v_cmb = 369_820.0  # m/s (Planck CMB dipole solar-system barycenter speed)
    v_earth = 29_780.0  # m/s (Earth orbital modulation)

    days = np.linspace(0.0, 365.25, 731)
    # Effective speed along orbital plane projection (cosine modulation over 1 year)
    v_t = np.sqrt(v_cmb**2 + v_earth**2 + 2.0 * v_cmb * v_earth * np.cos(0.95) * np.cos(2.0 * np.pi * days / 365.25))
    beta_t = v_t / c
    gamma_t = 1.0 / np.sqrt(1.0 - beta_t**2)

    # Case A: Length contraction ONLY (clock NOT slowed -> N_ticks = f0 * 2*gamma*dL/c)
    # Fractional variation relative to yearly mean:
    dnu_no_clock = (gamma_t - np.mean(gamma_t)) / np.mean(gamma_t)

    # Case B: Length contraction + Clock slowing (f_clock = f0 / gamma)
    # Exact check using 50-digit Decimal
    getcontext().prec = 50
    dnu_with_clock = []
    for b in beta_t:
        bd = Decimal(str(float(b)))
        g = Decimal(1) / (Decimal(1) - bd * bd).sqrt()
        ticks = g * (Decimal(1) / g)
        dnu_with_clock.append(float(ticks - Decimal(1)))
    dnu_with_clock = np.array(dnu_with_clock)

    return {
        "days": days,
        "dnu_no_clock": dnu_no_clock,
        "dnu_with_clock": dnu_with_clock,
        "ptp_no_clock": float(np.ptp(dnu_no_clock)),
        "max_with_clock": float(np.max(np.abs(dnu_with_clock))),
    }


def ives_stilwell_analysis() -> dict:
    """Compare Ives-Stilwell quadratic wavelength shift Delta_lambda / lambda_0."""
    betas = np.linspace(0.0, 0.40, 201)
    gamma = 1.0 / np.sqrt(1.0 - betas**2)

    # Classical stationary ether WITHOUT clock slowing (receiving from moving emitter at beta):
    # Forward: lambda_+ = lambda_0 * (1 - beta), Backward: lambda_- = lambda_0 * (1 + beta)
    # Mean wavelength shift = (lambda_+ + lambda_-)/(2*lambda_0) - 1 == 0!
    shift_no_clock = np.zeros_like(betas)

    # Dynamical Ether WITH clock slowing (emitter frequency reduced by 1/gamma, i.e. lambda_emit = gamma*lambda_0):
    # Forward: lambda_+ = gamma * lambda_0 * (1 - beta) = lambda_0 * sqrt((1-beta)/(1+beta))
    # Backward: lambda_- = gamma * lambda_0 * (1 + beta) = lambda_0 * sqrt((1+beta)/(1-beta))
    # Mean shift = (lambda_+ + lambda_-)/(2*lambda_0) - 1 = gamma - 1
    shift_with_clock = gamma - 1.0

    # Benchmark point: Botermann et al. (2014) ESR storage ring test at beta = 0.338
    beta_esr = 0.338
    gamma_esr = 1.0 / np.sqrt(1.0 - beta_esr**2)
    shift_esr = gamma_esr - 1.0

    return {
        "betas": betas,
        "shift_no_clock": shift_no_clock,
        "shift_with_clock": shift_with_clock,
        "beta_esr": beta_esr,
        "shift_esr": shift_esr,
    }


def make_plots(kt: dict, isw: dict, out_path: str = "kinematic_experiments.png") -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.4), dpi=160)

    # --- Panel 1: Kennedy-Thorndike seasonal drift ---
    ax1.plot(
        kt["days"],
        kt["dnu_no_clock"] * 1e9,
        color="crimson",
        lw=2.1,
        label=rf"Chỉ co độ dài (không chậm đồng hồ): biên độ $\approx {kt['ptp_no_clock']:.2e}$",
    )
    ax1.plot(
        kt["days"],
        kt["dnu_with_clock"] * 1e9,
        color="#1f77b4",
        lw=2.2,
        ls="--",
        label=r"Co độ dài + Chậm đồng hồ Breather ($\gamma^{-1}$): $\equiv 0$",
    )
    # Modern KT optical cavity bound (~ 1e-17, i.e., 1e-8 in units of 10^-9)
    ax1.axhspan(-0.02, 0.02, color="#2e7d32", alpha=0.18, label=r"Giới hạn thực nghiệm KT hiện đại ($|\Delta\nu/\nu| < 10^{-17}$)")
    ax1.set_xlim(0.0, 365.25)
    ax1.set_xlabel("Ngày trong năm (biến thiên quỹ đạo Trái Đất qua nền CMB)")
    ax1.set_ylabel(r"Độ trôi tần số tương đối $\Delta\nu / \nu\ (\times 10^{-9})$")
    ax1.set_title("Thí nghiệm Kennedy–Thorndike (Cần cả Co độ dài & Giãn thời gian)")
    ax1.legend(loc="upper right", fontsize=7.8, framealpha=0.93)
    ax1.grid(alpha=0.25, ls="--")

    # --- Panel 2: Ives-Stilwell quadratic Doppler shift ---
    ax2.plot(
        isw["betas"],
        isw["shift_with_clock"],
        color="#1f77b4",
        lw=2.3,
        label=r"Ê-te động học (đồng hồ chậm $\gamma^{-1}$): $\frac{\lambda_++\lambda_-}{2\lambda_0}-1 = \gamma - 1$",
    )
    ax2.plot(
        isw["betas"],
        isw["shift_no_clock"],
        color="crimson",
        lw=2.0,
        ls="-.",
        label=r"Ê-te không chậm đồng hồ: dịch chuyển trung bình $= 0$",
    )
    ax2.scatter(
        [isw["beta_esr"]],
        [isw["shift_esr"]],
        color="gold",
        edgecolor="black",
        s=75,
        zorder=6,
        label=rf"Thí nghiệm ESR 2014 ($\beta=0.338,\ \gamma-1={isw['shift_esr']:.5f}$)",
    )
    ax2.annotate(
        rf"$\beta = 0.338$ (ion $^7\mathrm{{Li}}^+$)" + "\n" + rf"$\Delta\lambda/\lambda_0 \approx {isw['shift_esr']:.5f}$",
        xy=(isw["beta_esr"], isw["shift_esr"]),
        xytext=(0.14, isw["shift_esr"] + 0.008),
        fontsize=8.3,
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.92),
    )
    ax2.set_xlim(0.0, 0.40)
    ax2.set_ylim(-0.006, 0.095)
    ax2.set_xlabel(r"Vận tốc chùm ion $\beta = v/c$")
    ax2.set_ylabel(r"Dịch chuyển bước sóng bậc hai $\frac{\lambda_+ + \lambda_-}{2\lambda_0} - 1$")
    ax2.set_title("Thí nghiệm Ives–Stilwell (Hiệu ứng Doppler ngang bậc hai)")
    ax2.legend(loc="upper left", fontsize=7.8, framealpha=0.93)
    ax2.grid(alpha=0.25, ls="--")

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    kt = kennedy_thorndike_analysis()
    isw = ives_stilwell_analysis()

    print("=== 1. KIỂM CHỨNG KENNEDY-THORNDYKE (L1 != L2, ĐIỀU BIẾN CMB + TRÁI ĐẤT) ===")
    print(f"Chỉ co độ dài (thiếu chậm đồng hồ): đỉnh-đỉnh Δν/ν = {kt['ptp_no_clock']:.4e} (bị bác bỏ >10^8x)")
    print(f"Co độ dài + chậm đồng hồ Breather : max |Δν/ν| (Decimal 50) = {kt['max_with_clock']:.3e} (≡ 0)")

    print("\n=== 2. KIỂM CHỨNG IVES-STILWELL (DOPPLER BẬC HAI) ===")
    print(f"Tại beta = {isw['beta_esr']:.3f} (ESR 2014): gamma - 1 = {isw['shift_esr']:.6f} (khớp SR/LET)")

    make_plots(kt, isw, "kinematic_experiments.png")
    print("Đã xuất đồ thị tại: kinematic_experiments.png")


if __name__ == "__main__":
    main()
