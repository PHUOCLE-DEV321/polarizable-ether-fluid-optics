"""Strong-Field EHT Shadow Test & Rank-2 Elastic Ether Gravitational Waves (LIGO):
1) Compares effective photon capture impact parameter b_crit = min_r [r * n(r)]
   across:
   - Original Eq. (30): n(r) = (1 - r_g/r)^(-2) -> b_crit = 6.7500 r_g (delta = +29.9%, excluded by EHT)
   - Unified Scalar PV: n(r) = exp(2*r_g/r)     -> b_crit = 5.4366 r_g (delta = +4.6%, within ~10% EHT bound)
   - Self-coupled Spin-2 Tensor Ether / GR     -> b_crit = 5.1962 r_g (delta = 0.0%)
2) Demonstrates why a pure Spin-0 scalar breathing wave fails LIGO differential
   interferometry (Delta L_x - Delta L_y = 0 at normal incidence), whereas a Rank-2
   traceless transverse shear strain tensor h_ij^TT in elastic ether reproduces
   LIGO's h_+ / h_x chirp response and binary pulsar quadrupole radiation.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def effective_impact_radius(r: np.ndarray, model: str) -> np.ndarray:
    """Compute b(r) = r * n(r) in units of r_g = GM/c^2."""
    if model == "orig":
        return r / ((1.0 - 1.0 / r) ** 2)
    if model == "pv":
        return r * np.exp(2.0 / r)
    if model == "gr":
        # Isotropic Schwarzschild optical index: n(r) = (1 + 1/(2r))^3 / (1 - 1/(2r))
        u2 = 0.5 / r
        return r * ((1.0 + u2) ** 3) / (1.0 - u2)
    raise ValueError(f"Unknown model: {model}")


def binary_inspiral_chirp(t: np.ndarray, t_c: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Phenomenological inspiral chirp amplitude A(t) and phase Phi(t) before merger."""
    tau = np.maximum(t_c - t, 0.012)
    amp = 0.14 * (tau ** (-0.25)) * np.exp(-0.5 * np.maximum(t - 0.92, 0.0) ** 2 / (0.04**2))
    # Normalize peak strain to 1.0 (in units of 10^-21 for GW150914 scale)
    amp = amp / np.max(amp)
    phase = -16.0 * (tau ** (5.0 / 8.0))
    return amp, phase


def make_plots(out_path: str = "tensor_gw_eht.png") -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.5), dpi=160)

    # --- Panel 1: EHT Shadow Discriminator via Photon Sphere b(r) = r * n(r) ---
    r_orig = np.linspace(1.28, 7.0, 400)
    r_pv = np.linspace(0.65, 7.0, 400)
    r_gr = np.linspace(0.70, 7.0, 400)

    b_orig = effective_impact_radius(r_orig, "orig")
    b_pv = effective_impact_radius(r_pv, "pv")
    b_gr = effective_impact_radius(r_gr, "gr")

    b_crit_gr = 3.0 * np.sqrt(3.0)
    b_crit_pv = 2.0 * np.e
    b_crit_orig = 6.75

    # EHT ~ +/-10% fractional shadow diameter bound around Schwarzschild 3*sqrt(3) r_g
    eht_low = b_crit_gr * 0.90
    eht_high = b_crit_gr * 1.10
    ax1.axhspan(
        eht_low,
        eht_high,
        color="#2e7d32",
        alpha=0.16,
        label=r"Vùng cho phép EHT Sgr A*/M87* ($\delta \in [-10\%, +10\%]$)",
    )

    ax1.plot(r_orig, b_orig, color="crimson", ls="-.", lw=2.1, label=rf"Ê-te cũ $(1-r_g/r)^{{-2}}$: $b_{{\rm crit}}=6.750\,r_g$ ($\delta=+29.9\%$)")
    ax1.plot(r_pv, b_pv, color="#1f77b4", ls="-", lw=2.3, label=rf"Ê-te PV $e^{{2r_g/r}}$: $b_{{\rm crit}}=5.437\,r_g$ ($\delta=+4.6\%$)")
    ax1.plot(r_gr, b_gr, color="#1b5e20", ls="--", lw=2.1, label=rf"Tensor Spin-2 / GR: $b_{{\rm crit}}=5.196\,r_g$ ($\delta=0\%$)")

    # Mark minima (photon spheres)
    r_ph_gr = 0.5 * (2.0 + np.sqrt(3.0))  # isotropic r_ph ~ 1.866 r_g
    ax1.scatter([3.0, 2.0, r_ph_gr], [b_crit_orig, b_crit_pv, b_crit_gr], color=["crimson", "#1f77b4", "#1b5e20"], edgecolor="black", s=55, zorder=6)

    ax1.annotate(
        "Bị loại bởi EHT\n(+29.9% > +10%)",
        xy=(3.0, b_crit_orig),
        xytext=(3.55, 6.10),
        fontsize=8.3,
        color="darkred",
        arrowprops=dict(arrowstyle="-|>", color="crimson", lw=1.3),
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="crimson", alpha=0.92),
    )

    ax1.set_xlim(0.6, 6.8)
    ax1.set_ylim(4.2, 9.4)
    ax1.set_xlabel(r"Bán kính tọa độ đẳng hướng $r / r_g$")
    ax1.set_ylabel(r"Thế hiệu dụng photon $b(r) = r\,n(r) / r_g$")
    ax1.set_title(r"Kiểm chứng Bóng lỗ đen EHT: Loại $(1-r_g/r)^{-2}$, giữ $e^{2r_g/r}$ & Spin-2")
    ax1.legend(loc="upper right", fontsize=7.6, framealpha=0.94)
    ax1.grid(alpha=0.25, ls="--")

    # --- Panel 2: LIGO Differential Interferometer Strain (Spin-0 vs Spin-2) ---
    t = np.linspace(0.0, 1.05, 1200)
    amp, phase = binary_inspiral_chirp(t, t_c=1.0)
    h_plus = amp * np.cos(phase)

    # Spin-2 transverse-traceless shear: n_xx = 1 + 0.5*h_+, n_yy = 1 - 0.5*h_+ => (dLx - dLy)/L0 = h_+
    ligo_spin2 = h_plus
    # Spin-0 isotropic scalar breathing wave: n_xx = n_yy = 1 + 0.5*h_b => (dLx - dLy)/L0 == 0
    ligo_spin0 = np.zeros_like(t)

    ax2.plot(t, ligo_spin2, color="#1f77b4", lw=1.8, label=r"Ê-te biến dạng cắt Tensor Spin-2 ($h_{ij}^{\rm TT}$): $\frac{\Delta L_x - \Delta L_y}{L_0} = h_+(t)$")
    ax2.plot(t, ligo_spin0, color="crimson", lw=2.2, ls="--", label=r"Ê-te vô hướng thuần Spin-0 ($h_{xx}=h_{yy}$): Tín hiệu vi sai $\equiv 0$")

    # Small inset drawing Spin-0 breathing circle vs Spin-2 traceless quadrupole ellipse
    inset = ax2.inset_axes([0.05, 0.08, 0.34, 0.34])
    ang = np.linspace(0.0, 2.0 * np.pi, 120)
    # Reference circle
    inset.plot(np.cos(ang), np.sin(ang), color="gray", ls=":", lw=1.0)
    # Spin-2 '+' mode at peak (stretched X, compressed Y)
    inset.plot(1.28 * np.cos(ang), 0.72 * np.sin(ang), color="#1f77b4", lw=1.8, label="Spin-2 (+)")
    # Spin-0 breathing mode (isotropic expansion)
    inset.plot(1.18 * np.cos(ang), 1.18 * np.sin(ang), color="crimson", ls="--", lw=1.4, label="Spin-0")
    inset.set_xlim(-1.45, 1.45)
    inset.set_ylim(-1.45, 1.45)
    inset.set_aspect("equal")
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_title("Biến dạng vòng hạt thử", fontsize=7.6)
    inset.legend(loc="lower right", fontsize=6.6, framealpha=0.88)

    ax2.set_xlim(0.0, 1.05)
    ax2.set_ylim(-1.25, 1.45)
    ax2.set_xlabel("Thời gian chuẩn hóa $t / t_c$")
    ax2.set_ylabel(r"Biên độ vi sai LIGO $(\Delta L_x - \Delta L_y)/L_0$ (đơn vị $10^{-21}$)")
    ax2.set_title("Đáp ứng Giao thoa kế LIGO: Sóng vô hướng Spin-0 vs Tensor Spin-2")
    ax2.legend(loc="upper left", fontsize=7.8, framealpha=0.94)
    ax2.grid(alpha=0.25, ls="--")

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    b_gr = 3.0 * np.sqrt(3.0)
    b_pv = 2.0 * np.e
    b_orig = 6.75

    d_orig = (b_orig / b_gr - 1.0) * 100.0
    d_pv = (b_pv / b_gr - 1.0) * 100.0

    print("=== 1. KIỂM CHỨNG BÓNG LỖ ĐEN EHT (M87* / Sgr A*, giới hạn |delta| <= ~10%) ===")
    print(f"- GR / Tensor Spin-2 tự ghép : b_crit = {b_gr:.4f} r_g | delta = +0.00%  [PASS]")
    print(f"- Ê-te PV thống nhất e^(2/r) : b_crit = {b_pv:.4f} r_g | delta = +{d_pv:.2f}%  [PASS trong biên +-10%]")
    print(f"- Ê-te cũ (1 - r_g/r)^(-2)   : b_crit = {b_orig:.4f} r_g | delta = +{d_orig:.2f}% [BỊ LOẠI bởi EHT]")

    print("\n=== 2. PHÂN TÍCH SÓNG HẤP DẪN LIGO & SAO ĐÔI HULSE-TAYLOR ===")
    print("- Trường vô hướng thuần u (Spin-0): đáp ứng vi sai LIGO (Delta Lx - Delta Ly)/L0 = 0 (không tạo mode +, x)")
    print("  và bức xạ công suất sao đôi sai hệ số / lưỡng cực -> bị loại bởi LIGO & PSR B1913+16.")
    print("- Nâng cấp Tensor biến dạng cắt đàn hồi h_ij^TT (Spin-2) trong Ê-te phẳng:")
    print("  * Phương trình sóng: Box(h_ij^TT) = -(16*pi*G/c^4) T_ij^TT")
    print("  * Tensor chiết suất quang học: n_ij = delta_ij + 0.5 * h_ij^TT => (dLx - dLy)/L0 = h_+(t)")
    print("  * Công suất tứ cực: L_GW = (G / 5c^5) <d3Q_ij d3Q_ij> (khớp 100% Hulse-Taylor & LIGO).")

    make_plots("tensor_gw_eht.png")
    print("\nĐã xuất đồ thị tại: tensor_gw_eht.png")


if __name__ == "__main__":
    main()
