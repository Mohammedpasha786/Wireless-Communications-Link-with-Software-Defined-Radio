Generate BER vs SNR curves comparing theoretical and simulated performance.

Usage:
    python scripts/ber_sweep.py --modulation QPSK --channel awgn
    python scripts/ber_sweep.py --modulation QAM16 --channel rayleigh --plot

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.modulation.modulation import DPSK, QPSK, QAM
from src.channel.channel import AWGNChannel, RayleighChannel
from src.utils.metrics import bit_error_rate, ber_vs_snr_theoretical


def parse_args():
    p = argparse.ArgumentParser(description="BER vs SNR sweep")
    p.add_argument("--modulation", default="QPSK", choices=["DPSK","QPSK","QAM16","QAM64"])
    p.add_argument("--channel",    default="awgn",  choices=["awgn","rayleigh"])
    p.add_argument("--snr-min",    type=float, default=-5.0)
    p.add_argument("--snr-max",    type=float, default=30.0)
    p.add_argument("--snr-step",   type=float, default=2.0)
    p.add_argument("--n-bits",     type=int,   default=50_000)
    p.add_argument("--n-trials",   type=int,   default=3)
    p.add_argument("--plot",       action="store_true")
    p.add_argument("--save",       type=str,   default=None)
    return p.parse_args()


def main():
    args = parse_args()
    snr_range = np.arange(args.snr_min, args.snr_max + args.snr_step, args.snr_step)

    MODS = {"DPSK": DPSK, "QPSK": QPSK, "QAM16": lambda: QAM(16), "QAM64": lambda: QAM(64)}
    CHANS = {
        "awgn": lambda snr: AWGNChannel(snr_db=snr),
        "rayleigh": lambda snr: RayleighChannel(snr_db=snr),
    }

    print(f"\nBER Sweep: {args.modulation} over {args.channel.upper()} channel")
    print(f"{'SNR (dB)':>10}  {'BER (sim)':>12}  {'BER (theory)':>14}")
    print("-" * 42)

    sim_bers = []
    rng = np.random.default_rng(42)

    for snr_db in snr_range:
        trial_bers = []
        for _ in range(args.n_trials):
            bits = rng.integers(0, 2, args.n_bits).astype(np.uint8)
            mod = MODS[args.modulation]()
            symbols = mod.modulate(bits)
            rx = CHANS[args.channel](snr_db)(symbols)
            rx_bits = mod.demodulate(rx)
            trial_bers.append(bit_error_rate(bits, rx_bits))
        ber = float(np.mean(trial_bers))
        sim_bers.append(ber)

        theory = ber_vs_snr_theoretical(np.array([snr_db]), args.modulation)[0]
        print(f"{snr_db:>10.1f}  {ber:>12.2e}  {theory:>14.2e}")

    if args.plot or args.save:
        try:
            import matplotlib
            if not args.plot:
                matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            theory_bers = ber_vs_snr_theoretical(snr_range, args.modulation)

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.semilogy(snr_range, sim_bers, "o-", label=f"{args.modulation} (simulated)", lw=2)
            ax.semilogy(snr_range, theory_bers, "--", label=f"{args.modulation} (theory)", lw=2)
            ax.axhline(1e-3, color="gray", linestyle=":", lw=1, label="BER = 1e-3")
            ax.set_xlabel("SNR (dB)")
            ax.set_ylabel("BER")
            ax.set_title(f"BER vs SNR — {args.modulation}, {args.channel.upper()}")
            ax.legend()
            ax.grid(True, which="both", alpha=0.3)
            ax.set_ylim(1e-5, 1.0)
            plt.tight_layout()

            if args.save:
                plt.savefig(args.save, dpi=150)
                print(f"\nSaved plot: {args.save}")
            if args.plot:
                plt.show()
        except ImportError:
            print("matplotlib not installed — skipping plot")


if __name__ == "__main__":
    main()
