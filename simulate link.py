End-to-end simulation of the wireless communications link.

Usage:
    python scripts/simulate_link.py --config configs/qpsk_awgn.yaml
    python scripts/simulate_link.py --modulation QAM16 --snr 15 --channel rayleigh
    python scripts/simulate_link.py --optimize throughput
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.modulation.modulation import DPSK, QPSK, QAM, OFDM
from src.channel.channel import AWGNChannel, RayleighChannel, MultipathChannel
from src.sync.sync import CostasLoop, GardnerTED, SchmidlCox, FrameSynchronizer
from src.coding.fec import ConvolutionalCode, HammingCode, RepetitionCode
from src.utils.metrics import bit_error_rate, error_vector_magnitude, throughput_bps


MODULATIONS = {
    "DPSK":   lambda: DPSK(M=2),
    "QPSK":   lambda: QPSK(),
    "QAM16":  lambda: QAM(M=16),
    "QAM64":  lambda: QAM(M=64),
    "OFDM":   lambda: OFDM(n_fft=64, cp_len=16),
}

CHANNELS = {
    "awgn":      lambda snr: AWGNChannel(snr_db=snr),
    "rayleigh":  lambda snr: RayleighChannel(snr_db=snr, doppler_hz=10.0),
    "multipath": lambda snr: MultipathChannel(
        delays_samples=[0, 3, 7],
        path_gains_db=[0.0, -3.0, -10.0],
        snr_db=snr,
    ),
}

FEC_CODECS = {
    "none":       None,
    "hamming":    HammingCode(),
    "conv":       ConvolutionalCode(),
    "repetition": RepetitionCode(N=3),
}

OPTIMIZE_PRESETS = {
    "throughput":  dict(modulation="QAM64",  channel="awgn",  fec="none",    snr=30.0),
    "reliability": dict(modulation="DPSK",   channel="awgn",  fec="conv",    snr=5.0),
    "security":    dict(modulation="QPSK",   channel="awgn",  fec="hamming", snr=15.0),
}


def parse_args():
    p = argparse.ArgumentParser(description="SDR wireless link simulator")
    p.add_argument("--config",      type=str,   default=None)
    p.add_argument("--modulation",  type=str,   default="QPSK",  choices=list(MODULATIONS))
    p.add_argument("--channel",     type=str,   default="awgn",  choices=list(CHANNELS))
    p.add_argument("--fec",         type=str,   default="none",  choices=list(FEC_CODECS))
    p.add_argument("--snr",         type=float, default=15.0)
    p.add_argument("--n-bits",      type=int,   default=10_000)
    p.add_argument("--optimize",    type=str,   default=None,    choices=list(OPTIMIZE_PRESETS))
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--verbose",     action="store_true")
    return p.parse_args()


def run_link(
    n_bits: int,
    modulation_name: str,
    channel_name: str,
    fec_name: str,
    snr_db: float,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    rng = np.random.default_rng(seed)

    # 1. Generate random source bits
    tx_bits_raw = rng.integers(0, 2, n_bits).astype(np.uint8)

    # 2. FEC encode
    codec = FEC_CODECS[fec_name]
    if codec is not None:
        tx_bits = codec.encode(tx_bits_raw)
    else:
        tx_bits = tx_bits_raw.copy()

    # 3. Modulate
    mod = MODULATIONS[modulation_name]()
    tx_symbols = mod.modulate(tx_bits)

    # 4. Channel
    channel = CHANNELS[channel_name](snr_db)
    rx_symbols = channel(tx_symbols)

    # 5. Carrier sync (QPSK and QAM only)
    if modulation_name in ("QPSK", "QAM16", "QAM64"):
        M = {"QPSK": 4, "QAM16": 4, "QAM64": 4}[modulation_name]
        costas = CostasLoop(M=min(M, 4), loop_bw=0.01)
        rx_symbols, _ = costas.process(rx_symbols)

    # 6. Demodulate
    rx_bits_coded = mod.demodulate(rx_symbols)

    # 7. FEC decode
    if codec is not None:
        # Soft LLR for Viterbi; hard bits for Hamming/Repetition
        if fec_name == "conv":
            # Convert hard bits to LLR (±1 mapping)
            llr = 1.0 - 2.0 * rx_bits_coded.astype(float)
            rx_bits = codec.decode(llr)
        else:
            rx_bits = codec.decode(rx_bits_coded)
    else:
        rx_bits = rx_bits_coded

    # Align lengths
    n = min(len(tx_bits_raw), len(rx_bits))
    ber = bit_error_rate(tx_bits_raw[:n], rx_bits[:n])
    evm = error_vector_magnitude(tx_symbols, rx_symbols)

    # Throughput
    bits_per_symbol = getattr(mod, "bits_per_symbol", 2)
    if hasattr(mod, "bits_per_symbol"):
        bps = bits_per_symbol
    elif hasattr(mod, "_submod"):
        bps = mod._submod.bits_per_symbol
    else:
        bps = 2

    fec_rate = 1.0 if codec is None else getattr(codec, "rate", 0.5)
    symbol_rate = 1e6  # nominal 1 Msps
    tp = throughput_bps(n_bits, fec_rate, symbol_rate, bps, ber)

    results = {
        "modulation":   modulation_name,
        "channel":      channel_name,
        "fec":          fec_name,
        "snr_db":       snr_db,
        "ber":          ber,
        "evm_pct":      evm,
        "n_errors":     int(ber * n),
        **tp,
    }

    if verbose:
        print(f"\n{'='*55}")
        print(f"  Wireless Link Simulation Results")
        print(f"{'='*55}")
        print(f"  Modulation : {modulation_name}")
        print(f"  Channel    : {channel_name}")
        print(f"  FEC        : {fec_name}")
        print(f"  SNR        : {snr_db:.1f} dB")
        print(f"  BER        : {ber:.2e}  ({results['n_errors']} errors / {n} bits)")
        print(f"  EVM        : {evm:.2f}%")
        print(f"  Throughput : {tp['goodput_bps']/1e3:.1f} kbps (goodput)")
        print(f"  Gross rate : {tp['gross_bps']/1e3:.1f} kbps")
        print(f"{'='*55}\n")

    return results


def main():
    args = parse_args()

    # Apply optimization preset if selected
    if args.optimize:
        preset = OPTIMIZE_PRESETS[args.optimize]
        print(f"\n[Optimize: {args.optimize.upper()}] Using preset: {preset}")
        results = run_link(
            n_bits=args.n_bits,
            modulation_name=preset["modulation"],
            channel_name=preset["channel"],
            fec_name=preset["fec"],
            snr_db=preset["snr"],
            seed=args.seed,
            verbose=True,
        )
        return

    # Load YAML config if provided
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        results = run_link(
            n_bits=cfg.get("n_bits", args.n_bits),
            modulation_name=cfg["modulation"],
            channel_name=cfg["channel"],
            fec_name=cfg.get("fec", "none"),
            snr_db=cfg.get("snr_db", args.snr),
            seed=args.seed,
            verbose=args.verbose,
        )
        return

    # CLI arguments
    run_link(
        n_bits=args.n_bits,
        modulation_name=args.modulation,
        channel_name=args.channel,
        fec_name=args.fec,
        snr_db=args.snr,
        seed=args.seed,
        verbose=True,
    )


if __name__ == "__main__":
    main()
