Communication link performance metrics:
  - BER  : Bit Error Rate
  - EVM  : Error Vector Magnitude
  - SNR  : Measured Signal-to-Noise Ratio
  - Throughput
  - BER vs SNR sweep
import numpy as np
from typing import Dict, List, Tuple, Optional


def bit_error_rate(tx_bits: np.ndarray, rx_bits: np.ndarray) -> float:
    """Compute Bit Error Rate between transmitted and received bits."""
    tx = np.asarray(tx_bits).flatten()
    rx = np.asarray(rx_bits).flatten()
    n = min(len(tx), len(rx))
    if n == 0:
        return 0.0
    return float(np.sum(tx[:n] != rx[:n])) / n


def error_vector_magnitude(
    tx_symbols: np.ndarray, rx_symbols: np.ndarray
) -> float:
    """
    Error Vector Magnitude (EVM) in percent.
    EVM = RMS(error vectors) / RMS(reference) × 100%
    """
    tx = np.asarray(tx_symbols)
    rx = np.asarray(rx_symbols)
    n = min(len(tx), len(rx))
    error = rx[:n] - tx[:n]
    evm_rms = np.sqrt(np.mean(np.abs(error) ** 2))
    ref_rms = np.sqrt(np.mean(np.abs(tx[:n]) ** 2))
    return float(evm_rms / (ref_rms + 1e-12) * 100.0)


def measured_snr_db(signal: np.ndarray, noise: np.ndarray) -> float:
    """Compute SNR in dB given separate signal and noise arrays."""
    sig_power = np.mean(np.abs(signal) ** 2)
    noise_power = np.mean(np.abs(noise) ** 2)
    return float(10 * np.log10(sig_power / (noise_power + 1e-30)))


def throughput_bps(
    n_bits: int,
    fec_rate: float,
    symbol_rate: float,
    bits_per_symbol: int,
    ber: float,
) -> Dict[str, float]:
    """
    Compute link throughput metrics.

    Args:
        n_bits:          Total bits transmitted
        fec_rate:        FEC code rate (e.g. 0.5 for rate-1/2)
        symbol_rate:     Symbol rate [symbols/second]
        bits_per_symbol: Modulation order bits per symbol
        ber:             Measured bit error rate

    Returns:
        dict with gross_bps, net_bps, goodput_bps
    """
    gross_bps = symbol_rate * bits_per_symbol
    net_bps = gross_bps * fec_rate
    goodput_bps = net_bps * (1.0 - ber)
    return {
        "gross_bps": gross_bps,
        "net_bps": net_bps,
        "goodput_bps": goodput_bps,
        "spectral_efficiency": bits_per_symbol * fec_rate * (1.0 - ber),
    }


def ber_vs_snr_theoretical(
    snr_db_range: np.ndarray,
    modulation: str = "QPSK",
) -> np.ndarray:
    """
    Theoretical BER vs SNR curves for AWGN channel.

    Args:
        snr_db_range: Array of SNR values [dB]
        modulation:   'BPSK', 'QPSK', 'QAM16', 'QAM64'

    Returns:
        BER array (same shape as snr_db_range)
    """
    from scipy.special import erfc

    snr_linear = 10 ** (np.asarray(snr_db_range) / 10.0)

    if modulation in ("BPSK", "DPSK"):
        return 0.5 * erfc(np.sqrt(snr_linear))
    elif modulation == "QPSK":
        return 0.5 * erfc(np.sqrt(snr_linear / 2.0))
    elif modulation == "QAM16":
        return (3.0 / 8.0) * erfc(np.sqrt(snr_linear / 10.0))
    elif modulation == "QAM64":
        return (7.0 / 24.0) * erfc(np.sqrt(snr_linear / 42.0))
    else:
        raise ValueError(f"Unknown modulation: {modulation}")


def ber_sweep(
    tx_bits: np.ndarray,
    modulator,
    channel_factory,
    demodulator,
    snr_range_db: np.ndarray,
    n_trials: int = 3,
) -> Dict[str, np.ndarray]:
    """
    Run BER vs SNR sweep simulation.

    Args:
        tx_bits:         Bits to transmit
        modulator:       Callable: bits → symbols
        channel_factory: Callable: snr_db → channel callable
        demodulator:     Callable: symbols → bits
        snr_range_db:    Array of SNR test points
        n_trials:        Number of Monte Carlo trials per SNR point

    Returns:
        dict with 'snr_db', 'ber_mean', 'ber_std'
    """
    ber_results = np.zeros((len(snr_range_db), n_trials))

    for i, snr_db in enumerate(snr_range_db):
        for trial in range(n_trials):
            symbols = modulator(tx_bits)
            channel = channel_factory(snr_db)
            rx_symbols = channel(symbols)
            rx_bits = demodulator(rx_symbols)
            ber_results[i, trial] = bit_error_rate(tx_bits, rx_bits)

    return {
        "snr_db": snr_range_db,
        "ber_mean": ber_results.mean(axis=1),
        "ber_std": ber_results.std(axis=1),
    }


def link_budget_table(
    freq_hz: float,
    tx_power_dbm: float,
    tx_gain_dbi: float,
    rx_gain_dbi: float,
    distances_m: List[float],
    bandwidth_hz: float,
    noise_figure_db: float = 5.0,
) -> List[Dict]:
    """
    Generate a link budget table for a range of distances.
    """
    c = 3e8
    wavelength = c / freq_hz
    k_b = 1.38e-23
    temp_k = 290.0

    thermal_noise_dbm = 10 * np.log10(k_b * temp_k * bandwidth_hz * 1000)
    noise_floor_dbm = thermal_noise_dbm + noise_figure_db

    rows = []
    for d in distances_m:
        fspl_db = 20 * np.log10(4 * np.pi * d / wavelength)
        rx_power_dbm = tx_power_dbm + tx_gain_dbi + rx_gain_dbi - fspl_db
        snr_db = rx_power_dbm - noise_floor_dbm
        rows.append({
            "distance_m": d,
            "fspl_db": round(fspl_db, 1),
            "rx_power_dbm": round(rx_power_dbm, 1),
            "snr_db": round(snr_db, 1),
        })
    return rows
