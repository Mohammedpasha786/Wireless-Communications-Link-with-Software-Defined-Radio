Wireless channel models: AWGN, flat Rayleigh fading, multipath (frequency-selective).

All channels accept complex baseband samples and return impaired samples.
import numpy as np
from scipy.signal import fftconvolve
from typing import List, Optional, Tuple


class AWGNChannel:
    """
    Additive White Gaussian Noise channel.

    Args:
        snr_db: Signal-to-Noise Ratio in dB
        seed:   Random seed for reproducibility
    """

    def __init__(self, snr_db: float = 20.0, seed: Optional[int] = None):
        self.snr_db = snr_db
        self.rng = np.random.default_rng(seed)

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        snr_linear = 10 ** (self.snr_db / 10.0)
        signal_power = np.mean(np.abs(signal) ** 2)
        noise_power = signal_power / snr_linear
        noise = self.rng.standard_normal(signal.shape) + 1j * self.rng.standard_normal(signal.shape)
        noise *= np.sqrt(noise_power / 2)
        return signal + noise

    def set_snr(self, snr_db: float):
        self.snr_db = snr_db


class RayleighChannel:
    """
    Flat Rayleigh fading channel with optional AWGN.
    Models a single-path environment with rich multipath (no dominant LoS).

    Args:
        snr_db:         SNR after fading
        doppler_hz:     Maximum Doppler frequency [Hz] (set 0 for quasi-static)
        sample_rate:    System sample rate [Hz]
        coherence_time: Coherence time in samples (overrides doppler_hz if set)
    """

    def __init__(
        self,
        snr_db: float = 20.0,
        doppler_hz: float = 10.0,
        sample_rate: float = 1e6,
        seed: Optional[int] = None,
    ):
        self.snr_db = snr_db
        self.doppler_hz = doppler_hz
        self.sample_rate = sample_rate
        self.rng = np.random.default_rng(seed)

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        N = len(signal)
        # Generate Jakes-like fading envelope
        fading = self._jakes_fading(N)
        faded = signal * fading

        # Add AWGN
        snr_linear = 10 ** (self.snr_db / 10.0)
        signal_power = np.mean(np.abs(faded) ** 2)
        noise_power = signal_power / snr_linear
        noise = (
            self.rng.standard_normal(N) + 1j * self.rng.standard_normal(N)
        ) * np.sqrt(noise_power / 2)
        return faded + noise

    def _jakes_fading(self, N: int, n_sinusoids: int = 16) -> np.ndarray:
        """Approximate Jakes model using sum-of-sinusoids."""
        t = np.arange(N) / self.sample_rate
        h = np.zeros(N, dtype=complex)
        for n in range(1, n_sinusoids + 1):
            theta_n = np.pi * n / n_sinusoids
            phi_n = self.rng.uniform(0, 2 * np.pi)
            h += np.exp(1j * (2 * np.pi * self.doppler_hz * np.cos(theta_n) * t + phi_n))
        h /= np.sqrt(n_sinusoids)
        return h


class MultipathChannel:
    """
    Frequency-selective multipath channel with AWGN.
    Models multiple propagation paths with independent delays and gains.

    Args:
        delays_samples:  List of path delays in samples
        path_gains_db:   List of path gains in dB (same length as delays)
        snr_db:          SNR at receiver
        doppler_hz:      Doppler per path (quasi-static if 0)
        sample_rate:     System sample rate
    """

    def __init__(
        self,
        delays_samples: List[int] = None,
        path_gains_db: List[float] = None,
        snr_db: float = 20.0,
        doppler_hz: float = 0.0,
        sample_rate: float = 1e6,
        seed: Optional[int] = None,
    ):
        self.delays = delays_samples or [0, 3, 7]
        self.path_gains_db = path_gains_db or [0.0, -3.0, -10.0]
        assert len(self.delays) == len(self.path_gains_db)
        self.snr_db = snr_db
        self.doppler_hz = doppler_hz
        self.sample_rate = sample_rate
        self.rng = np.random.default_rng(seed)

        self._build_cir()

    def _build_cir(self):
        """Build channel impulse response (CIR) from path parameters."""
        max_delay = max(self.delays)
        self.cir = np.zeros(max_delay + 1, dtype=complex)
        for delay, gain_db in zip(self.delays, self.path_gains_db):
            gain_linear = 10 ** (gain_db / 20.0)
            phase = self.rng.uniform(0, 2 * np.pi)
            self.cir[delay] = gain_linear * np.exp(1j * phase)

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        # Apply multipath via convolution
        received = fftconvolve(signal, self.cir)[:len(signal)]

        # Add AWGN
        snr_linear = 10 ** (self.snr_db / 10.0)
        sig_power = np.mean(np.abs(received) ** 2)
        noise_power = sig_power / snr_linear
        N = len(received)
        noise = (
            self.rng.standard_normal(N) + 1j * self.rng.standard_normal(N)
        ) * np.sqrt(noise_power / 2)
        return received + noise

    @property
    def channel_impulse_response(self) -> np.ndarray:
        return self.cir.copy()

    def get_frequency_response(self, n_fft: int = 512) -> np.ndarray:
        """Compute frequency response of the channel (useful for OFDM equalization)."""
        return np.fft.fft(self.cir, n=n_fft)


class PropagationLoss:
    """
    Free-space path loss model (Friis equation).
    Useful for link budget calculations.

    Args:
        freq_hz:     Carrier frequency [Hz]
        tx_power_w:  Transmit power [Watts]
        tx_gain_db:  TX antenna gain [dBi]
        rx_gain_db:  RX antenna gain [dBi]
    """

    def __init__(
        self,
        freq_hz: float = 915e6,
        tx_power_w: float = 0.1,
        tx_gain_db: float = 0.0,
        rx_gain_db: float = 0.0,
    ):
        self.freq_hz = freq_hz
        self.tx_power_w = tx_power_w
        self.tx_gain_db = tx_gain_db
        self.rx_gain_db = rx_gain_db
        self.c = 3e8  # speed of light

    def path_loss_db(self, distance_m: float) -> float:
        """Free-space path loss in dB."""
        wavelength = self.c / self.freq_hz
        fspl = (4 * np.pi * distance_m / wavelength) ** 2
        return 10 * np.log10(max(fspl, 1e-30))

    def rx_power_dbm(self, distance_m: float) -> float:
        """Received power in dBm."""
        tx_dbm = 10 * np.log10(self.tx_power_w * 1000)
        return tx_dbm + self.tx_gain_db + self.rx_gain_db - self.path_loss_db(distance_m)

    def snr_db(self, distance_m: float, noise_figure_db: float = 5.0,
               bandwidth_hz: float = 200e3, temp_k: float = 290.0) -> float:
        """Estimated received SNR in dB."""
        k_b = 1.38e-23
        thermal_noise_dbm = 10 * np.log10(k_b * temp_k * bandwidth_hz * 1000)
        noise_floor_dbm = thermal_noise_dbm + noise_figure_db
        return self.rx_power_dbm(distance_m) - noise_floor_dbm
