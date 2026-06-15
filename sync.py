Receiver synchronization algorithms:
  - Costas Loop          : carrier frequency & phase recovery
  - Gardner TED          : symbol timing recovery
  - Schmidl-Cox          : OFDM coarse timing + frequency offset
  - Frame Synchronizer   : preamble correlation

import numpy as np
from typing import Tuple, Optional


class CostasLoop:
    """
    Second-order Costas loop for carrier phase and frequency recovery.
    Supports BPSK (M=2) and QPSK (M=4) modulations.

    Args:
        M:             Modulation order (2=BPSK, 4=QPSK)
        loop_bw:       Normalized loop bandwidth (BnT, typ. 0.01)
        damping:       Loop damping factor (typ. 0.707 = critically damped)
        sample_rate:   Sample rate [Hz]
    """

    def __init__(
        self,
        M: int = 4,
        loop_bw: float = 0.01,
        damping: float = 0.707,
        sample_rate: float = 1.0,
    ):
        self.M = M
        self.sample_rate = sample_rate

        # Compute loop filter coefficients (Gardner 1993)
        theta_n = loop_bw / (damping + 1.0 / (4.0 * damping))
        Kp = (4.0 * damping * theta_n) / (1.0 + 2.0 * damping * theta_n + theta_n**2)
        Ki = (4.0 * theta_n**2) / (1.0 + 2.0 * damping * theta_n + theta_n**2)
        self.alpha = Kp     # proportional gain
        self.beta  = Ki     # integral gain

        # Loop state
        self._phase = 0.0
        self._freq  = 0.0

    def process(self, samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process input samples through Costas loop.

        Returns:
            corrected: phase-corrected complex samples
            phase_err: phase error at each sample (for diagnostics)
        """
        N = len(samples)
        corrected = np.zeros(N, dtype=complex)
        phase_err = np.zeros(N)

        for i, sample in enumerate(samples):
            # Apply current phase correction
            corrected[i] = sample * np.exp(-1j * self._phase)

            # Compute phase error (QPSK uses M=4 phase detector)
            err = self._phase_detector(corrected[i])
            phase_err[i] = err

            # Update loop filter (PI controller)
            self._freq  += self.beta * err
            self._phase += self._alpha_update(err)
            self._phase  = self._phase % (2 * np.pi)

        return corrected, phase_err

    def _phase_detector(self, symbol: complex) -> float:
        """Decision-directed phase error detector."""
        if self.M == 2:
            return np.sign(symbol.real) * symbol.imag
        elif self.M == 4:
            return np.sign(symbol.real) * symbol.imag - np.sign(symbol.imag) * symbol.real
        else:
            raise ValueError(f"Costas loop only supports M=2 or M=4, got M={self.M}")

    def _alpha_update(self, err: float) -> float:
        return self.alpha * err + self._freq

    def reset(self):
        self._phase = 0.0
        self._freq  = 0.0

    @property
    def frequency_offset_hz(self) -> float:
        """Estimated carrier frequency offset in Hz."""
        return self._freq * self.sample_rate / (2 * np.pi)


class GardnerTED:
    """
    Gardner Timing Error Detector for symbol timing recovery.
    Works at 2 samples/symbol (sps=2).

    Args:
        sps:        Samples per symbol (must be 2)
        loop_bw:    Normalized timing loop bandwidth
        damping:    Loop damping factor
    """

    def __init__(self, sps: int = 2, loop_bw: float = 0.01, damping: float = 0.707):
        assert sps == 2, "Gardner TED requires sps=2"
        self.sps = sps

        theta_n = loop_bw / (damping + 1.0 / (4.0 * damping))
        self.alpha = (4.0 * damping * theta_n) / (1.0 + 2.0 * damping * theta_n + theta_n**2)
        self.beta  = (4.0 * theta_n**2) / (1.0 + 2.0 * damping * theta_n + theta_n**2)

        self._mu   = 0.0   # fractional timing offset [0, sps)
        self._freq = 0.0   # frequency correction

    def process(self, samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Downsample 2 sps → 1 sps with timing correction.

        Returns:
            symbols:   timing-corrected samples at 1 sps
            ted_error: timing error at each symbol
        """
        N = len(samples) // self.sps
        symbols = np.zeros(N, dtype=complex)
        errors  = np.zeros(N)

        for i in range(N):
            idx0 = i * self.sps
            idx1 = idx0 + 1

            if idx1 >= len(samples):
                break

            x0 = samples[idx0]
            x1 = samples[idx1]

            # Gardner TED: e[n] = Re{(x[n] - x[n-1]) * conj(x[n - 0.5])}
            if i > 0:
                err = (x1 - self._prev_x1).real * x0.real + \
                      (x1 - self._prev_x1).imag * x0.imag
            else:
                err = 0.0

            errors[i]  = err
            symbols[i] = x0

            self._freq += self.beta  * err
            self._mu   += self.alpha * err + self._freq
            self._mu    = self._mu % self.sps

            self._prev_x1 = x1

        return symbols, errors

    def reset(self):
        self._mu   = 0.0
        self._freq = 0.0


class SchmidlCox:
    """
    Schmidl-Cox OFDM timing and frequency offset estimator.

    The algorithm uses a preamble with two identical halves in frequency
    domain to estimate:
      - Coarse symbol timing (correlation peak)
      - Fractional and integer carrier frequency offset

    Reference: Schmidl & Cox, IEEE Trans. Commun., 1997.

    Args:
        n_fft:  OFDM FFT size
        cp_len: Cyclic prefix length
    """

    def __init__(self, n_fft: int = 64, cp_len: int = 16):
        self.n_fft = n_fft
        self.cp_len = cp_len
        self.L = n_fft // 2   # half-preamble length

    def generate_preamble(self) -> np.ndarray:
        """
        Generate Schmidl-Cox preamble: OFDM symbol with equal first and second halves.
        Even subcarriers carry random BPSK, odd subcarriers are zeroed.
        """
        rng = np.random.default_rng(42)  # fixed seed for known preamble
        pilot = (2 * rng.integers(0, 2, self.n_fft // 2) - 1).astype(complex)
        freq = np.zeros(self.n_fft, dtype=complex)
        freq[::2] = pilot                    # even subcarriers only
        time = np.fft.ifft(freq) * np.sqrt(self.n_fft)
        # Add cyclic prefix
        return np.concatenate([time[-self.cp_len :], time])

    def estimate(
        self, received: np.ndarray
    ) -> Tuple[int, float]:
        """
        Find timing offset and fractional CFO from received signal.

        Returns:
            timing_offset: sample index of OFDM symbol start
            cfo_normalized: fractional carrier frequency offset (cycles/sample)
        """
        L = self.L
        N = len(received)

        P = np.zeros(N, dtype=complex)   # correlation metric
        R = np.zeros(N)                   # energy metric
        M = np.zeros(N)                   # timing metric

        for d in range(N - 2 * L):
            # Correlation between first and second half of preamble
            P[d] = np.sum(
                np.conj(received[d : d + L]) * received[d + L : d + 2 * L]
            )
            # Energy of second half
            R[d] = np.sum(np.abs(received[d + L : d + 2 * L]) ** 2)
            M[d] = (np.abs(P[d]) ** 2) / (R[d] ** 2 + 1e-12)

        # Timing estimate: peak of M
        timing_offset = int(np.argmax(M))

        # CFO estimate: angle of correlation at timing peak
        cfo_normalized = np.angle(P[timing_offset]) / (2 * np.pi * L)

        return timing_offset, cfo_normalized


class FrameSynchronizer:
    """
    Preamble-based frame synchronizer using cross-correlation.

    Searches for a known preamble sequence in received samples to
    identify the start of each data frame.

    Args:
        preamble:       Known preamble symbol sequence
        threshold:      Normalized correlation threshold (0 to 1)
    """

    def __init__(self, preamble: np.ndarray, threshold: float = 0.7):
        self.preamble = np.asarray(preamble, dtype=complex)
        self.threshold = threshold
        self._preamble_energy = np.sum(np.abs(preamble) ** 2)

    def find_frame_start(
        self, received: np.ndarray
    ) -> Optional[int]:
        """
        Cross-correlate received signal with known preamble.

        Returns:
            Frame start index, or None if preamble not detected.
        """
        L = len(self.preamble)
        N = len(received)
        if N < L:
            return None

        best_corr = 0.0
        best_idx  = None

        for i in range(N - L + 1):
            segment = received[i : i + L]
            corr = np.abs(np.dot(np.conj(self.preamble), segment))
            seg_energy = np.sum(np.abs(segment) ** 2)
            norm_corr = corr / np.sqrt(self._preamble_energy * seg_energy + 1e-12)

            if norm_corr > best_corr:
                best_corr = norm_corr
                best_idx  = i

        if best_corr >= self.threshold:
            return best_idx
        return None
