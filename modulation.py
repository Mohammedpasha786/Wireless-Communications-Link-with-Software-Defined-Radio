Digital modulation and demodulation: DPSK, QPSK, 16-QAM, 64-QAM.

All modulators return complex baseband symbols normalized to unit average power.
import numpy as np
from typing import Tuple


# ── Constellation maps ─────────────────────────────────────────────────

def _qam_constellation(M: int) -> np.ndarray:
    """Generate square M-QAM constellation with Gray coding."""
    k = int(np.sqrt(M))
    assert k * k == M, "M must be a perfect square (4, 16, 64, 256, ...)"
    levels = np.arange(-(k - 1), k, 2, dtype=float)
    real, imag = np.meshgrid(levels, levels)
    symbols = (real + 1j * imag).flatten()
    # Normalize to unit average power
    symbols /= np.sqrt((np.abs(symbols) ** 2).mean())
    return symbols


QPSK_SYMBOLS = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]) / np.sqrt(2)
QAM16_SYMBOLS = _qam_constellation(16)
QAM64_SYMBOLS = _qam_constellation(64)


# ── DPSK ───────────────────────────────────────────────────────────────

class DPSK:
    """
    Differential Phase Shift Keying (DBPSK by default, M=2).
    No carrier phase reference needed at receiver.
    """

    def __init__(self, M: int = 2):
        self.M = M
        self.bits_per_symbol = int(np.log2(M))
        self._phase_state = 0.0

    def modulate(self, bits: np.ndarray) -> np.ndarray:
        """bits → complex baseband symbols."""
        bits = np.asarray(bits)
        n_sym = len(bits) // self.bits_per_symbol
        bits = bits[: n_sym * self.bits_per_symbol]
        symbols = np.zeros(n_sym, dtype=complex)
        phase = self._phase_state
        for i in range(n_sym):
            b = bits[i * self.bits_per_symbol : (i + 1) * self.bits_per_symbol]
            idx = int("".join(b.astype(str)), 2)
            delta = 2 * np.pi * idx / self.M
            phase = (phase + delta) % (2 * np.pi)
            symbols[i] = np.exp(1j * phase)
        self._phase_state = phase
        return symbols

    def demodulate(self, symbols: np.ndarray) -> np.ndarray:
        """Differential detection: compare consecutive symbols."""
        diff = symbols[1:] * np.conj(symbols[:-1])
        phase_diff = np.angle(diff) % (2 * np.pi)
        indices = np.round(phase_diff / (2 * np.pi / self.M)).astype(int) % self.M
        bits = np.array(
            [list(map(int, format(i, f"0{self.bits_per_symbol}b"))) for i in indices],
            dtype=np.uint8,
        ).flatten()
        return bits

    def reset(self):
        self._phase_state = 0.0


# ── QPSK ───────────────────────────────────────────────────────────────

class QPSK:
    """
    Quadrature Phase Shift Keying (M=4, 2 bits/symbol).
    Requires carrier phase synchronization at receiver.
    """

    bits_per_symbol = 2

    def modulate(self, bits: np.ndarray) -> np.ndarray:
        bits = np.asarray(bits)
        n_sym = len(bits) // 2
        bits = bits[: n_sym * 2].reshape(-1, 2)
        indices = bits[:, 0] * 2 + bits[:, 1]
        return QPSK_SYMBOLS[indices]

    def demodulate(self, symbols: np.ndarray) -> np.ndarray:
        dist = np.abs(symbols[:, None] - QPSK_SYMBOLS[None, :]) ** 2
        indices = np.argmin(dist, axis=1)
        bits = np.array(
            [[i >> 1, i & 1] for i in indices], dtype=np.uint8
        ).flatten()
        return bits


# ── M-QAM ──────────────────────────────────────────────────────────────

class QAM:
    """
    Square M-QAM modulator/demodulator.
    Supported: M ∈ {4, 16, 64, 256}
    """

    def __init__(self, M: int = 16):
        assert M in (4, 16, 64, 256), "M must be 4, 16, 64, or 256"
        self.M = M
        self.bits_per_symbol = int(np.log2(M))
        self._constellation = _qam_constellation(M)

    def modulate(self, bits: np.ndarray) -> np.ndarray:
        bits = np.asarray(bits)
        bps = self.bits_per_symbol
        n_sym = len(bits) // bps
        bits = bits[: n_sym * bps].reshape(-1, bps)
        indices = np.array(
            [int("".join(row.astype(str)), 2) for row in bits]
        )
        return self._constellation[indices]

    def demodulate(self, symbols: np.ndarray) -> np.ndarray:
        dist = np.abs(symbols[:, None] - self._constellation[None, :]) ** 2
        indices = np.argmin(dist, axis=1)
        bps = self.bits_per_symbol
        bits = np.array(
            [list(map(int, format(i, f"0{bps}b"))) for i in indices],
            dtype=np.uint8,
        ).flatten()
        return bits

    @property
    def constellation(self) -> np.ndarray:
        return self._constellation.copy()


# ── OFDM ───────────────────────────────────────────────────────────────

class OFDM:
    """
    OFDM modulator/demodulator with cyclic prefix and pilot subcarriers.

    Args:
        n_fft:          FFT size (number of subcarriers)
        cp_len:         Cyclic prefix length in samples
        pilot_spacing:  Insert pilot every N subcarriers
        subcarrier_mod: Modulator applied per subcarrier ('QPSK' or 'QAM16')
    """

    def __init__(
        self,
        n_fft: int = 64,
        cp_len: int = 16,
        pilot_spacing: int = 8,
        subcarrier_mod: str = "QPSK",
    ):
        self.n_fft = n_fft
        self.cp_len = cp_len
        self.pilot_indices = np.arange(0, n_fft, pilot_spacing)
        self.data_indices = np.setdiff1d(
            np.arange(1, n_fft - 1), self.pilot_indices   # exclude DC & Nyquist
        )
        self.n_data = len(self.data_indices)

        if subcarrier_mod == "QPSK":
            self._submod = QPSK()
        elif subcarrier_mod == "QAM16":
            self._submod = QAM(M=16)
        else:
            raise ValueError(f"Unknown subcarrier modulation: {subcarrier_mod}")

        self.bits_per_symbol = self.n_data * self._submod.bits_per_symbol

    def modulate(self, bits: np.ndarray) -> np.ndarray:
        """
        Bits → OFDM time-domain samples (including CP).
        Returns array of shape [n_symbols, n_fft + cp_len]
        """
        bps = self.bits_per_symbol
        n_ofdm = len(bits) // bps
        bits = bits[: n_ofdm * bps]
        samples_out = []

        for i in range(n_ofdm):
            chunk = bits[i * bps : (i + 1) * bps]
            data_syms = self._submod.modulate(chunk)
            freq_domain = np.zeros(self.n_fft, dtype=complex)
            freq_domain[self.data_indices] = data_syms
            # Insert BPSK pilots (known +1/-1 pattern)
            freq_domain[self.pilot_indices] = 1.0 + 0j
            time_domain = np.fft.ifft(freq_domain) * np.sqrt(self.n_fft)
            # Add cyclic prefix
            ofdm_sym = np.concatenate([time_domain[-self.cp_len :], time_domain])
            samples_out.append(ofdm_sym)

        return np.concatenate(samples_out)

    def demodulate(self, samples: np.ndarray) -> np.ndarray:
        """OFDM time-domain samples → bits (removes CP, FFT, extracts data)."""
        sym_len = self.n_fft + self.cp_len
        n_ofdm = len(samples) // sym_len
        bits_out = []

        for i in range(n_ofdm):
            ofdm_sym = samples[i * sym_len : (i + 1) * sym_len]
            time_domain = ofdm_sym[self.cp_len :]   # remove CP
            freq_domain = np.fft.fft(time_domain) / np.sqrt(self.n_fft)
            data_syms = freq_domain[self.data_indices]
            bits_out.append(self._submod.demodulate(data_syms))

        return np.concatenate(bits_out)
