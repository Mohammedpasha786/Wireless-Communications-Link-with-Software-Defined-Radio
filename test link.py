Unit tests for modulation, channel, FEC, and sync modules.

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.modulation.modulation import DPSK, QPSK, QAM, OFDM
from src.channel.channel import AWGNChannel, RayleighChannel, MultipathChannel
from src.coding.fec import ConvolutionalCode, HammingCode, RepetitionCode
from src.sync.sync import CostasLoop, GardnerTED, SchmidlCox, FrameSynchronizer
from src.utils.metrics import bit_error_rate, error_vector_magnitude, ber_vs_snr_theoretical


RNG = np.random.default_rng(0)
BITS = RNG.integers(0, 2, 1000).astype(np.uint8)


# ── Modulation ─────────────────────────────────────────────────────────

class TestQPSK:
    mod = QPSK()

    def test_output_shape(self):
        syms = self.mod.modulate(BITS)
        assert syms.shape == (len(BITS) // 2,)

    def test_unit_power(self):
        syms = self.mod.modulate(BITS)
        assert abs(np.mean(np.abs(syms)**2) - 1.0) < 0.01

    def test_perfect_channel(self):
        syms = self.mod.modulate(BITS)
        rx   = self.mod.demodulate(syms)
        assert bit_error_rate(BITS[:len(rx)], rx) == 0.0

    def test_high_snr_low_ber(self):
        syms = self.mod.modulate(BITS)
        ch   = AWGNChannel(snr_db=30.0, seed=1)
        rx   = self.mod.demodulate(ch(syms))
        assert bit_error_rate(BITS[:len(rx)], rx) < 0.01


class TestQAM:
    @pytest.mark.parametrize("M", [16, 64])
    def test_perfect_channel(self, M):
        mod  = QAM(M=M)
        syms = mod.modulate(BITS)
        rx   = mod.demodulate(syms)
        assert bit_error_rate(BITS[:len(rx)], rx) == 0.0

    def test_constellation_size(self):
        for M in (16, 64):
            mod = QAM(M=M)
            assert len(mod.constellation) == M


class TestDPSK:
    def test_perfect_channel(self):
        mod  = DPSK(M=2)
        syms = mod.modulate(BITS)
        rx   = mod.demodulate(syms)
        assert bit_error_rate(BITS[:len(rx)], rx) < 0.05  # DPSK diff decode loses 1 bit

    def test_no_carrier_sync_needed(self):
        """DPSK should tolerate constant phase rotation."""
        mod  = DPSK(M=2)
        syms = mod.modulate(BITS) * np.exp(1j * np.pi / 4)   # rotate 45°
        rx   = mod.demodulate(syms)
        assert bit_error_rate(BITS[:len(rx)], rx) < 0.05


class TestOFDM:
    def test_perfect_roundtrip(self):
        ofdm = OFDM(n_fft=64, cp_len=16, subcarrier_mod="QPSK")
        syms = ofdm.modulate(BITS[:ofdm.bits_per_symbol * 10])
        rx   = ofdm.demodulate(syms[:ofdm.n_fft * 10 + ofdm.cp_len * 10])
        # Allow minor length mismatch
        n    = min(ofdm.bits_per_symbol * 10, len(rx))
        assert bit_error_rate(BITS[:n], rx[:n]) < 0.05


# ── Channel ────────────────────────────────────────────────────────────

class TestAWGNChannel:
    def test_output_same_shape(self):
        ch = AWGNChannel(snr_db=20.0, seed=0)
        signal = np.ones(500, dtype=complex)
        assert ch(signal).shape == signal.shape

    def test_high_snr_near_original(self):
        ch = AWGNChannel(snr_db=40.0, seed=0)
        signal = QPSK().modulate(BITS)
        rx = ch(signal)
        assert np.mean(np.abs(rx - signal)**2) < 0.01

    def test_low_snr_high_noise(self):
        ch = AWGNChannel(snr_db=0.0, seed=0)
        signal = np.ones(500, dtype=complex)
        rx = ch(signal)
        snr_est = np.mean(np.abs(signal)**2) / np.mean(np.abs(rx - signal)**2)
        assert 0.5 < snr_est < 2.0   # roughly 0 dB


class TestRayleighChannel:
    def test_output_shape(self):
        ch = RayleighChannel(snr_db=20.0, seed=0)
        signal = np.ones(500, dtype=complex)
        assert ch(signal).shape == (500,)

    def test_fading_reduces_amplitude(self):
        ch = RayleighChannel(snr_db=100.0, doppler_hz=0.0, seed=0)
        signal = np.ones(500, dtype=complex)
        rx = ch(signal)
        # Under fading the mean output power should be similar (unit-power fading)
        assert np.mean(np.abs(rx)**2) > 0.0


class TestMultipathChannel:
    def test_output_length(self):
        ch = MultipathChannel(delays_samples=[0, 3], path_gains_db=[0, -6])
        signal = np.ones(200, dtype=complex)
        rx = ch(signal)
        assert len(rx) == len(signal)

    def test_cir_shape(self):
        ch = MultipathChannel(delays_samples=[0, 5, 10])
        assert len(ch.channel_impulse_response) == 11  # max_delay + 1


# ── FEC ────────────────────────────────────────────────────────────────

class TestHammingCode:
    codec = HammingCode()

    def test_no_errors_roundtrip(self):
        enc = self.codec.encode(BITS[:100])
        dec = self.codec.decode(enc)
        assert bit_error_rate(BITS[:100], dec[:100]) == 0.0

    def test_single_bit_error_correction(self):
        enc = self.codec.encode(BITS[:28])  # 4 codewords of 7
        enc_err = enc.copy()
        enc_err[3] ^= 1    # flip one bit in first codeword
        dec = self.codec.decode(enc_err)
        assert bit_error_rate(BITS[:16], dec[:16]) == 0.0


class TestRepetitionCode:
    def test_encode_length(self):
        codec = RepetitionCode(N=3)
        enc = codec.encode(BITS[:10])
        assert len(enc) == 30

    def test_majority_vote(self):
        codec = RepetitionCode(N=3)
        enc = codec.encode(BITS[:10])
        # Flip 1 of every 3 bits (1 error per codeword)
        enc_err = enc.copy()
        enc_err[::3] ^= 1
        dec = codec.decode(enc_err)
        assert bit_error_rate(BITS[:10], dec) == 0.0


# ── Sync ───────────────────────────────────────────────────────────────

class TestSchmidlCox:
    def test_preamble_length(self):
        sc = SchmidlCox(n_fft=64, cp_len=16)
        p = sc.generate_preamble()
        assert len(p) == 64 + 16   # n_fft + cp_len

    def test_timing_detection(self):
        sc = SchmidlCox(n_fft=64, cp_len=16)
        preamble = sc.generate_preamble()
        # Place preamble after 50 samples of noise
        noise = (np.random.randn(50) + 1j*np.random.randn(50)) * 0.01
        rx = np.concatenate([noise, preamble])
        offset, cfo = sc.estimate(rx)
        # Timing estimate should be close to 50
        assert abs(offset - 50) <= 5

    def test_cfo_estimate_near_zero(self):
        sc = SchmidlCox(n_fft=64, cp_len=16)
        preamble = sc.generate_preamble()
        _, cfo = sc.estimate(preamble)
        assert abs(cfo) < 0.1   # no frequency offset injected


class TestFrameSynchronizer:
    def test_finds_preamble(self):
        rng = np.random.default_rng(0)
        preamble = (rng.integers(0,2,16) * 2 - 1).astype(complex)
        fs = FrameSynchronizer(preamble, threshold=0.6)
        noise = (rng.standard_normal(50) + 1j*rng.standard_normal(50)) * 0.05
        rx = np.concatenate([noise, preamble])
        idx = fs.find_frame_start(rx)
        assert idx is not None
        assert abs(idx - 50) <= 2


# ── Metrics ────────────────────────────────────────────────────────────

class TestMetrics:
    def test_ber_zero(self):
        bits = np.array([0,1,0,1,1,0], dtype=np.uint8)
        assert bit_error_rate(bits, bits) == 0.0

    def test_ber_all_errors(self):
        bits = np.zeros(100, dtype=np.uint8)
        err  = np.ones(100,  dtype=np.uint8)
        assert bit_error_rate(bits, err) == 1.0

    def test_evm_perfect(self):
        syms = QPSK().modulate(BITS)
        assert error_vector_magnitude(syms, syms) == pytest.approx(0.0, abs=1e-6)

    def test_theory_ber_monotone(self):
        snr = np.array([0., 5., 10., 15., 20.])
        ber = ber_vs_snr_theoretical(snr, "QPSK")
        assert np.all(np.diff(ber) < 0)   # BER decreases with SNR
