# Wireless Communications Link with Software-Defined Radio
> A full software-defined radio (SDR) communications link: modulation, channel modeling, synchronization, FEC, and OTA transmission via USRP / ADALM-Pluto — designed for throughput, reliability, or security optimization.

## Overview

This project implements a **unidirectional digital wireless communications link** from bit source to decoded bits, including:

- Selectable modulation: **DPSK, QPSK, 16-QAM, 64-QAM, OFDM**
- Forward Error Correction: **Convolutional codes + Viterbi, LDPC, Reed-Solomon**
- Channel models: **AWGN, Rayleigh fading, multipath, Doppler**
- Receiver synchronization: **carrier, timing, frame** (Costas loop, Gardner TED, Schmidl-Cox)
- OTA transmission via **USRP** or **ADALM-Pluto** SDR hardware
- Optimization modes: **throughput | reliability | security**

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  TRANSMITTER                                                          │
│                                                                      │
│  [Source Bits] → [FEC Encoder] → [Modulator] → [Pulse Shaping]      │
│                                                  → [SDR TX / Channel]│
└──────────────────────────────────────────────────────────────────────┘
                              │ OTA / Simulated Channel
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│  RECEIVER                                                             │
│                                                                      │
│  [SDR RX] → [AGC] → [Carrier Sync] → [Timing Sync] → [Frame Sync]  │
│          → [Equalizer] → [Demodulator] → [FEC Decoder] → [Bits Out] │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Modulation Schemes

| Scheme | Bits/Symbol | Carrier Sync | Use Case |
|---|---|---|---|
| DPSK | 1 | None needed | Simplest, most robust |
| QPSK | 2 | Costas loop | Balanced throughput/reliability |
| 16-QAM | 4 | Costas loop | Higher throughput |
| 64-QAM | 6 | Costas loop | Maximum throughput |
| OFDM/QPSK | 2×N | Schmidl-Cox | Multipath resilience |

---

## Hardware Support

| Device | Interface | Status |
|---|---|---|
| ADALM-Pluto (PlutoSDR) | USB / libiio |  Supported |
| USRP B200/B210 | USB3 / UHD | Supported |
| HackRF One | USB / SoapySDR | Supported |
| Simulation only | — | No hardware needed |

---

## Installation

```bash
git clone https://github.com/yourusername/sdr-wireless-link.git
cd sdr-wireless-link

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .

# Optional: hardware drivers
# PlutoSDR: pip install pyadi-iio
# USRP:     install UHD + pip install uhd
```

---

## Usage

### Simulate link end-to-end
```bash
python scripts/simulate_link.py --config configs/qpsk_awgn.yaml
```

### BER sweep over SNR range
```bash
python scripts/ber_sweep.py --modulation QPSK --snr-min -5 --snr-max 30
```

### OTA transmission (ADALM-Pluto)
```bash
# Terminal 1 — Transmit
python scripts/transmit.py --config configs/qpsk_ota.yaml --device pluto

# Terminal 2 — Receive
python scripts/receive.py --config configs/qpsk_ota.yaml --device pluto
```

### Optimization mode
```bash
# Maximize throughput (64-QAM, minimal FEC)
python scripts/simulate_link.py --optimize throughput

# Maximize reliability (DPSK + strong LDPC)
python scripts/simulate_link.py --optimize reliability

# Maximize security (QPSK + AES payload encryption)
python scripts/simulate_link.py --optimize security
```

---

## Project Structure

```
sdr-wireless-link/
├── src/
│   ├── modulation/       # DPSK, QPSK, QAM, OFDM modulators/demodulators
│   ├── channel/          # AWGN, Rayleigh, multipath channel models
│   ├── sync/             # Costas loop, Gardner TED, Schmidl-Cox, frame sync
│   ├── coding/           # Convolutional, LDPC, Reed-Solomon FEC
│   ├── receiver/         # AGC, equalizer, full receiver pipeline
│   └── utils/            # Bit error rate, metrics, plotting, hardware I/O
├── configs/              # YAML experiment configs
├── scripts/              # CLI entry points
├── tests/                # Unit tests
├── notebooks/            # Interactive demos
├── docs/                 # Architecture, hardware setup, theory
└── .github/workflows/    # CI, simulation sweep, BER benchmark
```

---

## Evaluation Metrics

- **BER** — Bit Error Rate vs SNR curves
- **EVM** — Error Vector Magnitude (constellation quality)
- **Throughput** — Effective bits/second after FEC overhead
- **Synchronization latency** — frames to acquire lock
- **PSD** — Power Spectral Density plots

---

## References

1. Schmidl & Cox, "Robust Frequency and Timing Synchronization for OFDM," IEEE Trans. Commun., 1997
2. Proakis & Salehi, *Digital Communications*, 5th ed., McGraw-Hill
3. Rice, *Digital Communications: A Discrete-Time Approach*, Pearson

---

## License

MIT License — see [LICENSE](LICENSE)
