"""
pipelines.py — All ablation preprocessing pipelines A1–A8
Paper: Offset-Normalised SavGol for LoRa V2X Key Agreement

A1: Raw RSSI (no preprocessing)
A2: Offset normalisation only                    ← essential first stage
A3: Offset + Wavelet denoising
A4: Offset + Moving Average
A5: Offset + Savitzky–Golay (PROPOSED METHOD ★)
A6: Offset + Kalman filter
A7: Offset + BiLSTM
A8: Offset + Wavelet + BiLSTM

Filter details (A5 — Proposed):
  scipy.signal.savgol_filter, w=5, p=2, mode='interp'
  Convolution weights: [−3, 12, 17, 12, −3] / 35
  Centered (two-sided): requires 2 future samples
  NOT strictly real-time without a 2-sample look-ahead buffer
"""

import numpy as np
from scipy.signal import savgol_filter
import pywt


# ── A2: Offset Normalisation ──────────────────────────────────────────────────
def compute_offset(alice_train: np.ndarray,
                   bob_train: np.ndarray) -> float:
    """
    Compute median-based offset estimate from training partition.
    δ = median{A_train − B_train} < 0  (since Alice < Bob numerically)
    """
    return float(np.median(alice_train - bob_train))


def apply_offset(alice: np.ndarray, delta: float) -> np.ndarray:
    """
    Apply offset correction to Alice.
    A_corr = A − δ  (adds positive quantity since δ < 0)
    This raises Alice RSSI values toward Bob's level.
    """
    return alice - delta


# ── A1: Raw RSSI ──────────────────────────────────────────────────────────────
def pipeline_A1(data: dict) -> dict:
    """A1: Raw RSSI — no preprocessing."""
    return {
        'alice_train': data['alice_train'].copy(),
        'alice_test':  data['alice_test'].copy(),
        'bob_train':   data['bob_train'].copy(),
        'bob_test':    data['bob_test'].copy(),
        'label': 'A1_Raw',
    }


# ── A2: Offset Only ───────────────────────────────────────────────────────────
def pipeline_A2(data: dict) -> dict:
    """A2: Offset normalisation only (essential first stage)."""
    delta = compute_offset(data['alice_train'], data['bob_train'])
    return {
        'alice_train': apply_offset(data['alice_train'], delta),
        'alice_test':  apply_offset(data['alice_test'], delta),
        'bob_train':   data['bob_train'].copy(),
        'bob_test':    data['bob_test'].copy(),
        'delta':       delta,
        'label': 'A2_Offset',
    }


# ── A3: Offset + Wavelet ──────────────────────────────────────────────────────
def pipeline_A3(data: dict, wavelet: str = 'db4',
                level: int = 3) -> dict:
    """A3: Offset + Wavelet denoising."""
    delta = compute_offset(data['alice_train'], data['bob_train'])
    a_tr  = apply_offset(data['alice_train'], delta)
    a_te  = apply_offset(data['alice_test'],  delta)

    def wavelet_denoise(sig):
        coeffs = pywt.wavedec(sig, wavelet, level=min(level, pywt.dwt_max_level(len(sig), wavelet)))
        thr    = np.median(np.abs(coeffs[-1])) / 0.6745 * np.sqrt(2 * np.log(len(sig)))
        coeffs_thr = [coeffs[0]] + [pywt.threshold(c, thr, 'soft') for c in coeffs[1:]]
        return pywt.waverec(coeffs_thr, wavelet)[:len(sig)]

    return {
        'alice_train': wavelet_denoise(a_tr),
        'alice_test':  wavelet_denoise(a_te),
        'bob_train':   wavelet_denoise(data['bob_train']),
        'bob_test':    wavelet_denoise(data['bob_test']),
        'delta': delta,
        'label': 'A3_Offset_Wavelet',
    }


# ── A4: Offset + Moving Average ───────────────────────────────────────────────
def pipeline_A4(data: dict, k: int = 3) -> dict:
    """A4: Offset + Moving Average (achieves lowest median KDR in ablation)."""
    delta = compute_offset(data['alice_train'], data['bob_train'])
    a_tr  = apply_offset(data['alice_train'], delta)
    a_te  = apply_offset(data['alice_test'],  delta)

    def movavg(sig):
        return np.convolve(sig, np.ones(k)/k, mode='same')

    return {
        'alice_train': movavg(a_tr),
        'alice_test':  movavg(a_te),
        'bob_train':   movavg(data['bob_train']),
        'bob_test':    movavg(data['bob_test']),
        'delta': delta,
        'label': 'A4_Offset_MovAvg',
    }


# ── A5: Offset + Savitzky–Golay (PROPOSED ★) ─────────────────────────────────
def pipeline_A5(data: dict, window: int = 5, polyorder: int = 2) -> dict:
    """
    A5: Offset + Savitzky–Golay smoothing — PROPOSED METHOD.

    Filter details:
      scipy.signal.savgol_filter(x, window_length=5, polyorder=2, mode='interp')
      Convolution weights: [−3, 12, 17, 12, −3] / 35
      Centered (two-sided): requires 2 future samples at each interior point.
      NOT strictly causal without a 2-sample look-ahead buffer.
      Causal SavGol implementation is identified as future work.

    Selected over A4 (Moving Average) based on smaller inter-condition IQR
    (0.1222 vs 0.1587) — a post hoc criterion with N=8, not statistically
    validated. Moving Average achieves lower median KDR (0.1302 vs 0.2417).
    """
    delta = compute_offset(data['alice_train'], data['bob_train'])
    a_tr  = apply_offset(data['alice_train'], delta)
    a_te  = apply_offset(data['alice_test'],  delta)

    def savgol(sig):
        w = min(window, len(sig))
        w = w if w % 2 == 1 else max(3, w - 1)
        return savgol_filter(sig, w, polyorder, mode='interp')

    return {
        'alice_train': savgol(a_tr),
        'alice_test':  savgol(a_te),
        'bob_train':   savgol(data['bob_train']),
        'bob_test':    savgol(data['bob_test']),
        'delta': delta,
        'savgol_window':    window,
        'savgol_polyorder': polyorder,
        'savgol_mode':      'interp',
        'label': 'A5_Offset_SavGol',
    }


# ── A6: Offset + Kalman ───────────────────────────────────────────────────────
def pipeline_A6(data: dict, Q: float = 1e-5,
                R: float = 0.1) -> dict:
    """A6: Offset + scalar Kalman filter."""
    delta = compute_offset(data['alice_train'], data['bob_train'])
    a_tr  = apply_offset(data['alice_train'], delta)
    a_te  = apply_offset(data['alice_test'],  delta)

    def kalman(sig, Q=Q, R=R):
        n   = len(sig)
        xh  = np.zeros(n)
        P   = np.zeros(n)
        xh[0], P[0] = sig[0], 1.0
        for i in range(1, n):
            xp  = xh[i-1]
            Pp  = P[i-1] + Q
            K   = Pp / (Pp + R)
            xh[i] = xp + K * (sig[i] - xp)
            P[i]  = (1 - K) * Pp
        return xh

    return {
        'alice_train': kalman(a_tr),
        'alice_test':  kalman(a_te),
        'bob_train':   kalman(data['bob_train']),
        'bob_test':    kalman(data['bob_test']),
        'delta': delta,
        'label': 'A6_Offset_Kalman',
    }


# ── A7 & A8: BiLSTM (negative findings) ──────────────────────────────────────
def pipeline_A7_A8(data: dict, use_wavelet: bool = False,
                   seed: int = 42) -> dict:
    """
    A7 (Offset+BiLSTM) and A8 (Offset+Wavelet+BiLSTM).
    NEGATIVE FINDINGS: KDR increases to 0.4150 under
    50 or fewer training samples (7/8 conditions).

    Requires separate BiLSTM training — see ablation/bilstm_model.py
    """
    label = 'A8_Offset_Wavelet_BiLSTM' if use_wavelet else 'A7_Offset_BiLSTM'
    raise NotImplementedError(
        f"{label}: BiLSTM training requires ablation/bilstm_model.py. "
        f"See run_ablation.py for full pipeline execution."
    )


# ── Pipeline registry ─────────────────────────────────────────────────────────
PIPELINES = {
    'A1': pipeline_A1,
    'A2': pipeline_A2,
    'A3': pipeline_A3,
    'A4': pipeline_A4,
    'A5': pipeline_A5,  # ← PROPOSED
    'A6': pipeline_A6,
}
