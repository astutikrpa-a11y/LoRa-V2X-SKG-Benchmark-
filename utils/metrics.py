"""
metrics.py — KDR, KGR, KAR, Shannon entropy calculations
Paper: Offset-Normalised SavGol for LoRa V2X Key Agreement
"""

import numpy as np


def gray_encode(n: int, bits: int = 5) -> list:
    """Convert integer to Gray-code bit list."""
    g = n ^ (n >> 1)
    return [(g >> (bits - 1 - i)) & 1 for i in range(bits)]


def quantise_gray(signal: np.ndarray, lo: float, hi: float,
                  n_levels: int = 32, n_bits: int = 5) -> np.ndarray:
    """
    32-level Gray-code quantisation.
    Args:
        signal: RSSI values to quantise
        lo, hi: quantisation range (from training partition)
        n_levels: number of quantisation levels (default 32)
        n_bits: bits per sample (default 5 for 32 levels)
    Returns:
        Binary bit array
    """
    step = (hi - lo) / n_levels
    bits = []
    for v in signal:
        level = max(0, min(n_levels - 1, int((v - lo) / step)))
        bits.extend(gray_encode(level, n_bits))
    return np.array(bits, dtype=int)


def kdr(alice_bits: np.ndarray, bob_bits: np.ndarray) -> float:
    """
    Key Disagreement Rate.
    KDR = number of bit mismatches / total bits
    """
    n = min(len(alice_bits), len(bob_bits))
    if n == 0:
        return 1.0
    return float(np.sum(alice_bits[:n] != bob_bits[:n]) / n)


def kgr(signal: np.ndarray, n_bits: int = 5,
         sampling_rate: float = 1.0) -> float:
    """
    Raw Key Generation Rate (bps).
    KGR = bits_per_sample × sampling_rate
    """
    return n_bits * sampling_rate


def agreed_kgr(kdr_val: float, raw_kgr: float) -> float:
    """
    Agreed Key Generation Rate.
    Agreed KGR = KGR × (1 − KDR)
    """
    return raw_kgr * (1 - kdr_val)


def kar(alice_bits: np.ndarray, bob_bits: np.ndarray) -> float:
    """
    Key Agreement Rate = 1 − KDR.
    Oracle-assisted upper bound (post-reconciliation).
    """
    return 1.0 - kdr(alice_bits, bob_bits)


def shannon_entropy(bits: np.ndarray) -> float:
    """
    Shannon entropy of bit string.
    H = −p1·log2(p1) − p0·log2(p0)
    Note: This reflects output bit distribution,
    NOT channel randomness or cryptographic security.
    """
    if len(bits) == 0:
        return 0.0
    p1 = float(np.mean(bits))
    p0 = 1.0 - p1
    if p1 == 0 or p0 == 0:
        return 0.0
    return -(p1 * np.log2(p1) + p0 * np.log2(p0))


def is_degenerate(bits: np.ndarray, threshold: float = 0.5) -> bool:
    """
    Check if bit string is degenerate (entropy < threshold).
    Degenerate = cryptographically useless despite possible KDR=0.
    """
    return shannon_entropy(bits) < threshold


def compute_all_metrics(alice_signal: np.ndarray,
                        bob_signal: np.ndarray,
                        alice_train: np.ndarray,
                        bob_train: np.ndarray,
                        n_levels: int = 32,
                        n_bits: int = 5) -> dict:
    """
    Compute all metrics for a given test partition.
    
    Args:
        alice_signal: Alice RSSI test partition (preprocessed)
        bob_signal: Bob RSSI test partition (preprocessed)
        alice_train: Alice RSSI training partition (for quantisation range)
        bob_train: Bob RSSI training partition (for quantisation range)
    
    Returns:
        Dictionary of all metrics
    """
    # Quantisation range from training partition
    lo = alice_train.min()
    hi = alice_train.max()

    # Quantise
    alice_bits = quantise_gray(alice_signal, lo, hi, n_levels, n_bits)
    bob_bits   = quantise_gray(bob_signal,   lo, hi, n_levels, n_bits)

    # Metrics
    kdr_val     = kdr(alice_bits, bob_bits)
    raw_kgr_val = kgr(alice_signal, n_bits, sampling_rate=1.0)
    agr_kgr     = agreed_kgr(kdr_val, raw_kgr_val)
    kar_val     = kar(alice_bits, bob_bits)
    entropy_val = shannon_entropy(alice_bits)
    degen       = is_degenerate(alice_bits)

    return {
        'KDR':          round(kdr_val, 6),
        'KAR':          round(kar_val, 6),
        'Raw_KGR_bps':  round(raw_kgr_val, 4),
        'Agreed_KGR_bps': round(agr_kgr, 4),
        'Entropy':      round(entropy_val, 6),
        'Degenerate':   degen,
        'N_bits':       len(alice_bits),
        'N_samples':    len(alice_signal),
    }
