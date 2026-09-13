"""
partition.py — Chronological file-level data partitioning
Paper: Offset-Normalised SavGol for LoRa V2X Key Agreement

Two partition schemes:
  - 60/20/20: baseline comparison (Section 6.6)
  - 50/25/25: ablation study (Section 6.4)

Partitioning is CHRONOLOGICAL (not random).
Floor rounding applied per-file.
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ── Per-file partition counts (Table 6 in paper) ─────────────────────────────
# Computed as: tr=floor(N*frac_tr), va=floor(N*frac_va), te=N-tr-va
PARTITION_COUNTS = {
    'V2I_Turning_20':  {'N': 35,  '60_20_20': (21,7,7),   '50_25_25': (17,8,10)},
    'V2I_Turning_30':  {'N': 50,  '60_20_20': (30,10,10),  '50_25_25': (25,12,13)},
    'V2I_Straight_20': {'N': 80,  '60_20_20': (48,16,16),  '50_25_25': (40,20,20)},
    'V2I_Straight_30': {'N': 89,  '60_20_20': (53,17,19),  '50_25_25': (44,22,23)},
    'V2V_Opposite_20': {'N': 26,  '60_20_20': (15,5,6),    '50_25_25': (13,6,7)},
    'V2V_Opposite_30': {'N': 19,  '60_20_20': (11,3,5),    '50_25_25': (9,4,6)},
    'V2V_Same_20':     {'N': 102, '60_20_20': (61,20,21),  '50_25_25': (51,25,26)},
    'V2V_Same_30':     {'N': 94,  '60_20_20': (56,18,20),  '50_25_25': (47,23,24)},
}


def get_split_indices(n: int, scheme: str = '50_25_25') -> dict:
    """
    Compute chronological split indices for a file of size n.
    
    Args:
        n: total number of samples in file
        scheme: '60_20_20' or '50_25_25'
    
    Returns:
        dict with 'train', 'val', 'test' index ranges
    """
    if scheme == '60_20_20':
        tr = int(n * 0.60)
        va = int(n * 0.20)
    elif scheme == '50_25_25':
        tr = int(n * 0.50)
        va = int(n * 0.25)
    else:
        raise ValueError(f"Unknown scheme: {scheme}. Use '60_20_20' or '50_25_25'")

    te = n - tr - va

    return {
        'train': (0, tr),
        'val':   (tr, tr + va),
        'test':  (tr + va, n),
        'n_train': tr,
        'n_val':   va,
        'n_test':  te,
    }


def load_and_split(filepath: str, scheme: str = '50_25_25') -> dict:
    """
    Load a CSV file and apply chronological partitioning.
    
    Args:
        filepath: path to CSV file with columns rssi_alice, rssi_bob
        scheme: '60_20_20' or '50_25_25'
    
    Returns:
        dict with alice and bob arrays for train/val/test partitions
    """
    df = pd.read_csv(filepath)
    alice = df['rssi_alice'].values.astype(float)
    bob   = df['rssi_bob'].values.astype(float)
    n     = len(alice)

    idx = get_split_indices(n, scheme)
    tr_s, tr_e = idx['train']
    va_s, va_e = idx['val']
    te_s, te_e = idx['test']

    return {
        'alice_train': alice[tr_s:tr_e],
        'alice_val':   alice[va_s:va_e],
        'alice_test':  alice[te_s:te_e],
        'bob_train':   bob[tr_s:tr_e],
        'bob_val':     bob[va_s:va_e],
        'bob_test':    bob[te_s:te_e],
        'n_total':     n,
        'n_train':     idx['n_train'],
        'n_val':       idx['n_val'],
        'n_test':      idx['n_test'],
        'scheme':      scheme,
    }


def print_partition_table():
    """Print Table 6 from the paper: per-file partition counts."""
    print(f"\n{'='*75}")
    print("Table 6: Per-file Data Partition Counts")
    print(f"{'='*75}")
    print(f"{'Scenario':<22} {'N':>4} | "
          f"{'Tr(60%)':>7} {'Va(20%)':>7} {'Te(20%)':>7} | "
          f"{'Tr(50%)':>7} {'Va(25%)':>7} {'Te(25%)':>7}")
    print("-"*75)
    for cond, info in PARTITION_COUNTS.items():
        n = info['N']
        tr60, va60, te60 = info['60_20_20']
        tr50, va50, te50 = info['50_25_25']
        flag = '*' if tr50 > 50 else ' '
        print(f"{cond:<22} {n:>4} | "
              f"{tr60:>7} {va60:>7} {te60:>7} | "
              f"{tr50:>6}{flag} {va50:>7} {te50:>7}")
    print("-"*75)
    print("* Only condition with training > 50 under 50/25/25 scheme")
    print(f"{'='*75}\n")


if __name__ == '__main__':
    print_partition_table()
