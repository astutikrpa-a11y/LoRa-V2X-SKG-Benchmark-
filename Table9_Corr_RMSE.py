"""
Table 9 — Corr and RMSE columns (A1–A6), 50/25/25 chronological split.

Uses exactly the same preprocessing functions and split as
Ablation_SavGol_Final.py and reports, per configuration, the median across the
eight conditions of:
  - Pearson correlation between processed Alice and Bob test sequences
  - RMSE (dBm) between processed Alice and Bob test sequences

Run from the folder that contains the eight CSV files:
  python Table9_Corr_RMSE.py
"""
import warnings
import numpy as np
import pandas as pd
import pywt
from scipy.signal import savgol_filter

warnings.filterwarnings("ignore")

FILES = [
    'combined_v2i lurus 20.csv', 'combined_v2i lurus 30.csv',
    'combined_v2i belok 20.csv', 'combined_v2i belok 30.csv',
    'combined_V2V berlawanan20.csv', 'combined_V2V berlawanan30.csv',
    'combined_V2V searah20.csv', 'combined_V2V searah30.csv',
]
TRAIN, VAL = 0.50, 0.25   # chronological 50/25/25


def wavelet_den(sig):
    sig = np.array(sig, dtype=float)
    if len(sig) < 4:
        return sig
    c = pywt.wavedec(sig, 'db4', level=1)
    thr = (np.median(np.abs(c[-1])) / 0.6745) * np.sqrt(2 * np.log(max(len(sig), 2)))
    nc = [c[0]] + [pywt.threshold(x, thr, 'soft') for x in c[1:]]
    return pywt.waverec(nc, 'db4')[:len(sig)]


def savgol_den(sig, w=5, p=2):
    sig = np.array(sig, dtype=float); n = len(sig)
    ww = min(w, n); ww = ww if ww % 2 == 1 else max(3, ww - 1)
    return savgol_filter(sig, ww, min(p, ww - 1)) if n >= ww else sig


def kalman_fn(sig, Q=1e-3, R=0.1):
    x = float(sig[0]); P = 1.0; out = []
    for z in sig:
        P += Q; K = P / (P + R); x += K * (z - x); P = (1 - K) * P; out.append(x)
    return np.array(out)


def moving_avg(sig, k=3):
    return np.convolve(sig, np.ones(k) / k, 'same')


SMOOTH = {'A3': wavelet_den, 'A4': moving_avg, 'A5': savgol_den, 'A6': kalman_fn}
LABEL = {'A1': 'Raw RSSI', 'A2': 'Offset norm', 'A3': 'Offset+Wavelet',
         'A4': 'Offset+MovAvg', 'A5': 'Offset+SavGol', 'A6': 'Offset+Kalman'}

print(f"{'ID':<4}{'Configuration':<18}{'Corr':>8}{'RMSE (dBm)':>12}")
for cfg in LABEL:
    corr, rmse = [], []
    for f in FILES:
        df = pd.read_csv(f)
        A = df['rssi_alice'].values.astype(float)
        B = df['rssi_bob'].values.astype(float)
        n = len(A); tr = int(n * TRAIN); va = int(n * VAL)
        a_tr, b_tr, a_ts, b_ts = A[:tr], B[:tr], A[tr + va:], B[tr + va:]
        if cfg != 'A1':                       # offset from training partition only
            a_ts = a_ts - np.median(a_tr - b_tr)
        if cfg in SMOOTH:
            a_ts, b_ts = SMOOTH[cfg](a_ts), SMOOTH[cfg](b_ts)
        corr.append(float(np.corrcoef(a_ts, b_ts)[0, 1]))
        rmse.append(float(np.sqrt(np.mean((a_ts - b_ts) ** 2))))
    print(f"{cfg:<4}{LABEL[cfg]:<18}{np.median(corr):>8.3f}{np.median(rmse):>12.3f}")
