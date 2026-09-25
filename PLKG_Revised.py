"""
PLKG BASELINE — Revised [13]
Wavelet + Gray code 5-bit + Level Crossing + Cascade
Perubahan dari versi asli:
  1. Chronological 60/20/20 split
  2. Gray code 5-bit menggantikan Gray 4-bit / 16-level
  3. Threshold dari training partition
  4. Agreed KGR ditambahkan
  5. T_akuisisi = (N-1) × 1 detik
"""

import pandas as pd
import numpy as np
import pywt, math, hashlib
from time import perf_counter
from scipy.special import erfc, gammaincc
from scipy.fft import fft
from scipy.stats import entropy as sp_entropy

FILE_PATH = "combined_v2i lurus 20.csv"   # ← ganti sesuai scenario
N_LEVELS  = 32
N_BITS    = 5
SERIAL_M  = 2

# ══════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════
df = pd.read_csv(FILE_PATH)
rssi_alice = df['rssi_alice'].values.astype(float)
rssi_bob   = df['rssi_bob'].values.astype(float)
N = len(rssi_alice)
T_AKUISISI = (N - 1) * 1.0

print("=" * 55)
print(f"FILE: {FILE_PATH}  |  N={N}  |  T_akuisisi={T_AKUISISI:.0f}s")
print(f"Corr raw: {np.corrcoef(rssi_alice,rssi_bob)[0,1]:.4f}"
      f"  |  Gap: {np.median(rssi_alice-rssi_bob):.2f} dB")
print("=" * 55)

# ══════════════════════════════════════════════════════════════════════
# 2. CHRONOLOGICAL SPLIT 60 / 20 / 20
# ══════════════════════════════════════════════════════════════════════
n_train = int(N * 0.60); n_val = int(N * 0.20)
A_train = rssi_alice[:n_train];       B_train = rssi_bob[:n_train]
A_test  = rssi_alice[n_train+n_val:]; B_test  = rssi_bob[n_train+n_val:]
print(f"Split → Train:{n_train} | Val:{n_val} | Test:{len(A_test)}")

# ══════════════════════════════════════════════════════════════════════
# 3. WAVELET DENOISING (threshold dari training)
# ══════════════════════════════════════════════════════════════════════
t_pre0 = perf_counter()

def wavelet_denoise(sig, wavelet='db4', level=2):
    sig = np.array(sig, dtype=float)
    if len(sig) < 2**(level+1): level = max(1, int(np.log2(len(sig)))-1)
    c   = pywt.wavedec(sig, wavelet, level=level)
    thr = (np.median(np.abs(c[-1]))/0.6745) * np.sqrt(2*np.log(len(sig)))
    nc  = [c[0]] + [pywt.threshold(x, thr, 'soft') for x in c[1:]]
    return pywt.waverec(nc, wavelet)[:len(sig)]

A_tr_den = wavelet_denoise(A_train)
B_tr_den = wavelet_denoise(B_train)
A_ts_den = wavelet_denoise(A_test)
B_ts_den = wavelet_denoise(B_test)
t_pre    = perf_counter() - t_pre0

# Threshold dari training partition
THR_MIN = A_tr_den.min(); THR_MAX = A_tr_den.max()

# ══════════════════════════════════════════════════════════════════════
# 4. GRAY CODE 5-BIT QUANTISASI (test partition)
# ══════════════════════════════════════════════════════════════════════
t_q0 = perf_counter()

def int_to_gray(n, bits=N_BITS):
    g = n ^ (n >> 1)
    return [(g >> (bits-1-i)) & 1 for i in range(bits)]

def gray_q(sig, lo=THR_MIN, hi=THR_MAX, nl=N_LEVELS, nb=N_BITS):
    step = (hi - lo) / nl; out = []
    for v in sig:
        lv = max(0, min(nl-1, int((v-lo)/step)))
        out.extend(int_to_gray(lv, nb))
    return np.array(out, dtype=int)

key_a_raw = gray_q(A_ts_den)
key_b_raw = gray_q(B_ts_den)

# ── Level Crossing Filter ────────────────────────────────────────────
def level_crossing(bits, m=2):
    """Pertahankan hanya run dengan m bit identik berturutan."""
    out = []; i = 0
    while i < len(bits) - m:
        seg = bits[i:i+m]
        if all(b == seg[0] for b in seg):
            out.extend(seg); i += m
        else:
            i += 1
    return np.array(out, dtype=int)

key_a_lc = level_crossing(key_a_raw)
key_b_lc = level_crossing(key_b_raw)

n_bits = min(len(key_a_lc), len(key_b_lc))
key_a  = key_a_lc[:n_bits]; key_b = key_b_lc[:n_bits]
t_q    = perf_counter() - t_q0

def kdr(a, b):
    n=min(len(a),len(b)); return float(np.sum(a[:n]!=b[:n])/n) if n else 1.0
def ent(bits):
    p1=np.mean(bits); p0=1-p1; h=0.0
    if p1>0: h-=p1*np.log2(p1)
    if p0>0: h-=p0*np.log2(p0)
    return h

T_quant  = T_AKUISISI + t_pre + t_q
kdr_q    = kdr(key_a, key_b)
kgr_q    = n_bits / T_quant
agreed_q = kgr_q * (1 - kdr_q)

print(f"\n--- QUANTISASI + LEVEL CROSSING ---")
print(f"Bits raw     : {len(key_a_raw)}")
print(f"Bits after LC: {n_bits}")
print(f"KDR          : {kdr_q:.5f}")
print(f"KGR (raw)    : {kgr_q:.4f} bps")
print(f"Agreed KGR   : {agreed_q:.4f} bps")

# ══════════════════════════════════════════════════════════════════════
# 5. CASCADE RECONCILIATION
# ══════════════════════════════════════════════════════════════════════
t_r0 = perf_counter()

def dyn_block(n):
    return 16 if n<=256 else 32 if n<=512 else 64 if n<=1024 else 128

def parity(b): return int(np.sum(b) % 2)

def bsearch(a_blk, b_blk, start, corr):
    l, r = 0, len(a_blk)-1
    while l <= r:
        if l == r:
            if a_blk[l] != b_blk[l]: corr[start+l] = a_blk[l]
            break
        m = (l+r)//2
        if parity(a_blk[l:m+1]) != parity(b_blk[l:m+1]): r = m
        else: l = m+1

def cascade(alice, bob):
    corr = bob.copy(); bs = dyn_block(len(alice))
    while True:
        for i in range(0, len(alice), bs):
            a_b=alice[i:i+bs]; b_b=corr[i:i+bs]
            if parity(a_b) != parity(b_b): bsearch(a_b,b_b,i,corr)
        mis = np.where(alice != corr)[0]
        if len(mis)==0 or bs==1: break
        bs = max(1, bs//2)
    return corr

key_corr = cascade(key_a, key_b)
t_recon  = perf_counter() - t_r0

T_recon  = T_quant + t_recon
kdr_r    = kdr(key_corr, key_b)
kgr_r    = n_bits / T_recon
agreed_r = kgr_r * (1 - kdr_r)

print(f"\n--- REKONSILIASI (CASCADE) ---")
print(f"KDR       : {kdr_r:.5f}")
print(f"BAR       : {1-kdr_r:.5f}")
print(f"KGR (raw) : {kgr_r:.4f} bps")
print(f"Agreed KGR: {agreed_r:.4f} bps")

# ── SHA-256 ──────────────────────────────────────────────────────────
t_pa0 = perf_counter()

def nist_f(b):
    n=len(b)
    if n<16: return 0.0
    return float(erfc(abs(sum(1 if x else -1 for x in b))/math.sqrt(2*n)))
def nist_c(b):
    n=len(b)
    if n<100: return 0.0
    x=np.array([1 if v else -1 for v in b]); s=np.cumsum(x); z=int(np.max(np.abs(s)))
    if z==0: return 1.0
    s1=sum(math.exp(-((4*k+1)**2*z**2)/(2*n)) for k in range(int(np.floor((-n/z+1)/4)),int(np.floor((n/z-1)/4))+1))
    s2=sum(math.exp(-((4*k+3)**2*z**2)/(2*n)) for k in range(int(np.floor((-n/z-3)/4)),int(np.floor((n/z-1)/4))+1))
    return float(max(0,min(1,1-s1+s2)))

freq_v=nist_f(key_corr); cum_v=nist_c(key_corr)
ha=hashlib.sha256(''.join(map(str,key_corr)).encode()).hexdigest()
hb=hashlib.sha256(''.join(map(str,key_b)).encode()).hexdigest()
pa_bits=len(bin(int(ha,16))[2:].zfill(256))
bit_pa = bin(int(ha, 16))[2:].zfill(256)
entro_pa=ent(np.array([int(b) for b in bit_pa]))
t_pa=perf_counter()-t_pa0
T_pa=T_recon+t_pa; kgr_pa=pa_bits/T_pa

print(f"\n--- SHA-256 ---")
print(f"Match : {ha==hb} | KGR: {kgr_pa:.4f} bps | Entropy: {ent(key_corr):.5f}")
print(f"Freq  : {freq_v:.5f} | Cumsum: {cum_v:.5f}")

print(f"\n{'='*55}\nRINGKASAN PLKG [13]")
print(f"  T_akuisisi : {T_AKUISISI:.0f}s")
print(f"  KDR quant  : {kdr_q:.5f}")
print(f"  Agreed KGR : {agreed_q:.4f} bps  |  Bits: {n_bits}")
print(f"  Entropy PA : {entro_pa:.5f}")
print(f"  T_preproc  : {t_pre*1000:.2f}ms  |  T_quant: {t_q*1000:.3f}ms")
print(f"  T_Recon    : {t_recon*1000:.4f}ms")
print(f"  T_Ampli    : {t_pa*1000:.4f}ms")
print(f"  T_Total    : {T_pa:.4f}s")
