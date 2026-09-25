"""
PROPOSED METHOD — FINAL VERSION
Offset Normalisation + Savitzky-Golay + Gray Code 5-bit
+ Cascade Reconciliation + SHA-256

Pipeline:
  Raw RSSI
    → Offset Normalisation  (median dari training partition)
    → Savitzky-Golay Filter (window=5, polyorder=2)
    → Gray Code 5-bit Quantisation (32 levels, threshold dari train)
    → Cascade Reconciliation
    → SHA-256 Post-processing

Keunggulan vs baseline:
  - KDR median 0.252 (IQR 0.086) — paling konsisten lintas 8 skenario
  - Tidak ada training time (tidak ada ML model)
  - Preprocessing < 1 ms
  - Semua test samples dipakai (tidak ada window loss)
"""

import pandas as pd
import numpy as np
import hashlib
import math
import warnings
from time import perf_counter
from scipy.signal import savgol_filter
from scipy.special import erfc, gammaincc
from scipy.fft import fft

warnings.filterwarnings('ignore')

# ─── KONFIGURASI ──────────────────────────────────────────────────
FILE_PATH   = "combined_v2i lurus 20.csv"   # ← ganti sesuai scenario
N_LEVELS    = 32       # 32 level quantisasi
N_BITS      = 5        # log2(32) = 5 bit per sampel
SAVGOL_WIN  = 5        # window length Savitzky-Golay
SAVGOL_POLY = 2        # polynomial order
SERIAL_M    = 2

# ══════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════
df         = pd.read_csv(FILE_PATH)
rssi_alice = df['rssi_alice'].values.astype(float)
rssi_bob   = df['rssi_bob'].values.astype(float)
N          = len(rssi_alice)
T_AKUISISI = (N - 1) * 1.0

print("=" * 65)
print(f"FILE         : {FILE_PATH}")
print(f"Total sampel : {N}")
print(f"T_akuisisi   : {T_AKUISISI:.1f} detik ({N}-1 × 1.0s)")
print(f"Corr (raw)   : {np.corrcoef(rssi_alice, rssi_bob)[0,1]:.4f}")
print(f"Gap median   : {np.median(rssi_alice - rssi_bob):.2f} dB")
print(f"  Alice chip : RFM95  → RSSI = -164 + RegValue")
print(f"  Bob chip   : SX1262 → RSSI = -(Value/2)")
print(f"  Gap disebabkan perbedaan formula RSSI antar chip")
print("=" * 65)

# ══════════════════════════════════════════════════════════════════
# 2. CHRONOLOGICAL SPLIT 60 / 20 / 20
# ══════════════════════════════════════════════════════════════════
n_train = int(N * 0.60)
n_val   = int(N * 0.20)

A_train = rssi_alice[:n_train]
B_train = rssi_bob[:n_train]
A_val   = rssi_alice[n_train : n_train + n_val]
B_val   = rssi_bob[n_train : n_train + n_val]
A_test  = rssi_alice[n_train + n_val:]
B_test  = rssi_bob[n_train + n_val:]

print(f"\nSplit → Train:{n_train} | Val:{n_val} | Test:{len(A_test)}")

# ══════════════════════════════════════════════════════════════════
# 3. PRE-PROCESSING STAGE
# ══════════════════════════════════════════════════════════════════
t_pre0 = perf_counter()

# ── Stage 1: Offset Normalisation ─────────────────────────────────
# Dihitung HANYA dari training partition (tidak bocor ke test)
# Menggunakan median karena lebih robust terhadap RSSI spike
median_offset = np.median(A_train - B_train)
mean_offset   = np.mean(A_train - B_train)

print(f"\n--- STAGE 1: OFFSET NORMALISATION ---")
print(f"Median offset (train) : {median_offset:.3f} dB")
print(f"Mean offset   (train) : {mean_offset:.3f} dB")
print(f"Menggunakan           : MEDIAN (lebih robust)")

A_train_corr = A_train - median_offset
A_val_corr   = A_val   - median_offset
A_test_corr  = A_test  - median_offset

corr_raw    = np.corrcoef(rssi_alice, rssi_bob)[0,1]
corr_offset = np.corrcoef(A_test_corr, B_test)[0,1]
print(f"Corr sebelum offset   : {corr_raw:.4f}")
print(f"Corr setelah offset   : {corr_offset:.4f}")

# ── Stage 2: Savitzky-Golay Smoothing ─────────────────────────────
# Window adaptif sesuai ukuran data
def savgol_adaptive(signal, win=SAVGOL_WIN, poly=SAVGOL_POLY):
    signal = np.array(signal, dtype=float)
    n      = len(signal)
    # Window harus ganjil dan <= panjang sinyal
    w = min(win, n)
    w = w if w % 2 == 1 else max(3, w - 1)
    if n < w:
        return signal   # terlalu pendek, skip
    return savgol_filter(signal, window_length=w, polyorder=min(poly, w-1))

print(f"\n--- STAGE 2: SAVITZKY-GOLAY SMOOTHING ---")
print(f"Window length : {SAVGOL_WIN} (adaptif jika data kecil)")
print(f"Polyorder     : {SAVGOL_POLY}")

A_tr_sg = savgol_adaptive(A_train_corr)
B_tr_sg = savgol_adaptive(B_train)
A_ts_sg = savgol_adaptive(A_test_corr)
B_ts_sg = savgol_adaptive(B_test)

corr_sg = np.corrcoef(A_ts_sg, B_ts_sg)[0,1]
print(f"Corr setelah SavGol   : {corr_sg:.4f}")
print(f"Improvement vs raw    : {corr_sg - corr_raw:+.4f}")

t_pre = perf_counter() - t_pre0

# ── Threshold dari training partition ─────────────────────────────
THR_MIN = A_tr_sg.min()
THR_MAX = A_tr_sg.max()
print(f"Threshold (train)     : [{THR_MIN:.2f}, {THR_MAX:.2f}] dBm")

# ══════════════════════════════════════════════════════════════════
# 4. GRAY CODE 5-BIT QUANTISATION
# ══════════════════════════════════════════════════════════════════
def int_to_gray(n, bits=N_BITS):
    """Konversi integer ke Gray code — adjacent levels differ by 1 bit."""
    gray = n ^ (n >> 1)
    return [(gray >> (bits - 1 - i)) & 1 for i in range(bits)]

def gray_quantise(signal, lo=THR_MIN, hi=THR_MAX,
                  n_levels=N_LEVELS, n_bits=N_BITS):
    step     = (hi - lo) / n_levels
    bits_out = []
    for v in signal:
        lv = int((v - lo) / step)
        lv = max(0, min(n_levels - 1, lv))
        bits_out.extend(int_to_gray(lv, n_bits))
    return np.array(bits_out, dtype=int)

t_q0 = perf_counter()
bits_alice = gray_quantise(A_ts_sg)
bits_bob   = gray_quantise(B_ts_sg)
t_q        = perf_counter() - t_q0

n_bits_total = min(len(bits_alice), len(bits_bob))
bits_alice   = bits_alice[:n_bits_total]
bits_bob     = bits_bob[:n_bits_total]

print(f"\n--- STAGE 3: GRAY CODE QUANTISATION ---")
print(f"Test samples  : {len(A_test)} (semua, tanpa window loss)")
print(f"Total bits    : {n_bits_total} ({len(A_test)} × {N_BITS})")

# ══════════════════════════════════════════════════════════════════
# 5. HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════
def compute_kdr(a, b):
    n = min(len(a), len(b))
    return float(np.sum(a[:n] != b[:n]) / n) if n > 0 else 1.0

def compute_entropy(bits):
    bits = np.array(bits, dtype=int)
    p1   = np.mean(bits); p0 = 1.0 - p1
    h    = 0.0
    if p1 > 0: h -= p1 * np.log2(p1)
    if p0 > 0: h -= p0 * np.log2(p0)
    return h

# ══════════════════════════════════════════════════════════════════
# 6. METRICS — QUANTISATION
# ══════════════════════════════════════════════════════════════════
T_quant     = T_AKUISISI + t_pre + t_q
kdr_quant   = compute_kdr(bits_alice, bits_bob)
kgr_quant   = n_bits_total / T_quant
agreed_q    = kgr_quant * (1 - kdr_quant)
entropy_q   = compute_entropy(bits_alice)

print(f"\n--- HASIL QUANTISASI ---")
print(f"KDR          : {kdr_quant:.5f}")
print(f"BAR          : {1 - kdr_quant:.5f}")
print(f"KGR (raw)    : {kgr_quant:.4f} bps")
print(f"Agreed KGR   : {agreed_q:.4f} bps")
print(f"Entropy      : {entropy_q:.6f}")
print(f"Bits         : {n_bits_total}")

# ══════════════════════════════════════════════════════════════════
# 7. CASCADE RECONCILIATION
# ══════════════════════════════════════════════════════════════════
t_recon0 = perf_counter()

def parity(block): return int(np.sum(block) % 2)

def bisect_correct(a_blk, b_blk, start, corrected):
    l, r = 0, len(a_blk) - 1
    while l <= r:
        if l == r:
            if a_blk[l] != b_blk[l]:
                corrected[start + l] = a_blk[l]
            break
        m = (l + r) // 2
        if parity(a_blk[l:m+1]) != parity(b_blk[l:m+1]): r = m
        else: l = m + 1

def cascade(alice, bob):
    corrected  = bob.copy()
    block_size = 16 if len(alice)<=256 else 32 if len(alice)<=512 else 64
    for _ in range(4):   # max 4 passes
        for i in range(0, len(alice), block_size):
            ab = alice[i:i+block_size]
            bb = corrected[i:i+block_size]
            if parity(ab) != parity(bb):
                bisect_correct(ab, bb, i, corrected)
        if np.sum(alice != corrected) == 0: break
        block_size = max(1, block_size // 2)
    return corrected

bits_corr    = cascade(bits_alice, bits_bob)
t_recon      = perf_counter() - t_recon0
kdr_recon    = compute_kdr(bits_corr, bits_bob)
T_recon      = T_quant + t_recon
kgr_recon    = n_bits_total / T_recon
agreed_r     = kgr_recon * (1 - kdr_recon)
entropy_recon = compute_entropy(bits_corr)

print(f"\n--- HASIL REKONSILIASI (CASCADE) ---")
print(f"KDR          : {kdr_recon:.5f}")
print(f"BAR          : {1 - kdr_recon:.5f}")
print(f"KGR (raw)    : {kgr_recon:.4f} bps")
print(f"Agreed KGR   : {agreed_r:.4f} bps")
print(f"Entropy      : {entropy_recon:.6f}")
print(f"Recon time   : {t_recon*1000:.4f} ms")

# ══════════════════════════════════════════════════════════════════
# 8. NIST TESTS (supplementary)
# ══════════════════════════════════════════════════════════════════
t_pa0 = perf_counter()

def nist_freq(b):
    n=len(b)
    if n<16: return 0.0
    s=sum(1 if x else -1 for x in b)
    return float(erfc(abs(s)/math.sqrt(2*n)))

def nist_runs(b):
    n=len(b)
    if n<100: return 0.0
    pi=sum(int(x) for x in b)/n; tau=2/math.sqrt(n)
    if abs(pi-0.5)>=tau: return 0.0
    v=1+sum(1 for i in range(1,n) if b[i]!=b[i-1])
    num=abs(v-2*n*pi*(1-pi)); den=2*math.sqrt(2*n)*pi*(1-pi)
    return 0.0 if den==0 else float(erfc(num/den))

def nist_cumsum(b):
    n=len(b)
    if n<100: return 0.0
    x=np.array([1 if v else -1 for v in b]); s=np.cumsum(x); z=int(np.max(np.abs(s)))
    if z==0: return 1.0
    s1=sum(math.exp(-((4*k+1)**2*z**2)/(2*n)) for k in range(int(np.floor((-n/z+1)/4)),int(np.floor((n/z-1)/4))+1))
    s2=sum(math.exp(-((4*k+3)**2*z**2)/(2*n)) for k in range(int(np.floor((-n/z-3)/4)),int(np.floor((n/z-1)/4))+1))
    return float(max(0,min(1,1-s1+s2)))

def safe_gi(a,x):
    try: r=gammaincc(a,x); return float(max(0,min(1,r))) if not (np.isnan(r) or np.isinf(r)) else 0.0
    except: return 0.0

def nist_serial(b,m=SERIAL_M):
    n=len(b)
    if n<max(16,2*m+1): return (0.0,0.0)
    def psi(mm):
        c={}
        for i in range(n): k=tuple(b[(i+j)%n] for j in range(mm)); c[k]=c.get(k,0)+1
        return sum(v*v for v in c.values())*(2**mm)/n-n
    d1=psi(m)-psi(m+1); d2=psi(m)-(psi(m-1) if m>1 else 0)
    return safe_gi(2**(m-1)/2,d1/2), (safe_gi(2**(m-2)/2,d2/2) if m>1 else 0.0)

nist_f  = nist_freq(bits_corr)
nist_r  = nist_runs(bits_corr)
nist_c  = nist_cumsum(bits_corr)
nist_s  = nist_serial(bits_corr)

# ══════════════════════════════════════════════════════════════════
# 9. SHA-256 POST-PROCESSING
# ══════════════════════════════════════════════════════════════════
def sha256_process(bit_array):
    bit_str = "".join(map(str, bit_array))
    n       = len(bit_str)
    keys=[]; ents=[]
    if n == 0: return [], [], 0.0
    if n < 256:
        print(f"  Note: {n} bit < 256 → dihash sebagai 1 blok")
        blocks = [bit_str]
    else:
        blocks = [bit_str[i:i+256] for i in range(0,n-255,256)]
    for block in blocks:
        h     = hashlib.sha256(block.encode()).hexdigest()
        hb    = bin(int(h,16))[2:].zfill(256)
        keys.append(hb)
        p1=hb.count('1')/256; p0=1-p1; e=0.0
        if p1>0: e-=p1*math.log2(p1)
        if p0>0: e-=p0*math.log2(p0)
        ents.append(e)
    return keys, ents, max(ents) if ents else 0.0

final_keys, ent_list, max_ent = sha256_process(bits_corr)
t_pa          = perf_counter() - t_pa0
total_bits_pa = len(final_keys) * 256
T_pa          = T_recon + t_pa
kgr_pa        = total_bits_pa / T_pa if total_bits_pa > 0 else 0.0
entropy_pa    = compute_entropy(
    np.array([int(b) for b in ''.join(final_keys)],dtype=int)
) if final_keys else 0.0

print(f"\n--- HASIL SHA-256 ---")
print(f"Total blok   : {len(final_keys)}")
print(f"Total bit    : {total_bits_pa}")
print(f"KGR (PA)     : {kgr_pa:.4f} bps")
print(f"Entropy (PA) : {entropy_pa:.6f}")

# ══════════════════════════════════════════════════════════════════
# 10. RINGKASAN LENGKAP
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("    RINGKASAN HASIL — PROPOSED (Offset + SavGol)")
print("=" * 65)
print(f"\nFILE    : {FILE_PATH}")
print(f"Sampel  : {N}  |  Test: {len(A_test)} × {N_BITS} = {n_bits_total} bit")

print(f"\n--- TIMING ---")
print(f"T_akuisisi    : {T_AKUISISI:.0f} s    [{N}-1 × 1.0s]")
print(f"T_offset      : (bagian dari preprocessing)")
print(f"T_SavGol      : (bagian dari preprocessing)")
print(f"T_preprocessing: {t_pre*1000:.4f} ms  [offset + SavGol]")
print(f"T_quantisation : {t_q*1000:.4f} ms  [Gray code encoding]")
print(f"T_rekonsiliasi : {t_recon*1000:.4f} ms  [Cascade]")
print(f"T_SHA-256      : {t_pa*1000:.4f} ms")
print(f"T_total        : {T_pa:.4f} s")
print(f"  → Training  : 0 ms  (tidak ada model ML)")
print(f"  → Inference : 0 ms  (tidak ada model ML)")

print(f"\n--- KEY AGREEMENT METRICS ---")
hdr=(f"{'Stage':<25} {'KDR':>8} {'BAR':>8} {'KGR':>8} "
     f"{'Agreed KGR':>12} {'Entropy':>9} {'Bits':>6}")
print(hdr); print("-"*len(hdr))
print(f"{'Quantisation':<25} {kdr_quant:>8.5f} {1-kdr_quant:>8.5f} "
      f"{kgr_quant:>8.4f} {agreed_q:>12.4f} {entropy_q:>9.6f} {n_bits_total:>6}")
print(f"{'Reconciliation':<25} {kdr_recon:>8.5f} {1-kdr_recon:>8.5f} "
      f"{kgr_recon:>8.4f} {agreed_r:>12.4f} {entropy_recon:>9.6f} {n_bits_total:>6}")
print(f"{'SHA-256 output':<25} {'N/A':>8} {'N/A':>8} "
      f"{kgr_pa:>8.4f} {'N/A':>12} {entropy_pa:>9.6f} {total_bits_pa:>6}")

print(f"\n--- NIST RANDOMNESS (SUPPLEMENTARY) ---")
print(f"  Note: {n_bits_total} bit << 10^6 minimum NIST → exploratory only")
print(f"  Frequency  : {nist_f:.6f}  {'PASS' if nist_f>0.01 else 'FAIL'}")
print(f"  Runs       : {nist_r:.6f}  {'PASS' if nist_r>0.01 else 'FAIL'}")
print(f"  Cumul. sum : {nist_c:.6f}  {'PASS' if nist_c>0.01 else 'FAIL'}")
print(f"  Serial     : {nist_s}")

print(f"\n--- CORR & GAP ---")
print(f"Corr (raw)        : {corr_raw:.4f}")
print(f"Corr (after offset): {corr_offset:.4f}")
print(f"Corr (after SavGol): {corr_sg:.4f}")
print(f"Offset correction  : {median_offset:.3f} dB")
print(f"  (RFM95 vs SX1262 chip-level RSSI formula difference)")
print("=" * 65)
