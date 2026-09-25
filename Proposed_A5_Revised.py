"""
PROPOSED METHOD — A5 Configuration
Offset Normalisation + Wavelet Denoising + Gray Code 5-bit
+ Cascade Reconciliation + SHA-256

Ini adalah konfigurasi TERBAIK dari hasil ablation study.
Tidak menggunakan BiLSTM karena overfitting pada data kecil (< 50 samples).

Keunggulan vs baseline:
  - KDR terendah di antara semua pipeline multibit
  - Semua test samples dipakai (tidak ada window loss)
  - Komputasi sangat ringan (< 1 ms preprocessing)
  - Konsisten lintas 8 skenario mobilitas
"""

import pandas as pd
import numpy as np
import pywt
import hashlib
import math
import warnings
from time import perf_counter
from scipy.special import erfc, gammaincc
from scipy.fft import fft

# Suppress known warnings
warnings.filterwarnings('ignore', category=UserWarning, module='pywt')

# ─── KONFIGURASI ──────────────────────────────────────────────────────
FILE_PATH = "combined_v2i lurus 20.csv"   # ← ganti sesuai scenario
N_LEVELS  = 32     # jumlah level quantisasi
N_BITS    = 5      # log2(32) = 5 bit per sampel (Gray code)
SERIAL_M  = 2

# ══════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════
df         = pd.read_csv(FILE_PATH)
rssi_alice = df['rssi_alice'].values.astype(float)
rssi_bob   = df['rssi_bob'].values.astype(float)
N          = len(rssi_alice)
T_AKUISISI = (N - 1) * 1.0   # (N-1) x 1 detik/sampel

print("=" * 60)
print(f"FILE          : {FILE_PATH}")
print(f"Total sampel  : {N}")
print(f"T_akuisisi    : {T_AKUISISI:.1f} detik ({N}-1 x 1.0s)")
print(f"Corr (raw)    : {np.corrcoef(rssi_alice, rssi_bob)[0,1]:.4f}")
print(f"Gap median    : {np.median(rssi_alice - rssi_bob):.2f} dB")
print("=" * 60)

# ══════════════════════════════════════════════════════════════════════
# 2. CHRONOLOGICAL SPLIT 60 / 20 / 20
# ══════════════════════════════════════════════════════════════════════
n_train = int(N * 0.60)
n_val   = int(N * 0.20)

A_train = rssi_alice[:n_train]
B_train = rssi_bob[:n_train]
A_val   = rssi_alice[n_train : n_train + n_val]
B_val   = rssi_bob[n_train : n_train + n_val]
A_test  = rssi_alice[n_train + n_val:]
B_test  = rssi_bob[n_train + n_val:]

print(f"\nSplit -> Train:{n_train} | Val:{n_val} | Test:{len(A_test)}")

# ══════════════════════════════════════════════════════════════════════
# 3. PRE-PROCESSING: OFFSET NORMALISATION + WAVELET DENOISING
# ══════════════════════════════════════════════════════════════════════
t_pre0 = perf_counter()

# ── 3a. Offset normalisation (dihitung HANYA dari training partition) ─
median_offset = np.median(A_train - B_train)
mean_offset   = np.mean(A_train - B_train)
print(f"\nOffset (median train) : {median_offset:.3f} dB")
print(f"Offset (mean train)   : {mean_offset:.3f} dB")

# Koreksi offset pada Alice (median lebih robust dari mean)
A_train_corr = A_train - median_offset
A_val_corr   = A_val   - median_offset
A_test_corr  = A_test  - median_offset
# Bob tidak diubah — offset dikoreksi dari sisi Alice

# ── 3b. Wavelet denoising (level=1, aman untuk semua ukuran data) ─────
def wavelet_denoising(signal, wavelet='db4'):
    """
    DWT soft thresholding dengan level=1.
    Level 1 dipilih agar tidak ada boundary effects
    pada data kecil (< 100 sampel).
    Formula threshold: T = (median|d| / 0.6745) x sqrt(2 log N)
    """
    signal = np.array(signal, dtype=float)
    n      = len(signal)
    if n < 4:
        return signal
    coeffs = pywt.wavedec(signal, wavelet, level=1)
    sigma  = np.median(np.abs(coeffs[-1])) / 0.6745
    thr    = sigma * np.sqrt(2 * np.log(max(n, 2)))
    new_c  = [coeffs[0]] + [pywt.threshold(c, thr, 'soft')
                             for c in coeffs[1:]]
    return pywt.waverec(new_c, wavelet)[:n]

# Denoising pada masing-masing partisi secara terpisah
A_tr_den = wavelet_denoising(A_train_corr)
B_tr_den = wavelet_denoising(B_train)
A_ts_den = wavelet_denoising(A_test_corr)
B_ts_den = wavelet_denoising(B_test)

t_pre = perf_counter() - t_pre0

# Korelasi setelah preprocessing
corr_raw = np.corrcoef(rssi_alice, rssi_bob)[0, 1]
corr_den = np.corrcoef(A_ts_den, B_ts_den)[0, 1]
print(f"Corr (raw)            : {corr_raw:.4f}")
print(f"Corr (denoised test)  : {corr_den:.4f}")

# ══════════════════════════════════════════════════════════════════════
# 4. GRAY CODE 5-BIT QUANTISATION
# ══════════════════════════════════════════════════════════════════════
# Threshold HANYA dari training partition (tidak bocor ke test)
THR_MIN = A_tr_den.min()
THR_MAX = A_tr_den.max()

print(f"\nThreshold (train)     : [{THR_MIN:.2f}, {THR_MAX:.2f}] dBm")

def int_to_gray(n, bits=N_BITS):
    """Konversi integer ke Gray code bit array."""
    gray = n ^ (n >> 1)
    return [(gray >> (bits - 1 - i)) & 1 for i in range(bits)]

def gray_quantise(signal, lo=THR_MIN, hi=THR_MAX,
                  n_levels=N_LEVELS, n_bits=N_BITS):
    """
    Quantisasi RSSI ke Gray code 5-bit.
    32 level -> 5 bit per sampel.
    Adjacent levels berbeda hanya 1 bit -> KDR minimal di boundary.
    Menggunakan SEMUA sampel (tidak ada guard band/window loss).
    """
    step     = (hi - lo) / n_levels
    bits_out = []
    for v in signal:
        lv = int((v - lo) / step)
        lv = max(0, min(n_levels - 1, lv))
        bits_out.extend(int_to_gray(lv, n_bits))
    return np.array(bits_out, dtype=int)

# Quantisasi test partition
t_q0 = perf_counter()
bits_alice = gray_quantise(A_ts_den)
bits_bob   = gray_quantise(B_ts_den)
t_q        = perf_counter() - t_q0

# Sinkronisasi panjang
n_bits_total = min(len(bits_alice), len(bits_bob))
bits_alice   = bits_alice[:n_bits_total]
bits_bob     = bits_bob[:n_bits_total]

print(f"\nTest samples used     : {len(A_test)} (semua, tanpa window loss)")
print(f"Total bits generated  : {n_bits_total} bit ({len(A_test)} x {N_BITS})")

# ══════════════════════════════════════════════════════════════════════
# 5. HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════
def compute_kdr(a, b):
    n = min(len(a), len(b))
    return float(np.sum(a[:n] != b[:n]) / n) if n > 0 else 1.0

def compute_entropy(bits):
    """Shannon entropy (0.0 = semua sama, 1.0 = perfectly balanced)."""
    bits = np.array(bits, dtype=int)
    p1   = np.mean(bits)
    p0   = 1.0 - p1
    h    = 0.0
    if p1 > 0: h -= p1 * np.log2(p1)
    if p0 > 0: h -= p0 * np.log2(p0)
    return h

# ══════════════════════════════════════════════════════════════════════
# 6. METRIK QUANTISASI
# ══════════════════════════════════════════════════════════════════════
T_quant      = T_AKUISISI + t_pre + t_q
kdr_quant    = compute_kdr(bits_alice, bits_bob)
kgr_quant    = n_bits_total / T_quant
agreed_q     = kgr_quant * (1 - kdr_quant)
entropy_q    = compute_entropy(bits_alice)

print(f"\n--- HASIL QUANTISASI ---")
print(f"KDR          : {kdr_quant:.5f}")
print(f"BAR          : {1 - kdr_quant:.5f}")
print(f"KGR (raw)    : {kgr_quant:.4f} bps")
print(f"Agreed KGR   : {agreed_q:.4f} bps")
print(f"Entropy      : {entropy_q:.6f}")
print(f"Bits         : {n_bits_total}")

# ══════════════════════════════════════════════════════════════════════
# 7. REKONSILIASI — Cascade
# ══════════════════════════════════════════════════════════════════════
t_recon0 = perf_counter()

def dynamic_block_size(n):
    if n <= 256:  return 16
    elif n <= 512: return 32
    elif n <= 1024: return 64
    else:          return 128

def parity(block):
    return int(np.sum(block) % 2)

def binary_search_correct(a_block, b_block, start_idx, corrected):
    left, right = 0, len(a_block) - 1
    while left <= right:
        if left == right:
            if a_block[left] != b_block[left]:
                corrected[start_idx + left] = a_block[left]
            break
        mid = (left + right) // 2
        if parity(a_block[left:mid+1]) != parity(b_block[left:mid+1]):
            right = mid
        else:
            left = mid + 1

def cascade_reconciliation(alice, bob):
    """
    Cascade protocol: iterative block parity exchange.
    Lebih aman dari direct bit-copy karena tidak mengekspos
    seluruh bit string ke public channel.
    """
    corrected  = bob.copy()
    block_size = dynamic_block_size(len(alice))

    while True:
        blocks_a = [alice[i:i+block_size]
                    for i in range(0, len(alice), block_size)]
        blocks_b = [corrected[i:i+block_size]
                    for i in range(0, len(corrected), block_size)]

        for i, (a_b, b_b) in enumerate(zip(blocks_a, blocks_b)):
            if parity(a_b) != parity(b_b):
                binary_search_correct(a_b, b_b,
                                      i * block_size, corrected)

        mismatch = np.where(alice != corrected)[0]
        if len(mismatch) == 0 or block_size == 1:
            break
        block_size = max(1, block_size // 2)

    return corrected

bits_alice_corr = cascade_reconciliation(bits_alice, bits_bob)

t_recon      = perf_counter() - t_recon0
kdr_recon    = compute_kdr(bits_alice_corr, bits_bob)
T_recon      = T_quant + t_recon
kgr_recon    = n_bits_total / T_recon
agreed_r     = kgr_recon * (1 - kdr_recon)
entropy_recon = compute_entropy(bits_alice_corr)

print(f"\n--- HASIL REKONSILIASI (CASCADE) ---")
print(f"KDR          : {kdr_recon:.5f}")
print(f"BAR          : {1 - kdr_recon:.5f}")
print(f"KGR (raw)    : {kgr_recon:.4f} bps")
print(f"Agreed KGR   : {agreed_r:.4f} bps")
print(f"Entropy      : {entropy_recon:.6f}")
print(f"Recon time   : {t_recon * 1000:.4f} ms")

# ══════════════════════════════════════════════════════════════════════
# 8. NIST RANDOMNESS TESTS (supplementary — exploratory)
# ══════════════════════════════════════════════════════════════════════
t_pa0 = perf_counter()

def nist_frequency(bits):
    n = len(bits)
    if n < 16: return 0.0
    s = sum(1 if b == 1 else -1 for b in bits)
    return float(erfc(abs(s) / math.sqrt(2.0 * n)))

def nist_runs(bits):
    n = len(bits)
    if n < 100: return 0.0
    pi  = sum(int(b) for b in bits) / n
    tau = 2.0 / math.sqrt(n)
    if abs(pi - 0.5) >= tau: return 0.0
    v   = 1 + sum(1 for i in range(1, n) if bits[i] != bits[i-1])
    num = abs(v - 2.0 * n * pi * (1 - pi))
    den = 2.0 * math.sqrt(2.0 * n) * pi * (1 - pi)
    return 0.0 if den == 0 else float(erfc(num / den))

def nist_cumsum(bits):
    n = len(bits)
    if n < 100: return 0.0
    x  = np.array([1 if b == 1 else -1 for b in bits])
    s  = np.cumsum(x)
    z  = int(np.max(np.abs(s)))
    if z == 0: return 1.0
    s1 = sum(math.exp(-((4*k+1)**2 * z**2) / (2.0*n))
             for k in range(int(np.floor((-n/z+1)/4)),
                            int(np.floor((n/z-1)/4)) + 1))
    s2 = sum(math.exp(-((4*k+3)**2 * z**2) / (2.0*n))
             for k in range(int(np.floor((-n/z-3)/4)),
                            int(np.floor((n/z-1)/4)) + 1))
    return float(max(0.0, min(1.0, 1.0 - s1 + s2)))

def nist_fft(bits):
    n = len(bits)
    if n < 1000: return 0.0
    x  = np.array([2*b - 1 for b in bits], dtype=float)
    s  = np.abs(fft(x))[:n//2]
    T  = math.sqrt(math.log(1.0/0.05) * n)
    N0 = 0.95 * n / 2.0
    N1 = np.sum(s < T)
    d  = (N1 - N0) / math.sqrt(n * 0.95 * 0.05 / 4.0)
    return float(erfc(abs(d) / math.sqrt(2.0)))

def safe_gi(a, x):
    try:
        r = gammaincc(a, x)
        return float(max(0.0, min(1.0, r))) \
               if not (np.isnan(r) or np.isinf(r)) else 0.0
    except:
        return 0.0

def nist_serial(bits, m=SERIAL_M):
    n = len(bits)
    if n < max(16, 2*m+1): return (0.0, 0.0)
    def psi(mm):
        counts = {}
        for i in range(n):
            patt = tuple(bits[(i+j) % n] for j in range(mm))
            counts[patt] = counts.get(patt, 0) + 1
        return (sum(v*v for v in counts.values()) * (2.0**mm) / n) - n
    d1 = psi(m) - psi(m+1)
    d2 = psi(m) - (psi(m-1) if m > 1 else 0.0)
    return (safe_gi(2**(m-1)/2.0, d1/2.0),
            safe_gi(2**(m-2)/2.0, d2/2.0) if m > 1 else 0.0)

nist_freq = nist_frequency(bits_alice_corr)
nist_run  = nist_runs(bits_alice_corr)
nist_cum  = nist_cumsum(bits_alice_corr)
nist_f    = nist_fft(bits_alice_corr)
nist_ser  = nist_serial(bits_alice_corr)

# ══════════════════════════════════════════════════════════════════════
# 9. SHA-256 POST-PROCESSING (block adaptif)
# ══════════════════════════════════════════════════════════════════════
def sha256_process(bit_array):
    """
    SHA-256 block adaptif.
    < 256 bit  : hash semua -> 1 blok 256-bit output.
    >= 256 bit : proses per blok 256 bit.
    """
    bit_str  = "".join(map(str, bit_array))
    n_bits   = len(bit_str)
    out_bits = []
    ent_list = []

    if n_bits == 0:
        return [], [], 0.0

    if n_bits < 256:
        print(f"  Note: {n_bits} bit < 256 -> dihash sebagai 1 blok")
        blocks_to_process = [bit_str]
    else:
        blocks_to_process = [bit_str[i:i+256]
                             for i in range(0, n_bits - 255, 256)]

    for block in blocks_to_process:
        h      = hashlib.sha256(block.encode()).hexdigest()
        h_bits = bin(int(h, 16))[2:].zfill(256)
        out_bits.append(h_bits)
        p1 = h_bits.count('1') / 256
        p0 = 1 - p1
        e  = 0.0
        if p1 > 0: e -= p1 * math.log2(p1)
        if p0 > 0: e -= p0 * math.log2(p0)
        ent_list.append(e)

    return out_bits, ent_list, max(ent_list) if ent_list else 0.0

final_keys, ent_list, max_entropy = sha256_process(bits_alice_corr)

t_pa          = perf_counter() - t_pa0
total_bits_pa = len(final_keys) * 256
T_pa          = T_recon + t_pa
kgr_pa        = total_bits_pa / T_pa if total_bits_pa > 0 else 0.0

if final_keys:
    pa_arr     = np.array([int(b) for b in ''.join(final_keys)], dtype=int)
    entropy_pa = compute_entropy(pa_arr)
else:
    entropy_pa = 0.0

print(f"\n--- HASIL SHA-256 POST-PROCESSING ---")
print(f"Total blok       : {len(final_keys)}")
print(f"Total bit output : {total_bits_pa}")
print(f"KGR (PA)         : {kgr_pa:.4f} bps")
print(f"Entropy (PA out) : {entropy_pa:.6f}")
if ent_list:
    print(f"Entropy per blok : {[f'{e:.6f}' for e in ent_list]}")

# ══════════════════════════════════════════════════════════════════════
# 10. RINGKASAN LENGKAP
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("         RINGKASAN HASIL — PROPOSED A5")
print("         (Offset Normalisation + Wavelet + Gray 5-bit)")
print("=" * 65)

print(f"\nFILE        : {FILE_PATH}")
print(f"N sampel    : {N}  |  Test: {len(A_test)} sampel x {N_BITS} = {n_bits_total} bit")

print(f"\n--- TIMING ---")
print(f"T_akuisisi     : {T_AKUISISI:.1f} s     [{N}-1 x 1.0s, LoRa setting]")
print(f"T_preprocessing: {t_pre*1000:.3f} ms   [Offset norm + Wavelet]")
print(f"T_quantisasi   : {t_q*1000:.4f} ms   [Gray code encoding]")
print(f"T_rekonsiliasi : {t_recon*1000:.4f} ms   [Cascade protocol]")
print(f"T_SHA-256      : {t_pa*1000:.4f} ms   [Privacy amplification]")
print(f"T_total        : {T_pa:.4f} s")
print(f"\n  Note: Tidak ada training time (tidak ada model ML)")
print(f"        Tidak ada inference time (tidak ada model ML)")

print(f"\n--- KEY AGREEMENT METRICS ---")
hdr = (f"{'Stage':<25} {'KDR':>8} {'BAR':>8} "
       f"{'KGR':>8} {'Agreed KGR':>12} {'Entropy':>9} {'Bits':>6}")
print(hdr)
print("-" * len(hdr))
print(f"{'Quantisation':<25} {kdr_quant:>8.5f} {1-kdr_quant:>8.5f} "
      f"{kgr_quant:>8.4f} {agreed_q:>12.4f} "
      f"{entropy_q:>9.6f} {n_bits_total:>6}")
print(f"{'Reconciliation':<25} {kdr_recon:>8.5f} {1-kdr_recon:>8.5f} "
      f"{kgr_recon:>8.4f} {agreed_r:>12.4f} "
      f"{entropy_recon:>9.6f} {n_bits_total:>6}")
print(f"{'SHA-256 output':<25} {'N/A':>8} {'N/A':>8} "
      f"{kgr_pa:>8.4f} {'N/A':>12} "
      f"{entropy_pa:>9.6f} {total_bits_pa:>6}")

print(f"\n--- NIST RANDOMNESS (SUPPLEMENTARY) ---")
print(f"  Note: {n_bits_total} bit << 10^6 minimum NIST -> exploratory only")
print(f"  Frequency  : {nist_freq:.6f}  {'PASS' if nist_freq > 0.01 else 'FAIL'}")
print(f"  Runs       : {nist_run:.6f}  {'PASS' if nist_run  > 0.01 else 'FAIL'}")
print(f"  Cumul. sum : {nist_cum:.6f}  {'PASS' if nist_cum  > 0.01 else 'FAIL'}")
print(f"  FFT        : {nist_f:.6f}  {'PASS' if nist_f    > 0.01 else 'FAIL'}")
print(f"  Serial     : {nist_ser}")

print(f"\n--- CORR & GAP ---")
print(f"Pearson corr (raw)      : {corr_raw:.4f}")
print(f"Pearson corr (denoised) : {corr_den:.4f}")
print(f"Median gap Alice-Bob    : {np.median(rssi_alice - rssi_bob):.2f} dB")
print(f"Offset correction       : {median_offset:.3f} dB")

print(f"\n--- KEUNGGULAN vs BASELINE (prediksi dari ablation) ---")
baselines = {
    'Toward [16]':     0.183,
    'Vehicle-Key [14]':0.200,
    'PLKG [13]':       0.200,
    'PLS-LoRa [8]':    0.188,
}
print(f"  KDR Proposed A5 : {kdr_quant:.5f}")
for name, bl_kdr in baselines.items():
    if kdr_quant < bl_kdr:
        red = (bl_kdr - kdr_quant) / bl_kdr * 100
        print(f"  vs {name:<20}: {bl_kdr:.3f} -> "
              f"reduction {red:.1f}% LEBIH BAIK")
    else:
        inc = (kdr_quant - bl_kdr) / bl_kdr * 100
        print(f"  vs {name:<20}: {bl_kdr:.3f} -> "
              f"{inc:.1f}% lebih tinggi")
print("=" * 65)
