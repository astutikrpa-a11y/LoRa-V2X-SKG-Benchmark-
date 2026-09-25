"""
PROPOSED PIPELINE — Revised v3
Wavelet + Offset Norm + BiLSTM + Gray Code 5-bit
Changelog v3:
  - warnings.filterwarnings suppress pywt dan torch warnings
  - wavelet level = 1 (aman untuk semua ukuran data)
  - scheduler.step(loss.item()) fix tensor scalar warning
  - warm-up run sebelum inference timing
  - entropy dilaporkan per stage
  - SHA-256 block adaptif (handle < 256 bit)
"""

import pandas as pd
import numpy as np
import pywt
import torch
import torch.nn as nn
import hashlib, math, warnings
from time import perf_counter
from collections import Counter
from scipy.special import erfc, gammaincc
from scipy.fft import fft

# Suppress warnings yang sudah diketahui penyebabnya
warnings.filterwarnings('ignore', category=UserWarning, module='pywt')
warnings.filterwarnings('ignore', category=UserWarning, module='torch')

# ─── KONFIGURASI ──────────────────────────────────────────────────────
FILE_PATH  = "combined_v2i lurus 20.csv"   # ← ganti sesuai scenario
N_LEVELS   = 32      # jumlah level quantisasi
N_BITS     = 5       # log2(32) = 5 bit per sampel Gray code
SEQ_LEN    = 5       # window BiLSTM
HIDDEN     = 16      # hidden units per direction
EPOCHS     = 150     # max epoch training
THETA      = 0.9     # bobot MSE dalam combined loss
SEED       = 42
SERIAL_M   = 2

torch.manual_seed(SEED)
np.random.seed(SEED)

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
# 3. PRE-PROCESSING
# ══════════════════════════════════════════════════════════════════════
t_pre0 = perf_counter()

# ── 3a. Offset normalisation dari training partition ──────────────────
median_offset = np.median(A_train - B_train)
print(f"Offset normalisation (median train): {median_offset:.3f} dB")

A_train_n = A_train - median_offset
A_val_n   = A_val   - median_offset
A_test_n  = A_test  - median_offset

# ── 3b. Wavelet denoising — level=1 aman untuk semua ukuran data ─────
def wavelet_denoising(signal, wavelet='db4'):
    """
    DWT soft thresholding.
    Level = 1 agar tidak ada boundary effects pada data kecil.
    """
    signal = np.array(signal, dtype=float)
    n      = len(signal)
    if n < 4:
        return signal   # terlalu pendek, skip denoising
    # Level 1 selalu aman untuk semua ukuran
    level   = 1
    coeffs  = pywt.wavedec(signal, wavelet, level=level)
    sigma   = np.median(np.abs(coeffs[-1])) / 0.6745
    thr     = sigma * np.sqrt(2 * np.log(max(n, 2)))
    new_c   = [coeffs[0]] + [pywt.threshold(c, thr, 'soft')
                              for c in coeffs[1:]]
    return pywt.waverec(new_c, wavelet)[:n]

A_tr_den = wavelet_denoising(A_train_n)
B_tr_den = wavelet_denoising(B_train)
A_ts_den = wavelet_denoising(A_test_n)
B_ts_den = wavelet_denoising(B_test)

corr_den = np.corrcoef(A_ts_den, B_ts_den)[0, 1]
print(f"Corr (denoised test): {corr_den:.4f}")

# ── 3c. Normalisasi RSSI untuk BiLSTM ────────────────────────────────
# Statistik dari training partition saja (tidak bocor ke test)
A_mean = A_tr_den.mean()
A_std  = A_tr_den.std() + 1e-9
B_mean = B_tr_den.mean()
B_std  = B_tr_den.std() + 1e-9

A_tr_norm = (A_tr_den - A_mean) / A_std
B_tr_norm = (B_tr_den - B_mean) / B_std
A_ts_norm = (A_ts_den - A_mean) / A_std   # pakai statistik TRAIN
B_ts_norm = (B_ts_den - B_mean) / B_std

t_pre = perf_counter() - t_pre0

# Threshold quantisasi dari training partition (skala asli dBm)
THR_MIN = A_tr_den.min()
THR_MAX = A_tr_den.max()

# ══════════════════════════════════════════════════════════════════════
# 4. GRAY CODE QUANTISATION
# ══════════════════════════════════════════════════════════════════════
def int_to_gray(n, bits=N_BITS):
    """Konversi integer ke Gray code bit array."""
    gray = n ^ (n >> 1)
    return [(gray >> (bits - 1 - i)) & 1 for i in range(bits)]

def gray_quantise(signal_dBm, lo=THR_MIN, hi=THR_MAX,
                  n_levels=N_LEVELS, n_bits=N_BITS):
    """
    Quantisasi RSSI (skala asli dBm) ke Gray code 5-bit.
    Threshold dihitung dari training partition.
    """
    step     = (hi - lo) / n_levels
    bits_out = []
    for v in signal_dBm:
        lv = int((v - lo) / step)
        lv = max(0, min(n_levels - 1, lv))
        bits_out.extend(int_to_gray(lv, n_bits))
    return np.array(bits_out, dtype=int)

# Target Gray bits Bob (skala asli dBm) untuk supervised training
bob_gray_tr = np.array([
    int_to_gray(
        max(0, min(N_LEVELS - 1,
            int((v - THR_MIN) / (THR_MAX - THR_MIN + 1e-9) * N_LEVELS))),
        N_BITS)
    for v in B_tr_den[SEQ_LEN:]])

# ══════════════════════════════════════════════════════════════════════
# 5. BUAT WINDOWS (hanya dalam partisi masing-masing)
# ══════════════════════════════════════════════════════════════════════
def create_windows(x, y, seq_len):
    """Sliding windows kronologis — tidak cross-partition."""
    X, Y = [], []
    for i in range(len(x) - seq_len):
        X.append(x[i : i + seq_len])
        Y.append(y[i : i + seq_len])
    return np.array(X), np.array(Y)

X_tr, Y_tr = create_windows(A_tr_norm, B_tr_norm, SEQ_LEN)
n_test_windows = max(0, len(A_ts_norm) - SEQ_LEN)
print(f"\nTrain windows: {len(X_tr)}  |  Test windows: {n_test_windows}")

if len(X_tr) < 2:
    print("ERROR: Training data terlalu sedikit. Gunakan file lebih besar.")
    exit()

# ══════════════════════════════════════════════════════════════════════
# 6. BiLSTM MODEL
# ══════════════════════════════════════════════════════════════════════
class BiLSTM_KeyAgreement(nn.Module):
    def __init__(self, hidden=HIDDEN, n_bits=N_BITS):
        super().__init__()
        self.lstm     = nn.LSTM(1, hidden, batch_first=True,
                                bidirectional=True)
        self.dropout  = nn.Dropout(0.1)
        self.fc_pred  = nn.Linear(hidden * 2, 1)
        self.fc_gray  = nn.Linear(hidden * 2, n_bits)
        self.sigmoid  = nn.Sigmoid()

    def forward(self, x):
        out, _    = self.lstm(x)
        out       = self.dropout(out)
        pred_rssi = self.fc_pred(out)
        bits_gray = self.sigmoid(self.fc_gray(out[:, -1, :]))
        return pred_rssi, bits_gray

model    = BiLSTM_KeyAgreement()
n_params = sum(p.numel() for p in model.parameters())
print(f"BiLSTM parameters: {n_params:,}")

# ══════════════════════════════════════════════════════════════════════
# 7. TRAINING (offline — diukur terpisah)
# ══════════════════════════════════════════════════════════════════════
Xtr = torch.tensor(X_tr, dtype=torch.float32).unsqueeze(-1)
Ytr = torch.tensor(Y_tr, dtype=torch.float32).unsqueeze(-1)
Btr = torch.tensor(bob_gray_tr[:len(X_tr)], dtype=torch.float32)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001,
                              weight_decay=1e-4)
mse_fn    = nn.MSELoss()
bce_fn    = nn.BCELoss()

# ReduceLROnPlateau tanpa verbose (kompatibel semua versi PyTorch)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=10, factor=0.5)

best_loss  = float('inf')
patience   = 20
no_improve = 0

t_train_start = perf_counter()

for epoch in range(EPOCHS):
    model.train()
    pred_rssi, pred_bits = model(Xtr)
    loss_mse = mse_fn(pred_rssi, Ytr)
    loss_bce = bce_fn(pred_bits, Btr)
    loss     = THETA * loss_mse + (1 - THETA) * loss_bce

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # .item() konversi tensor -> float Python (fix scheduler warning)
    scheduler.step(loss.item())

    if loss.item() < best_loss - 1e-5:
        best_loss  = loss.item()
        no_improve = 0
    else:
        no_improve += 1

    if no_improve >= patience:
        print(f"  Early stopping pada epoch {epoch}")
        break

    if epoch % 30 == 0:
        print(f"  Epoch {epoch:3d}  Loss: {loss.item():.5f}"
              f"  (MSE:{loss_mse.item():.4f}"
              f"  BCE:{loss_bce.item():.4f})")

t_train = perf_counter() - t_train_start
print(f"\nTraining time (offline) : {t_train:.4f} s  ({t_train*1000:.1f} ms)")

# ══════════════════════════════════════════════════════════════════════
# 8. INFERENCE pada TEST partition (warm-up sebelum timing)
# ══════════════════════════════════════════════════════════════════════
X_ts, Y_ts = create_windows(A_ts_norm, B_ts_norm, SEQ_LEN)

if len(X_ts) == 0:
    print(f"\nWARNING: test partition terlalu kecil untuk membuat window!")
    print(f"  Test size={len(A_ts_norm)}, SEQ_LEN={SEQ_LEN}")
    print(f"  Kurangi SEQ_LEN atau gunakan file dengan lebih banyak sampel.")
    exit()

Xts = torch.tensor(X_ts, dtype=torch.float32).unsqueeze(-1)

# Warm-up run — inisialisasi kernel PyTorch sebelum pengukuran
model.eval()
with torch.no_grad():
    _ = model(Xts[:1])   # 1 forward pass untuk warm-up

# Timing inference yang sebenarnya
t_infer_start = perf_counter()
with torch.no_grad():
    _, pred_gray = model(Xts)
t_infer = perf_counter() - t_infer_start

# Alice: bits dari BiLSTM (prediksi)
bits_alice = (pred_gray.numpy() > 0.5).astype(int)

# Bob: Gray quantisasi langsung dari RSSI asli (skala dBm)
bob_test_vals = B_ts_den[SEQ_LEN : SEQ_LEN + len(X_ts)]
bits_bob_arr  = np.array([
    int_to_gray(
        max(0, min(N_LEVELS - 1,
            int((v - THR_MIN) / (THR_MAX - THR_MIN + 1e-9) * N_LEVELS))),
        N_BITS)
    for v in bob_test_vals])

# Flatten dan sinkronisasi panjang
bits_alice_flat = bits_alice.flatten()
bits_bob_flat   = bits_bob_arr.flatten()
n_bits_total    = min(len(bits_alice_flat), len(bits_bob_flat))
bits_alice_flat = bits_alice_flat[:n_bits_total]
bits_bob_flat   = bits_bob_flat[:n_bits_total]

print(f"\nInference latency    : {t_infer * 1000:.3f} ms per block")
print(f"Test windows         : {len(X_ts)}")
print(f"Total bits generated : {n_bits_total} bit ({len(X_ts)} x {N_BITS})")

# ══════════════════════════════════════════════════════════════════════
# 9. HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════
def compute_kdr(a, b):
    n = min(len(a), len(b))
    return float(np.sum(a[:n] != b[:n]) / n) if n > 0 else 1.0

def compute_entropy(bits):
    """Shannon entropy dari bit array (0 sampai 1.0)."""
    bits = np.array(bits, dtype=int)
    p1   = np.mean(bits)
    p0   = 1.0 - p1
    h    = 0.0
    if p1 > 0: h -= p1 * np.log2(p1)
    if p0 > 0: h -= p0 * np.log2(p0)
    return h

# ══════════════════════════════════════════════════════════════════════
# 10. METRIK QUANTISASI
# ══════════════════════════════════════════════════════════════════════
kdr_quant     = compute_kdr(bits_alice_flat, bits_bob_flat)
T_quant       = T_AKUISISI + t_pre + t_train + t_infer
kgr_quant     = n_bits_total / T_quant
agreed_kgr_q  = kgr_quant * (1 - kdr_quant)
entropy_quant = compute_entropy(bits_alice_flat)

print(f"\n--- HASIL QUANTISASI ---")
print(f"KDR          : {kdr_quant:.5f}")
print(f"BAR          : {1 - kdr_quant:.5f}")
print(f"KGR (raw)    : {kgr_quant:.4f} bps")
print(f"Agreed KGR   : {agreed_kgr_q:.4f} bps")
print(f"Entropy      : {entropy_quant:.6f}")
print(f"Bits         : {n_bits_total}")

# ══════════════════════════════════════════════════════════════════════
# 11. REKONSILIASI — direct bit-correction (oracle-assisted)
# ══════════════════════════════════════════════════════════════════════
t_recon_start = perf_counter()

bits_alice_corr = bits_alice_flat.copy()
mismatch_idx    = np.where(bits_alice_flat != bits_bob_flat)[0]
bits_alice_corr[mismatch_idx] = bits_bob_flat[mismatch_idx]

t_recon       = perf_counter() - t_recon_start
kdr_recon     = compute_kdr(bits_alice_corr, bits_bob_flat)
T_recon       = T_quant + t_recon
kgr_recon     = n_bits_total / T_recon
agreed_kgr_r  = kgr_recon * (1 - kdr_recon)
entropy_recon = compute_entropy(bits_alice_corr)

print(f"\n--- HASIL REKONSILIASI ---")
print(f"KDR          : {kdr_recon:.5f}")
print(f"BAR          : {1 - kdr_recon:.5f}")
print(f"KGR (raw)    : {kgr_recon:.4f} bps")
print(f"Agreed KGR   : {agreed_kgr_r:.4f} bps")
print(f"Entropy      : {entropy_recon:.6f}")
print(f"Recon time   : {t_recon * 1000:.4f} ms")

# ══════════════════════════════════════════════════════════════════════
# 12. NIST RANDOMNESS TESTS (supplementary — exploratory)
# ══════════════════════════════════════════════════════════════════════
t_pa_start = perf_counter()

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
# 13. SHA-256 POST-PROCESSING (block adaptif)
# ══════════════════════════════════════════════════════════════════════
def sha256_process(bit_array):
    """
    SHA-256 dengan block adaptif.
    < 256 bit : hash semua bit menjadi 1 blok 256-bit output.
    >= 256 bit: proses per blok 256 bit.
    """
    bit_str  = "".join(map(str, bit_array))
    n_bits   = len(bit_str)
    out_bits = []
    ent_list = []

    if n_bits == 0:
        return [], [], 0.0

    if n_bits < 256:
        print(f"  Note: {n_bits} bit < 256 -> seluruh bit dihash 1 blok")
        h      = hashlib.sha256(bit_str.encode()).hexdigest()
        h_bits = bin(int(h, 16))[2:].zfill(256)
        out_bits.append(h_bits)
        p1 = h_bits.count('1') / 256; p0 = 1 - p1
        e  = 0.0
        if p1 > 0: e -= p1 * math.log2(p1)
        if p0 > 0: e -= p0 * math.log2(p0)
        ent_list.append(e)
    else:
        for i in range(0, n_bits - 255, 256):
            block  = bit_str[i : i + 256]
            h      = hashlib.sha256(block.encode()).hexdigest()
            h_bits = bin(int(h, 16))[2:].zfill(256)
            out_bits.append(h_bits)
            p1 = h_bits.count('1') / 256; p0 = 1 - p1
            e  = 0.0
            if p1 > 0: e -= p1 * math.log2(p1)
            if p0 > 0: e -= p0 * math.log2(p0)
            ent_list.append(e)

    max_ent = max(ent_list) if ent_list else 0.0
    return out_bits, ent_list, max_ent

final_keys, ent_list, max_entropy = sha256_process(bits_alice_corr)

t_pa          = perf_counter() - t_pa_start
total_bits_pa = len(final_keys) * 256
T_pa          = T_recon + t_pa
kgr_pa        = total_bits_pa / T_pa if total_bits_pa > 0 else 0.0

if final_keys:
    pa_bits_arr = np.array([int(b) for b in ''.join(final_keys)], dtype=int)
    entropy_pa  = compute_entropy(pa_bits_arr)
else:
    entropy_pa = 0.0

print(f"\n--- HASIL SHA-256 POST-PROCESSING ---")
print(f"Total blok       : {len(final_keys)}")
print(f"Total bit output : {total_bits_pa}")
print(f"KGR (PA)         : {kgr_pa:.4f} bps")
print(f"Entropy (PA out) : {entropy_pa:.6f}")
if ent_list:
    print(f"Entropy per blok : {[f'{e:.4f}' for e in ent_list]}")

# ══════════════════════════════════════════════════════════════════════
# 14. RINGKASAN LENGKAP
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("              RINGKASAN HASIL")
print("=" * 65)
print(f"\nFILE     : {FILE_PATH}")
print(f"N sampel : {N}  |  SEQ_LEN:{SEQ_LEN}  |"
      f"  Test:{len(X_ts)} windows x {N_BITS} = {n_bits_total} bit")

print(f"\n--- TIMING ---")
print(f"T_akuisisi (LoRa)  : {T_AKUISISI:.1f} s  [{N}-1 x 1.0s]")
print(f"T_preprocessing    : {t_pre*1000:.3f} ms  [Wavelet + offset + norm]")
print(f"T_train (offline)  : {t_train*1000:.1f} ms  [BiLSTM 1 kali]")
print(f"T_infer per block  : {t_infer*1000:.3f} ms  [BiLSTM forward pass]")
print(f"T_rekonsiliasi     : {t_recon*1000:.4f} ms")
print(f"T_SHA-256          : {t_pa*1000:.4f} ms")
print(f"T_total pipeline   : {T_pa:.4f} s")

print(f"\n--- KEY AGREEMENT METRICS ---")
header = f"{'Stage':<25} {'KDR':>8} {'BAR':>8} {'KGR':>8} {'Agreed KGR':>12} {'Entropy':>9} {'Bits':>6}"
print(header)
print("-" * len(header))
print(f"{'Quantisation':<25} {kdr_quant:>8.5f} {1-kdr_quant:>8.5f}"
      f" {kgr_quant:>8.4f} {agreed_kgr_q:>12.4f}"
      f" {entropy_quant:>9.6f} {n_bits_total:>6}")
print(f"{'Reconciliation':<25} {kdr_recon:>8.5f} {1-kdr_recon:>8.5f}"
      f" {kgr_recon:>8.4f} {agreed_kgr_r:>12.4f}"
      f" {entropy_recon:>9.6f} {n_bits_total:>6}")
print(f"{'SHA-256 output':<25} {'N/A':>8} {'N/A':>8}"
      f" {kgr_pa:>8.4f} {'N/A':>12}"
      f" {entropy_pa:>9.6f} {total_bits_pa:>6}")

print(f"\n--- NIST RANDOMNESS (SUPPLEMENTARY) ---")
print(f"  Note: {n_bits_total} bit << 10^6 minimum NIST -> exploratory only")
print(f"  Frequency  : {nist_freq:.6f}  {'PASS' if nist_freq > 0.01 else 'FAIL'}")
print(f"  Runs       : {nist_run:.6f}  {'PASS' if nist_run  > 0.01 else 'FAIL'}")
print(f"  Cumul. sum : {nist_cum:.6f}  {'PASS' if nist_cum  > 0.01 else 'FAIL'}")
print(f"  FFT        : {nist_f:.6f}  {'PASS' if nist_f    > 0.01 else 'FAIL'}")
print(f"  Serial     : {nist_ser}")

print(f"\n--- CORR & GAP ---")
print(f"Pearson corr (raw)       : {np.corrcoef(rssi_alice, rssi_bob)[0,1]:.4f}")
print(f"Pearson corr (denoised)  : {corr_den:.4f}")
print(f"Median gap Alice-Bob     : {np.median(rssi_alice - rssi_bob):.2f} dB")
print(f"Offset correction        : {median_offset:.3f} dB")
print("=" * 65)
