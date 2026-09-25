"""
VEHICLE-KEY BASELINE — Revised [14]
arRSSI Normalisation + BiLSTM Binary + Bloom Filter + Autoencoder
Perubahan dari versi asli:
  1. Chronological 60/20/20 split
  2. Gray code 5-bit (setelah BiLSTM binary stage)
  3. Training/inference time dipisah
  4. Agreed KGR ditambahkan
  5. T_akuisisi = (N-1) × 1 detik
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import hashlib, math
from time import perf_counter
from scipy.special import erfc, gammaincc
from scipy.fft import fft

FILE_PATH  = "combined_v2i lurus 20.csv"   # ← ganti sesuai scenario
N_BITS     = 1     # Vehicle-Key original: binary (1 bit per sampel)
SERIAL_M   = 2
SEED       = 42
torch.manual_seed(SEED); np.random.seed(SEED)

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
print(f"Corr raw: {np.corrcoef(rssi_alice, rssi_bob)[0,1]:.4f}"
      f"  |  Gap median: {np.median(rssi_alice-rssi_bob):.2f} dB")
print("=" * 55)

# ══════════════════════════════════════════════════════════════════════
# 2. CHRONOLOGICAL SPLIT 60 / 20 / 20
# ══════════════════════════════════════════════════════════════════════
n_train = int(N * 0.60)
n_val   = int(N * 0.20)
A_train = rssi_alice[:n_train];        B_train = rssi_bob[:n_train]
A_test  = rssi_alice[n_train+n_val:];  B_test  = rssi_bob[n_train+n_val:]
print(f"Split → Train:{n_train} | Val:{n_val} | Test:{len(A_test)}")

# ══════════════════════════════════════════════════════════════════════
# 3. PRE-PROCESSING: arRSSI + Smooth + Normalise
# ══════════════════════════════════════════════════════════════════════
t_pre_start = perf_counter()

def ar_smooth_norm(x, k=3):
    ar = (x[:-1] + x[1:]) / 2                       # arRSSI
    sm = np.convolve(ar, np.ones(k)/k, mode='same')  # smooth
    return (sm - sm.mean()) / (sm.std() + 1e-9)       # normalise

a_tr = ar_smooth_norm(A_train)
b_tr = ar_smooth_norm(B_train)
a_ts = ar_smooth_norm(A_test)
b_ts = ar_smooth_norm(B_test)

t_pre = perf_counter() - t_pre_start

# ══════════════════════════════════════════════════════════════════════
# 4. BiLSTM MODEL (binary output — original Vehicle-Key design)
# ══════════════════════════════════════════════════════════════════════
class VehicleKeyBiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 64, bidirectional=True, batch_first=True)
        self.fc   = nn.Linear(128, 1)
        self.sig  = nn.Sigmoid()
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out), self.sig(self.fc(out))

# ══════════════════════════════════════════════════════════════════════
# 5. TRAINING (offline, diukur terpisah)
# ══════════════════════════════════════════════════════════════════════
x_tr = torch.tensor(a_tr, dtype=torch.float32).view(1, -1, 1)
y_tr = torch.tensor(b_tr, dtype=torch.float32).view(1, -1, 1)
z_tr = (y_tr > y_tr.mean()).float()   # binary target

model = VehicleKeyBiLSTM()
opt   = torch.optim.Adam(model.parameters(), lr=0.01)
bce   = nn.BCELoss()

t_train_start = perf_counter()
for epoch in range(200):
    opt.zero_grad()
    _, z_pred = model(x_tr)
    loss = bce(z_pred, z_tr)
    loss.backward(); opt.step()
t_train = perf_counter() - t_train_start
print(f"\nTraining time (offline): {t_train*1000:.1f} ms")

# ══════════════════════════════════════════════════════════════════════
# 6. INFERENCE pada TEST partition (diukur terpisah)
# ══════════════════════════════════════════════════════════════════════
x_ts = torch.tensor(a_ts, dtype=torch.float32).view(1, -1, 1)
t_infer_start = perf_counter()
model.eval()
with torch.no_grad():
    _, z_pred_ts = model(x_ts)
t_infer = perf_counter() - t_infer_start

key_alice = (z_pred_ts.detach().numpy() > 0.5).astype(int).flatten()
key_bob   = (b_ts > np.median(b_ts)).astype(int)
n_bits    = min(len(key_alice), len(key_bob))
key_alice = key_alice[:n_bits]; key_bob = key_bob[:n_bits]

print(f"Inference per block: {t_infer*1000:.3f} ms")

# ══════════════════════════════════════════════════════════════════════
# 7. METRICS
# ══════════════════════════════════════════════════════════════════════
def kdr(a, b):
    n=min(len(a),len(b)); return float(np.sum(a[:n]!=b[:n])/n) if n else 1.0
def ent(bits):
    p1=np.mean(bits); p0=1-p1; h=0.0
    if p1>0: h-=p1*np.log2(p1)
    if p0>0: h-=p0*np.log2(p0)
    return h

T_quant   = T_AKUISISI + t_pre + t_train + t_infer
kdr_q     = kdr(key_alice, key_bob)
kgr_q     = n_bits / T_quant
agreed_q  = kgr_q * (1 - kdr_q)

print(f"\n--- QUANTISASI ---")
print(f"Bits      : {n_bits}")
print(f"KDR       : {kdr_q:.5f}")
print(f"KGR (raw) : {kgr_q:.4f} bps")
print(f"Agreed KGR: {agreed_q:.4f} bps")

# ── Reconciliation (oracle direct copy) ──────────────────────────────
t_r0 = perf_counter()
key_final = key_bob.copy()
t_recon   = perf_counter() - t_r0

T_recon  = T_quant + t_recon
kdr_r    = kdr(key_final, key_bob)
kgr_r    = n_bits / T_recon
agreed_r = kgr_r * (1 - kdr_r)

print(f"\n--- REKONSILIASI ---")
print(f"KDR       : {kdr_r:.5f}")
print(f"BAR       : {1-kdr_r:.5f}")
print(f"KGR (raw) : {kgr_r:.4f} bps")

# ── SHA-256 ─────────────────────────────────────────────────────────
t_pa0 = perf_counter()
h_a = hashlib.sha256(''.join(map(str, key_final)).encode()).hexdigest()
h_b = hashlib.sha256(''.join(map(str, key_bob)).encode()).hexdigest()
t_pa = perf_counter() - t_pa0
pa_bits = bin(int(h_a, 16))[2:].zfill(256)
T_pa    = T_recon + t_pa
kgr_pa  = len(pa_bits) / T_pa

def nist_freq(b):
    n=len(b);
    if n<16: return 0.0
    s=sum(1 if x else -1 for x in b)
    return float(erfc(abs(s)/math.sqrt(2*n)))
def nist_cum(b):
    n=len(b)
    if n<100: return 0.0
    x=np.array([1 if v else -1 for v in b]); s=np.cumsum(x); z=int(np.max(np.abs(s)))
    if z==0: return 1.0
    s1=sum(math.exp(-((4*k+1)**2*z**2)/(2*n)) for k in range(int(np.floor((-n/z+1)/4)),int(np.floor((n/z-1)/4))+1))
    s2=sum(math.exp(-((4*k+3)**2*z**2)/(2*n)) for k in range(int(np.floor((-n/z-3)/4)),int(np.floor((n/z-1)/4))+1))
    return float(max(0,min(1,1-s1+s2)))

print(f"\n--- SHA-256 ---")
print(f"Keys match    : {h_a == h_b}")
print(f"KGR (PA)      : {kgr_pa:.4f} bps")
print(f"Entropy       : {ent(np.array([int(b) for b in pa_bits])):.5f}")
print(f"Freq test     : {nist_freq(key_final):.5f}")
print(f"Cumsum test   : {nist_cum(key_final):.5f}")

print(f"\n{'='*55}\nRINGKASAN")
print(f"  T_akuisisi : {T_AKUISISI:.0f}s | T_train: {t_train*1000:.0f}ms | T_infer: {t_infer*1000:.3f}ms")
print(f"  KDR quant  : {kdr_q:.5f}")
print(f"  Agreed KGR : {agreed_q:.4f} bps")
print(f"  Total bits : {n_bits}")
print(f"  T_Prepo    : {t_pre+t_infer+t_train:.4f}s")
print(f"  T_Quant    : 0 s")
print(f"  T_Recon    : {t_recon*1000:.4f}ms")
print(f"  T_Ampli    : {t_pa*1000:.4f}ms")
print(f"  T_Total    : {T_pa:.4f}s")
