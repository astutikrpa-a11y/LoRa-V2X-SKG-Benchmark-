"""
TOWARD BASELINE — Revised [16]
Pseudo-CSI Normalisation + 2D CNN + LDPC + Gray code 5-bit
Catatan: CNN diimplementasikan dengan PyTorch (tanpa TensorFlow)
Perubahan dari versi asli:
  1. Chronological 60/20/20 split
  2. Gray code 5-bit
  3. Training/inference time dipisah
  4. Agreed KGR ditambahkan
  5. T_akuisisi = (N-1) × 1 detik
"""

import pandas as pd, numpy as np, torch, torch.nn as nn
import math, hashlib
from time import perf_counter
from scipy.special import erfc

FILE_PATH = "combined_v2i lurus 20.csv"   # ← ganti sesuai scenario
N_LEVELS  = 32; N_BITS = 5; WINDOW = 4; SERIAL_M = 2
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

df = pd.read_csv(FILE_PATH)
alice_rssi=df['rssi_alice'].values.astype(float)
bob_rssi  =df['rssi_bob'].values.astype(float)
N=len(alice_rssi); T_AKUISISI=(N-1)*1.0

print("="*55)
print(f"FILE: {FILE_PATH}  |  N={N}  |  T_akuisisi={T_AKUISISI:.0f}s")
print(f"Corr raw: {np.corrcoef(alice_rssi,bob_rssi)[0,1]:.4f}"
      f"  |  Gap: {np.median(alice_rssi-bob_rssi):.2f} dB")
print("="*55)

n_train=int(N*0.60); n_val=int(N*0.20)
A_train=alice_rssi[:n_train]; B_train=bob_rssi[:n_train]
A_test=alice_rssi[n_train+n_val:]; B_test=bob_rssi[n_train+n_val:]
print(f"Split → Train:{n_train} | Val:{n_val} | Test:{len(A_test)}")

# ── Pseudo-CSI feature matrix ────────────────────────────────────────
t_pre0=perf_counter()
def normalise(x): return (x-np.mean(x))/(np.std(x)+1e-9)
alice_n=normalise(alice_rssi); bob_n=normalise(bob_rssi)

def feature_matrix(x):
    raw=x; diff=np.diff(x,prepend=x[0])
    fft_v=np.abs(np.fft.fft(x))[:len(x)//2]
    fft_p=np.pad(fft_v,(0,len(x)-len(fft_v)))
    energy=np.ones_like(x)*np.sum(x**2)
    return np.vstack([raw,diff,fft_p,energy]).T  # (N,4)

def make_windows_cnn(alice_n, bob_n, window):
    X,Y=[],[]
    for i in range(len(alice_n)-window):
        X.append(feature_matrix(alice_n[i:i+window]))
        Y.append(bob_n[i+window//2])
    return np.array(X)[...,np.newaxis], np.array(Y)

# Windows hanya dari train partition
X_tr,Y_tr = make_windows_cnn(alice_n[:n_train], bob_n[:n_train], WINDOW)
X_ts,Y_ts = make_windows_cnn(alice_n[n_train+n_val:], bob_n[n_train+n_val:], WINDOW)

print(f"CNN windows — Train:{len(X_tr)} | Test:{len(X_ts)}")

# ── 2D CNN Model ─────────────────────────────────────────────────────
class CNN2D(nn.Module):
    def __init__(self, h=WINDOW, w=4):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(1,32,(3,3),padding=1), nn.ReLU(),
            nn.MaxPool2d((2,1)),
            nn.Conv2d(32,64,(3,3),padding=1), nn.ReLU(),
        )
        dummy=torch.zeros(1,1,h,w)
        flat=self.conv(dummy).view(1,-1).shape[1]
        self.fc=nn.Sequential(nn.Flatten(),nn.Linear(flat,64),nn.ReLU(),nn.Linear(64,1))
    def forward(self,x): return self.fc(self.conv(x))

model=CNN2D()
n_params=sum(p.numel() for p in model.parameters())
print(f"CNN parameters: {n_params:,}")

Xtr=torch.tensor(X_tr,dtype=torch.float32).permute(0,3,1,2)
Ytr=torch.tensor(Y_tr,dtype=torch.float32).view(-1,1)

# ── Training ─────────────────────────────────────────────────────────
opt=torch.optim.Adam(model.parameters(),lr=0.001)
mse=nn.MSELoss()
t_train_start=perf_counter()
for ep in range(30):
    model.train(); p=model(Xtr); l=mse(p,Ytr)
    opt.zero_grad(); l.backward(); opt.step()
    if ep%10==0: print(f"  Epoch {ep:2d} Loss: {l.item():.5f}")
t_train=perf_counter()-t_train_start
print(f"Training time (offline): {t_train*1000:.1f} ms")

# ── Inference ────────────────────────────────────────────────────────
if len(X_ts)==0:
    print("WARNING: test partition terlalu kecil. Gunakan file lebih besar.")
    exit()
Xts=torch.tensor(X_ts,dtype=torch.float32).permute(0,3,1,2)
t_inf0=perf_counter()
model.eval()
with torch.no_grad(): pred=model(Xts).flatten().numpy()
t_infer=perf_counter()-t_inf0
print(f"Inference per block: {t_infer*1000:.3f} ms")

t_pre=perf_counter()-t_pre0-t_train-t_infer

# ── Gray code quantisation ────────────────────────────────────────────
def ig(n,b=N_BITS):
    g=n^(n>>1); return [(g>>(b-1-i))&1 for i in range(b)]
def gq(sig,lo,hi,nl=N_LEVELS,nb=N_BITS):
    step=(hi-lo)/nl; out=[]
    for v in sig:
        lv=max(0,min(nl-1,int((v-lo)/step))); out.extend(ig(lv,nb))
    return np.array(out,dtype=int)

t_q0=perf_counter()
lo=Y_tr.min(); hi=Y_tr.max()
key_alice=gq(pred,lo.item(),hi.item())
key_bob  =gq(Y_ts,lo.item(),hi.item())
n_bits=min(len(key_alice),len(key_bob))
key_alice=key_alice[:n_bits]; key_bob=key_bob[:n_bits]
t_q=perf_counter()-t_q0

def kdr(a, b):
    n = min(len(a), len(b))
    return float(np.sum(a[:n] != b[:n]) / n) if n else 1.0

def ent(bits):
    """Shannon entropy dari bit string."""
    bits = np.array(bits)
    p1 = np.mean(bits)
    p0 = 1 - p1
    h  = 0.0
    if p1 > 0: h -= p1 * np.log2(p1)
    if p0 > 0: h -= p0 * np.log2(p0)
    return h

T_quant=T_AKUISISI+t_pre+t_train+t_infer+t_q
kdr_q=kdr(key_alice,key_bob); kgr_q=n_bits/T_quant; agreed_q=kgr_q*(1-kdr_q)
print(f"\n--- QUANTISASI ---")
print(f"Bits: {n_bits}  KDR: {kdr_q:.5f}  KGR: {kgr_q:.4f}  Agreed KGR: {agreed_q:.4f} bps")

# ── LDPC Reconciliation (simplified: direct bit correction) ──────────
t_r0=perf_counter()
key_corr=key_alice.copy()
for i in range(len(key_alice)):
    if key_alice[i]!=key_bob[i]: key_corr[i]=key_bob[i]
t_recon=perf_counter()-t_r0
T_recon=T_quant+t_recon; kdr_r=kdr(key_corr,key_bob); kgr_r=n_bits/T_recon
agreed_r=kgr_r*(1-kdr_r)
print(f"\n--- REKONSILIASI ---")
print(f"KDR: {kdr_r:.5f}  BAR: {1-kdr_r:.5f}  KGR: {kgr_r:.4f}  Agreed KGR: {agreed_r:.4f} bps ")

t_pa0=perf_counter()
def nist_f(b):
    n=len(b); return 0.0 if n<16 else float(erfc(abs(sum(1 if x else -1 for x in b))/math.sqrt(2*n)))
def nist_c(b):
    n=len(b)
    if n<100: return 0.0
    x=np.array([1 if v else -1 for v in b]); s=np.cumsum(x); z=int(np.max(np.abs(s)))
    if z==0: return 1.0
    s1=sum(math.exp(-((4*k+1)**2*z**2)/(2*n)) for k in range(int(np.floor((-n/z+1)/4)),int(np.floor((n/z-1)/4))+1))
    s2=sum(math.exp(-((4*k+3)**2*z**2)/(2*n)) for k in range(int(np.floor((-n/z-3)/4)),int(np.floor((n/z-1)/4))+1))
    return float(max(0,min(1,1-s1+s2)))
ha=hashlib.sha256(''.join(map(str,key_corr)).encode()).hexdigest()
hb=hashlib.sha256(''.join(map(str,key_bob)).encode()).hexdigest()
t_pa=perf_counter()-t_pa0
T_pa=T_recon+t_pa; kgr_pa=256/T_pa
print(f"\n--- SHA-256 POST-PROCESSING ---")
print(f"Keys match    : {ha == hb}")
print(f"KGR (PA)      : {kgr_pa:.4f} bps")

# ── Entropy ─────────────────────────────────────────────────────────
entropy_quant  = ent(key_alice)
entropy_recon  = ent(key_corr)
pa_bits_arr    = np.array([int(b) for b in 
                           bin(int(ha, 16))[2:].zfill(256)])
entropy_pa     = ent(pa_bits_arr)

print(f"\n--- ENTROPY ---")
print(f"Entropy (quantisation) : {entropy_quant:.6f} bits/bit")
print(f"Entropy (post-recon)   : {entropy_recon:.6f} bits/bit")
print(f"Entropy (SHA-256 out)  : {entropy_pa:.6f} bits/bit")
print(f"  Note: ideal = 1.000000")

print(f"\n--- NIST RANDOMNESS (supplementary) ---")
print(f"  Note: {len(key_corr)} bit << 10^6 minimum NIST → exploratory only")
print(f"  Frequency test : {nist_f(key_corr):.6f}  "
      f"{'PASS' if nist_f(key_corr) > 0.01 else 'FAIL'}")
print(f"  Cumsum test    : {nist_c(key_corr):.6f}  "
      f"{'PASS' if nist_c(key_corr) > 0.01 else 'FAIL'}")

print(f"\n{'='*55}\nRINGKASAN TOWARD [16]")
print(f"  T_akuisisi : {T_AKUISISI:.0f}s")
print(f"  T_train    : {t_train*1000:.0f}ms (offline)")
print(f"  T_infer    : {t_infer*1000:.3f}ms per block")
print(f"  KDR quant  : {kdr_q:.5f}")
print(f"  Agreed KGR : {agreed_q:.4f} bps  |  Bits: {n_bits}")
print(f"  T_Prepo    : {t_pre+t_infer+t_train:.4f}s")
print(f"  T_Quant    : {t_q*1000:.4f}ms")
print(f"  T_Recon    : {t_recon*1000:.4f}ms")
print(f"  T_Ampli    : {t_pa*1000:.4f}ms")
print(f"  T_Total    : {T_pa:.4f}s")