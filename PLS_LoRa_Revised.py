"""
PLS-LoRa BASELINE — Revised [8]
Savitzky-Golay + Gray code 5-bit + Cascade
Perubahan dari versi asli:
  1. Chronological 60/20/20 split
  2. Gray code 5-bit (32 level)
  3. Threshold dari training partition
  4. Agreed KGR ditambahkan
  5. T_akuisisi = (N-1) × 1 detik
"""

import pandas as pd, numpy as np, math, hashlib
from time import perf_counter
from scipy.signal import savgol_filter
from scipy.special import erfc

FILE_PATH = "combined_v2i lurus 20.csv"   # ← ganti sesuai scenario
N_LEVELS  = 32; N_BITS = 5; SERIAL_M = 2

df = pd.read_csv(FILE_PATH)
rssi_alice = df['rssi_alice'].values.astype(float)
rssi_bob   = df['rssi_bob'].values.astype(float)
N = len(rssi_alice); T_AKUISISI = (N-1)*1.0

print("="*55)
print(f"FILE: {FILE_PATH}  |  N={N}  |  T_akuisisi={T_AKUISISI:.0f}s")
print(f"Corr raw: {np.corrcoef(rssi_alice,rssi_bob)[0,1]:.4f}"
      f"  |  Gap: {np.median(rssi_alice-rssi_bob):.2f} dB")
print("="*55)

n_train=int(N*0.60); n_val=int(N*0.20)
A_train=rssi_alice[:n_train]; B_train=rssi_bob[:n_train]
A_test=rssi_alice[n_train+n_val:]; B_test=rssi_bob[n_train+n_val:]
print(f"Split → Train:{n_train} | Val:{n_val} | Test:{len(A_test)}")

# SavGol denoising
t_pre0=perf_counter()
wl=min(5,len(A_train)); wl=wl if wl%2==1 else wl-1
A_tr_den=savgol_filter(A_train, wl, 2)
B_tr_den=savgol_filter(B_train, wl, 2)
wl_ts=min(5,len(A_test)); wl_ts=wl_ts if wl_ts%2==1 else max(3,wl_ts-1)
A_ts_den=savgol_filter(A_test, wl_ts, 2) if len(A_test)>=wl_ts else A_test
B_ts_den=savgol_filter(B_test, wl_ts, 2) if len(B_test)>=wl_ts else B_test
t_pre=perf_counter()-t_pre0
THR_MIN=A_tr_den.min(); THR_MAX=A_tr_den.max()

def ig(n,b=N_BITS):
    g=n^(n>>1); return [(g>>(b-1-i))&1 for i in range(b)]
def gq(sig,lo,hi,nl=N_LEVELS,nb=N_BITS):
    step=(hi-lo)/nl; out=[]
    for v in sig:
        lv=max(0,min(nl-1,int((v-lo)/step))); out.extend(ig(lv,nb))
    return np.array(out,dtype=int)
def kdr(a,b): n=min(len(a),len(b)); return float(np.sum(a[:n]!=b[:n])/n) if n else 1.0
def ent(b): p1=np.mean(b);p0=1-p1;h=0.0; (h:=-p1*np.log2(p1)) if p1>0 else None; (h:=h-p0*np.log2(p0)) if p0>0 else None; return h

t_q0=perf_counter()
key_a=gq(A_ts_den,THR_MIN,THR_MAX)
key_b=gq(B_ts_den,THR_MIN,THR_MAX)
n_bits=min(len(key_a),len(key_b)); key_a=key_a[:n_bits]; key_b=key_b[:n_bits]
t_q=perf_counter()-t_q0
T_quant=T_AKUISISI+t_pre+t_q; kdr_q=kdr(key_a,key_b)
kgr_q=n_bits/T_quant; agreed_q=kgr_q*(1-kdr_q)

print(f"\n--- QUANTISASI ---")
print(f"Bits: {n_bits}  KDR: {kdr_q:.5f}  KGR: {kgr_q:.4f}  Agreed KGR: {agreed_q:.4f} bps")

# Cascade
def dyn_block(n): return 16 if n<=256 else 32 if n<=512 else 64 if n<=1024 else 128
def par(b): return int(np.sum(b)%2)
def bsearch(ab,bb,start,c):
    l,r=0,len(ab)-1
    while l<=r:
        if l==r:
            if ab[l]!=bb[l]: c[start+l]=ab[l]
            break
        m=(l+r)//2; (r:=m) if par(ab[l:m+1])!=par(bb[l:m+1]) else (l:=m+1)

t_r0=perf_counter()
corr=key_b.copy(); bs=dyn_block(n_bits)
while True:
    for i in range(0,n_bits,bs):
        ab=key_a[i:i+bs]; bb=corr[i:i+bs]
        if par(ab)!=par(bb): bsearch(ab,bb,i,corr)
    if len(np.where(key_a!=corr)[0])==0 or bs==1: break
    bs=max(1,bs//2)
t_recon=perf_counter()-t_r0
T_recon=T_quant+t_recon; kdr_r=kdr(corr,key_b); kgr_r=n_bits/T_recon
agreed_r=kgr_r*(1-kdr_r)
print(f"\n--- REKONSILIASI ---")
print(f"KDR: {kdr_r:.5f}  BAR: {1-kdr_r:.5f}  KGR: {kgr_r:.4f}  Agreed KGR: {agreed_r:.4f} bps")

def nist_f(b):
    n=len(b); return 0.0 if n<16 else float(erfc(abs(sum(1 if x else -1 for x in b))/math.sqrt(2*n)))
def nist_c(b):
    n=len(b);
    if n<100: return 0.0
    x=np.array([1 if v else -1 for v in b]); s=np.cumsum(x); z=int(np.max(np.abs(s)))
    if z==0: return 1.0
    s1=sum(math.exp(-((4*k+1)**2*z**2)/(2*n)) for k in range(int(np.floor((-n/z+1)/4)),int(np.floor((n/z-1)/4))+1))
    s2=sum(math.exp(-((4*k+3)**2*z**2)/(2*n)) for k in range(int(np.floor((-n/z-3)/4)),int(np.floor((n/z-1)/4))+1))
    return float(max(0,min(1,1-s1+s2)))

t_pa0=perf_counter()
ha=hashlib.sha256(''.join(map(str,corr)).encode()).hexdigest()
hb=hashlib.sha256(''.join(map(str,key_b)).encode()).hexdigest()
t_pa=perf_counter()-t_pa0
pa_bits=256; T_pa=T_recon+t_pa; kgr_pa=pa_bits/T_pa
ent_val=ent(corr)
freq_v=nist_f(corr); cum_v=nist_c(corr)
bit_pa = bin(int(ha, 16))[2:].zfill(256)
entro_pa=ent(np.array([int(b) for b in bit_pa]))

print(f"\n--- SHA-256 ---")
print(f"Match: {ha==hb} | KGR: {kgr_pa:.4f} bps | Entropy: {ent_val:.5f}")
print(f"Freq: {freq_v:.5f} | Cumsum: {cum_v:.5f}")

print(f"\n{'='*55}\nRINGKASAN PLS-LoRa [8]")
print(f"  T_akuisisi : {T_AKUISISI:.0f}s")
print(f"  KDR quant  : {kdr_q:.5f}")
print(f"  Agreed KGR : {agreed_q:.4f} bps  |  Bits: {n_bits}")
print(f"  Entropy PA : {entro_pa:.5f}")
print(f"  T_preproc  : {t_pre*1000:.2f}ms  |  T_quant: {t_q*1000:.3f}ms")
print(f"  T_Recon    : {t_recon*1000:.4f}ms")
print(f"  T_Ampli    : {t_pa*1000:.4f}ms")
print(f"  T_Total    : {T_pa:.4f}s")
