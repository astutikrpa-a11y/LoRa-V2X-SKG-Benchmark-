"""
ABLATION STUDY — FINAL (Sesuai Keputusan SavGol)
Mengisolasi kontribusi setiap komponen preprocessing:

  A1: Raw RSSI + quantization               ← baseline tanpa preprocessing
  A2: Offset normalization + quantization   ← kontribusi offset saja
  A3: Offset + Wavelet + quantization       ← wavelet di atas offset
  A4: Offset + Moving Average + quantization← MA sebagai alternatif smoothing
  A5: Offset + Savitzky-Golay + quantization← PROPOSED ★ (paling stabil)
  A6: Offset + Kalman + quantization        ← Kalman sebagai alternatif

Tambahan (negative finding):
  A7: Offset + BiLSTM + quantization        ← ML approach (gagal, dilaporkan jujur)
  A8: Offset + Wavelet + BiLSTM + quant     ← Full BiLSTM (degenerate)

Tujuan ablation:
  A1→A2: Berapa kontribusi offset normalization?
  A2→A5: Berapa kontribusi SavGol di atas offset?
  A3 vs A4 vs A5 vs A6: Metode smoothing mana yang terbaik?
  A5 vs A7: Apakah SavGol lebih baik dari BiLSTM? (seharusnya ya)
"""

import pandas as pd
import numpy as np
import pywt
import torch
import torch.nn as nn
import warnings
import os
from time import perf_counter
from scipy.signal import savgol_filter

warnings.filterwarnings('ignore')

# ─── KONFIGURASI ──────────────────────────────────────────────────
N_LEVELS = 32
N_BITS   = 5
SEQ_LEN  = 5
HIDDEN   = 16
N_SEEDS  = 20
EPOCHS   = 100

FILES = {
    'V2I_Lurus_20':      'combined_v2i lurus 20.csv',
    'V2I_Lurus_30':      'combined_v2i lurus 30.csv',
    'V2I_Belok_20':      'combined_v2i belok 20.csv',
    'V2I_Belok_30':      'combined_v2i belok 30.csv',
    'V2V_Berlawanan_20': 'combined_V2V berlawanan20.csv',
    'V2V_Berlawanan_30': 'combined_V2V berlawanan30.csv',
    'V2V_Searah_20':     'combined_V2V searah20.csv',
    'V2V_Searah_30':     'combined_V2V searah30.csv',
}

CONFIGS = {
    'A1': 'Raw RSSI + quantization',
    'A2': 'Offset + quantization',
    'A3': 'Offset + Wavelet + quantization',
    'A4': 'Offset + Moving Average + quantization',
    'A5': 'Offset + Savitzky-Golay + quantization ★',
    'A6': 'Offset + Kalman + quantization',
    'A7': 'Offset + BiLSTM + quantization (negative)',
    'A8': 'Offset + Wavelet + BiLSTM + quantization (negative)',
}

ENTROPY_MIN = 0.5   # threshold degenerate detection

# ── Helpers ───────────────────────────────────────────────────────
def int_to_gray(n, bits=N_BITS):
    g = n ^ (n >> 1)
    return [(g >> (bits-1-i)) & 1 for i in range(bits)]

def gray_q(sig, lo, hi, nl=N_LEVELS, nb=N_BITS):
    step=(hi-lo)/nl; out=[]
    for v in sig:
        lv=max(0,min(nl-1,int((v-lo)/step)))
        out.extend(int_to_gray(lv,nb))
    return np.array(out,dtype=int)

def kdr_fn(a, b):
    n=min(len(a),len(b))
    return float(np.sum(a[:n]!=b[:n])/n) if n else 1.0

def entropy_fn(bits):
    bits=np.array(bits,dtype=int)
    p1=np.mean(bits); p0=1-p1; h=0.0
    if p1>0: h-=p1*np.log2(p1)
    if p0>0: h-=p0*np.log2(p0)
    return h

def corr_fn(a, b):
    n=min(len(a),len(b))
    return float(np.corrcoef(a[:n],b[:n])[0,1]) if n>1 else 0.0

def wavelet_den(sig):
    sig=np.array(sig,dtype=float)
    if len(sig)<4: return sig
    c=pywt.wavedec(sig,'db4',level=1)
    thr=(np.median(np.abs(c[-1]))/0.6745)*np.sqrt(2*np.log(max(len(sig),2)))
    nc=[c[0]]+[pywt.threshold(x,thr,'soft') for x in c[1:]]
    return pywt.waverec(nc,'db4')[:len(sig)]

def savgol_den(sig, w=5, p=2):
    sig=np.array(sig,dtype=float); n=len(sig)
    ww=min(w,n); ww=ww if ww%2==1 else max(3,ww-1)
    return savgol_filter(sig,ww,min(p,ww-1)) if n>=ww else sig

def kalman_fn(sig, Q=1e-3, R=0.1):
    x=float(sig[0]); P=1.0; out=[]
    for z in sig:
        P+=Q; K=P/(P+R); x+=K*(z-x); P=(1-K)*P; out.append(x)
    return np.array(out)

def moving_avg(sig, k=3):
    return np.convolve(sig,np.ones(k)/k,'same')

def make_windows(x, y, sl):
    X,Y=[],[]
    for i in range(len(x)-sl):
        X.append(x[i:i+sl]); Y.append(y[i:i+sl])
    return np.array(X),np.array(Y)

# ── BiLSTM Model ──────────────────────────────────────────────────
class BiLSTM(nn.Module):
    def __init__(self,h=HIDDEN,nb=N_BITS):
        super().__init__()
        self.lstm=nn.LSTM(1,h,batch_first=True,bidirectional=True)
        self.fc=nn.Linear(h*2,nb); self.sig=nn.Sigmoid()
        self.drp=nn.Dropout(0.1)
    def forward(self,x):
        o,_=self.lstm(x); return self.sig(self.fc(self.drp(o[:,-1,:])))

# ── Evaluate one config on one file ───────────────────────────────
def eval_one(cfg, A_tr, B_tr, A_ts, B_ts, T, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    t0=perf_counter()

    a_tr=A_tr.copy(); b_tr=B_tr.copy()
    a_ts=A_ts.copy(); b_ts=B_ts.copy()

    # Step 1: Offset (semua kecuali A1)
    if cfg != 'A1':
        off  = np.median(a_tr - b_tr)
        a_tr = a_tr - off
        a_ts = a_ts - off

    # Step 2: Smoothing sesuai konfigurasi
    if cfg == 'A3':   # Offset + Wavelet
        a_tr=wavelet_den(a_tr); b_tr=wavelet_den(b_tr)
        a_ts=wavelet_den(a_ts); b_ts=wavelet_den(b_ts)
    elif cfg == 'A4': # Offset + Moving Average
        a_tr=moving_avg(a_tr); b_tr=moving_avg(b_tr)
        a_ts=moving_avg(a_ts); b_ts=moving_avg(b_ts)
    elif cfg == 'A5': # Offset + SavGol (PROPOSED)
        a_tr=savgol_den(a_tr); b_tr=savgol_den(b_tr)
        a_ts=savgol_den(a_ts); b_ts=savgol_den(b_ts)
    elif cfg == 'A6': # Offset + Kalman
        a_tr=kalman_fn(a_tr); b_tr=kalman_fn(b_tr)
        a_ts=kalman_fn(a_ts); b_ts=kalman_fn(b_ts)
    elif cfg in ['A7','A8']: # BiLSTM (dengan Wavelet untuk A8)
        if cfg == 'A8':
            a_tr=wavelet_den(a_tr); b_tr=wavelet_den(b_tr)
            a_ts=wavelet_den(a_ts); b_ts=wavelet_den(b_ts)

    lo=a_tr.min(); hi=a_tr.max()
    if lo==hi: return None
    t_pre=perf_counter()-t0

    # Step 3: BiLSTM untuk A7/A8
    t_train=0.0; t_infer=0.0
    if cfg in ['A7','A8'] and len(a_tr)>SEQ_LEN+2 and len(a_ts)>SEQ_LEN:
        mu=a_tr.mean(); sd=a_tr.std()+1e-9
        atn=(a_tr-mu)/sd; btn=(b_tr-b_tr.mean())/(b_tr.std()+1e-9)
        bgray=np.array([int_to_gray(max(0,min(N_LEVELS-1,
            int((v-lo)/(hi-lo+1e-9)*N_LEVELS))),N_BITS)
            for v in b_tr[SEQ_LEN:]])
        X_tr,Y_tr=make_windows(atn,btn,SEQ_LEN)
        if len(X_tr)<2: return None
        Xtr=torch.tensor(X_tr,dtype=torch.float32).unsqueeze(-1)
        Ytr=torch.tensor(Y_tr,dtype=torch.float32).unsqueeze(-1)
        Btr=torch.tensor(bgray[:len(X_tr)],dtype=torch.float32)
        m=BiLSTM(); opt=torch.optim.Adam(m.parameters(),lr=0.001,weight_decay=1e-4)
        bce=nn.BCELoss(); mse=nn.MSELoss()
        sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=8,factor=0.5)
        best=1e9; no_imp=0
        t_train=perf_counter()
        for ep in range(EPOCHS):
            m.train(); pb=m(Xtr)
            pr,_=m.lstm(Xtr); loss=0.9*mse(pr,Ytr)+0.1*bce(pb,Btr)
            opt.zero_grad(); loss.backward(); opt.step(); sch.step(loss.item())
            if loss.item()<best-1e-5: best=loss.item(); no_imp=0
            else: no_imp+=1
            if no_imp>=15: break
        t_train=perf_counter()-t_train
        X_ts,_=make_windows((a_ts-mu)/sd,(a_ts-mu)/sd,SEQ_LEN)
        if len(X_ts)==0: return None
        Xts=torch.tensor(X_ts,dtype=torch.float32).unsqueeze(-1)
        m.eval()
        with torch.no_grad(): _=m(Xts[:1])
        t_infer=perf_counter()
        with torch.no_grad(): pred=m(Xts)
        t_infer=perf_counter()-t_infer
        ba=(pred.numpy()>0.5).astype(int).flatten()
        bb=np.array([int_to_gray(max(0,min(N_LEVELS-1,
            int((v-lo)/(hi-lo+1e-9)*N_LEVELS))),N_BITS)
            for v in b_ts[SEQ_LEN:len(X_ts)+SEQ_LEN]]).flatten()
    else:
        # Direct Gray quantisation
        t_q=perf_counter()
        ba=gray_q(a_ts,lo,hi); bb=gray_q(b_ts,lo,hi)
        t_infer=perf_counter()-t_q

    n=min(len(ba),len(bb))
    if n==0: return None
    ba=ba[:n]; bb=bb[:n]

    k=kdr_fn(ba,bb)
    ent=entropy_fn(ba)
    kgr=n/(T+t_pre+t_train+t_infer)
    degenerate = ent < ENTROPY_MIN

    return {
        'kdr':kdr_fn(ba,bb),
        'kdr_sd':0.0,
        'kgr':kgr,
        'agreed':kgr*(1-k),
        'entropy':ent,
        'corr':corr_fn(a_ts,b_ts),
        'n_bits':n,
        'degenerate':degenerate,
        't_train_ms':t_train*1000,
        't_infer_ms':t_infer*1000,
    }

# ══════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════
all_results={cfg:{} for cfg in CONFIGS}

print("="*70)
print("ABLATION STUDY — SavGol Final (A1-A8)")
print("Non-ML: deterministic | ML (A7/A8): 20 seeds")
print("="*70)

for label, fpath in FILES.items():
    if not os.path.exists(fpath): continue
    df=pd.read_csv(fpath)
    A=df['rssi_alice'].values.astype(float)
    B=df['rssi_bob'].values.astype(float)
    N=len(A); T=(N-1)*1.0
    n_tr=int(N*0.50); n_v=int(N*0.25)
    A_tr=A[:n_tr]; B_tr=B[:n_tr]
    A_ts=A[n_tr+n_v:]; B_ts=B[n_tr+n_v:]
    if len(A_ts)<2: continue

    print(f"\n[{label}] N={N}")
    for cfg in CONFIGS:
        use_ml=(cfg in ['A7','A8'])
        if use_ml:
            runs=[]
            for s in range(N_SEEDS):
                r=eval_one(cfg,A_tr,B_tr,A_ts,B_ts,T,seed=s)
                if r and not r['degenerate']: runs.append(r)
                elif r and r['degenerate']:   runs.append({**r,'kdr':float('nan')})
            if runs:
                valid=[r for r in runs if not np.isnan(r['kdr'])]
                degen_pct=(N_SEEDS-len(valid))/N_SEEDS*100
                if valid:
                    res={k:np.mean([r[k] for r in valid if k in r and not np.isnan(r[k])])
                         for k in valid[0] if k not in ['degenerate']}
                    res['kdr_sd']=np.std([r['kdr'] for r in valid])
                    res['degenerate']=degen_pct>50
                    res['degen_pct']=degen_pct
                    all_results[cfg][label]=res
                    flag="⚠ DEGEN" if degen_pct>50 else f"({degen_pct:.0f}% degen)"
                    print(f"  {cfg}: KDR={res['kdr']:.4f}±{res['kdr_sd']:.4f} H={res['entropy']:.3f} {flag}")
                else:
                    print(f"  {cfg}: ALL DEGENERATE")
        else:
            r=eval_one(cfg,A_tr,B_tr,A_ts,B_ts,T,seed=0)
            if r:
                r['kdr_sd']=0.0; r['degen_pct']=0.0
                all_results[cfg][label]=r
                flag="⚠ DEGEN" if r['degenerate'] else ""
                print(f"  {cfg}: KDR={r['kdr']:.4f} H={r['entropy']:.4f} {flag}")

# ── Print table ───────────────────────────────────────────────────
print("\n\n"+"="*80)
print("ABLATION TABLE — Median (IQR) lintas semua file")
print("★ = proposed | ⚠ = degenerate (entropy < 0.5)")
print("="*80)
hdr=(f"{'ID':<4} {'Konfigurasi':<45} {'KDR':>12} "
     f"{'KGR':>8} {'Agreed KGR':>11} {'Entropy':>9} {'N':>4}")
print(hdr); print("-"*len(hdr))

rows=[]
for cfg,label in CONFIGS.items():
    vals=all_results[cfg]
    if not vals: continue
    kdr_list=[v['kdr'] for v in vals.values() if not np.isnan(v.get('kdr',np.nan))]
    kgr_list=[v['kgr'] for v in vals.values()]
    agr_list=[v['agreed'] for v in vals.values()]
    ent_list=[v['entropy'] for v in vals.values()]
    sd_list =[v.get('kdr_sd',0) for v in vals.values()]
    degen   =any(v.get('degenerate',False) for v in vals.values())
    if not kdr_list: continue
    med_kdr=np.median(kdr_list)
    iqr_kdr=np.subtract(*np.percentile(kdr_list,[75,25])) if len(kdr_list)>1 else 0
    med_kgr=np.median(kgr_list)
    med_agr=np.median(agr_list)
    med_ent=np.median(ent_list)
    mean_sd=np.mean(sd_list)
    n_f=len(kdr_list)
    use_ml=(cfg in ['A7','A8'])
    if use_ml: kdr_str=f"{med_kdr:.4f}±{mean_sd:.4f}"
    else:      kdr_str=f"{med_kdr:.4f}(IQR {iqr_kdr:.4f})"
    flag=" ⚠" if degen else ""
    print(f"{cfg:<4} {label:<45} {kdr_str:>12}{flag} "
          f"{med_kgr:>8.4f} {med_agr:>11.4f} {med_ent:>9.5f} {n_f:>4}")
    rows.append({'ID':cfg,'Config':label,'KDR_med':med_kdr,
                 'KDR_IQR':iqr_kdr,'KDR_SD':mean_sd,
                 'KGR_med':med_kgr,'Agreed_KGR':med_agr,
                 'Entropy':med_ent,'N_files':n_f,
                 'Degenerate':degen})

# ── Contribution analysis ─────────────────────────────────────────
print("\n"+"="*80)
print("ANALISIS KONTRIBUSI SETIAP KOMPONEN")
print("="*80)
base  =next((r['KDR_med'] for r in rows if r['ID']=='A1'),None)
a2    =next((r['KDR_med'] for r in rows if r['ID']=='A2'),None)
a5    =next((r['KDR_med'] for r in rows if r['ID']=='A5'),None)

if base and a2:
    print(f"\n  A1 (Raw RSSI)       : {base:.4f}  ← baseline")
    print(f"  A2 (+ Offset)       : {a2:.4f}  "
          f"→ REDUKSI {(base-a2)/base*100:.1f}% hanya dari offset")
if a2 and a5:
    print(f"  A5 (+ SavGol ★)     : {a5:.4f}  "
          f"→ REDUKSI tambahan {(a2-a5)/a2*100:.1f}% dari SavGol")
if base and a5:
    print(f"\n  TOTAL A1→A5         : REDUKSI {(base-a5)/base*100:.1f}% "
          f"({base:.4f} → {a5:.4f})")

print(f"\n  Smoothing comparison (di atas offset):")
for cid,cname in [('A3','Wavelet'),('A4','MovingAvg'),('A5','SavGol ★'),('A6','Kalman')]:
    r=next((x for x in rows if x['ID']==cid),None)
    if r:
        red=(a2-r['KDR_med'])/a2*100 if a2 else 0
        mark=" ← BEST KDR" if r['KDR_med']==min(
            x['KDR_med'] for x in rows if x['ID'] in ['A3','A4','A5','A6']) else ""
        stab=" ← MOST STABLE" if r['KDR_IQR']==min(
            x['KDR_IQR'] for x in rows if x['ID'] in ['A3','A4','A5','A6']) else ""
        print(f"    {cid} {cname:<12}: KDR={r['KDR_med']:.4f} IQR={r['KDR_IQR']:.4f}{mark}{stab}")

# ── Save ──────────────────────────────────────────────────────────
if rows:
    pd.DataFrame(rows).to_csv('ablation_SavGol_final.csv',index=False)
    print(f"\nSaved: ablation_SavGol_final.csv")

print("\n[DONE]")
