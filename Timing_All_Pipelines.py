"""
TIMING OF ALL SEVEN PIPELINES - one protocol for Table 11 (Section 4.7)

Runs each pipeline script of this repository repeatedly, unchanged, and reads
the stage times that the script itself measures (t_pre, t_train, t_infer, t_q,
t_recon, t_pa). For every pipeline:

  * the first run is a cold run (library / PyTorch initialisation) and is
    discarded,
  * then WARMUP runs are discarded,
  * then REPEATS runs are timed and the MEDIAN of each stage is reported.

T_proc  = preprocessing + training + inference + quantisation
          + reconciliation + SHA-256   (each counted once per run)
Total   = T_acq + T_proc,  T_acq = (N - 1) x 1.0 s
KGR     = bits / (T_acq + preprocessing + training + inference + quantisation)
          (the same formula as in each pipeline script / Table 10)

Usage
  Laptop : python Timing_All_Pipelines.py              (100 runs after 10 warm-up)
  RPi5   : python Timing_All_Pipelines.py 1 0          (single run, no warm-up)
  Other  : python Timing_All_Pipelines.py REPEATS WARMUP

The results are also written to timing_results_<REPEATS>x<WARMUP>.csv
"""
import sys, os, io, contextlib, runpy, warnings, time
import numpy as np

REPEATS = int(sys.argv[1]) if len(sys.argv) > 1 else 100
WARMUP  = int(sys.argv[2]) if len(sys.argv) > 2 else 10

PIPELINES = [
    ("Toward [13]",              "Toward_Revised_v2.py"),
    ("Scenario [14]",            "VehicleKey_Revised.py"),
    ("PLKG [15]",                "PLKG_Revised.py"),
    ("PLS-LoRa [16]",            "PLS_LoRa_Revised.py"),
    ("Proposed Wavelet+BiLSTM",  "Proposed_Revised_v3.py"),
    ("Proposed Wavelet",         "Proposed_A5_Revised.py"),
    ("Proposed Offset+SavGol",   "Proposed_SavGol_Final.py"),
]
STAGES = [("t_pre", "Preprocessing"), ("t_train", "Training"), ("t_infer", "Inference"),
          ("t_q", "Quantisation"), ("t_recon", "Reconciliation"), ("t_pa", "SHA-256")]

warnings.filterwarnings("ignore")

def run_once(path):
    """Execute one pipeline script silently and return its global variables."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            g = runpy.run_path(path, run_name="__main__")
        except SystemExit:
            raise RuntimeError(f"{path} stopped early:\n{buf.getvalue()[-500:]}")
    return g

def fmt(ms):
    if ms is None: return "-"
    return "< 0.01" if ms < 0.005 else f"{ms:,.2f}"

rows = []
t_start = time.time()
for name, path in PIPELINES:
    if not os.path.exists(path):
        print(f"!! {path} not found - skipped"); continue
    print(f"{name:26s} ...", end="", flush=True)
    run_once(path)                                   # cold run, discarded
    for _ in range(WARMUP): run_once(path)           # warm-up, discarded
    rec = {k: [] for k, _ in STAGES}; g = None
    for _ in range(REPEATS):
        g = run_once(path)
        for k, _ in STAGES:
            if k in g: rec[k].append(float(g[k]) * 1000.0)
    med = {k: (float(np.median(v)) if v else None) for k, v in rec.items()}
    t_proc = sum(v for v in med.values() if v is not None)
    t_acq = float(g["T_AKUISISI"])
    chain = sum(med[k] or 0.0 for k in ["t_pre", "t_train", "t_infer", "t_q"]) / 1000.0
    bits = g.get("n_bits", g.get("n_bits_total"))
    kdr = g.get("kdr_q", g.get("kdr_quant"))
    kgr = bits / (t_acq + chain) if bits else float("nan")
    agreed = kgr * (1 - kdr) if kdr is not None else float("nan")
    rows.append((name, med, t_proc, t_acq, bits, kdr, kgr, agreed))
    print(f" done  (T_proc {t_proc:,.2f} ms)")

print("\n" + "=" * 132)
print(f"TABLE 11 TIMING | repeats={REPEATS} warm-up={WARMUP} (median per stage, ms) | "
      f"elapsed {time.time() - t_start:.0f} s")
print("-" * 132)
hdr = f"{'Pipeline':26s}" + "".join(f"{lab:>15s}" for _, lab in STAGES) + f"{'T_proc':>12s}{'Total (s)':>11s}"
print(hdr)
for name, med, t_proc, t_acq, *_ in rows:
    print(f"{name:26s}" + "".join(f"{fmt(med[k]):>15s}" for k, _ in STAGES)
          + f"{t_proc:>12,.2f}{t_acq + t_proc / 1000:>11.2f}")
print("-" * 132)
print(f"{'Pipeline':26s}{'Bits':>6s}{'KDR':>10s}{'KGR (bps)':>12s}{'Agreed KGR':>12s}")
for name, med, t_proc, t_acq, bits, kdr, kgr, agreed in rows:
    print(f"{name:26s}{bits:>6d}{kdr:>10.5f}{kgr:>12.4f}{agreed:>12.4f}")
print("=" * 132)

out = f"timing_results_{REPEATS}x{WARMUP}.csv"
with open(out, "w") as f:
    f.write("pipeline," + ",".join(lab for _, lab in STAGES) + ",T_proc_ms,Total_s,bits,KDR,KGR,AgreedKGR\n")
    for name, med, t_proc, t_acq, bits, kdr, kgr, agreed in rows:
        f.write(f"{name}," + ",".join("" if med[k] is None else f"{med[k]:.4f}" for k, _ in STAGES)
                + f",{t_proc:.4f},{t_acq + t_proc / 1000:.4f},{bits},{kdr:.5f},{kgr:.4f},{agreed:.4f}\n")
print(f"Saved: {out}")
