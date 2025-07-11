#!/usr/bin/env python
# build_csv_numerics_dataset.py  –  numeric dataset + detailed logs + drop plots
# -----------------------------------------------------------------------------
import os, re, random, wfdb, numpy as np, pandas as pd, matplotlib.pyplot as plt

# ╭─ USER SETTINGS ─────────────────────────────────────────────────────────╮
ROOT_DIR   = os.getcwd()
OUT_DIR    = "csv_numerics_data"
LIMITS     = dict(HR=220, ABP=190, RESP=65)
FILTER_SPIKE_WINDOW = 10  # seconds to mask around spikes

DATA_WINDOW = 120    # seconds per input window
PREDICT_S   = 120    # seconds alarm lookahead
NAN_RUN_S   = 15     # sec of NaN to drop a window
ZERO_RUN_S  = 60     # sec of zeros to drop a window
N_PATIENTS  = None
random.seed(42)
# ╰──────────────────────────────────────────────────────────────────────────╯

# ───────── regex & helpers ─────────
ABP_TAG  = re.compile(r"(ABP|PAP|ART|PRESS)", re.I)
RESP_TAG = re.compile(r"RESP", re.I)
HR_TAG   = re.compile(r"(HR|ECG|PLETH|LEAD)", re.I)
NUM_RE   = re.compile(r"[0-9]+|<|>")

def alarm_channels(txt):
    ch=[]
    if ABP_TAG.search(txt):   ch.append("ABP")
    if RESP_TAG.search(txt):  ch.append("RESP")
    if HR_TAG.search(txt):    ch.append("HR")
    return ch or ["HR","ABP","RESP"]

def clean_alarm_text(txt):
    txt = NUM_RE.sub("", txt).strip().upper()
    txt = re.sub(r"\s{2,}", " ", txt)
    return "" if txt in {"HR","ABP","RESP"} or len(txt)<4 else txt

def is_patient_folder(p):
    return os.path.isdir(p) and os.path.basename(p).isdigit()

def numeric_header(files):
    num=[f for f in files if re.fullmatch(r"\d+n\.hea", f)]
    return num[0] if num else (files[0] if files else None)

def chan_idx(names,*cands):
    up=[n.upper() for n in names]
    for c in cands:
        if c.upper() in up:
            return up.index(c.upper())
    raise ValueError

def interp_minus_one(a: np.ndarray) -> np.ndarray:
    """
    Interpolate over -1.0 markers, preserving any np.nan as-is.
    """
    missing = (a == -1.0)
    if not missing.any():
        return a
    x = np.arange(len(a))
    valid = (~missing) & (~np.isnan(a))
    if valid.sum() < 2:
        return a  # not enough points to interpolate
    a[missing] = np.interp(x[missing], x[valid], a[valid])
    return a

def long_nan_run(arr, fs):
    bad = np.isnan(arr)
    run = max_run = 0
    thresh = NAN_RUN_S * fs
    for b in bad:
        if b:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return max_run >= thresh

def long_zero_run(arr, fs):
    bad = (arr == 0)
    run = max_run = 0
    thresh = ZERO_RUN_S * fs
    for b in bad:
        if b:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return max_run >= thresh

# ───────── main ─────────
patients = [
    os.path.join(ROOT_DIR, d)
    for d in os.listdir(ROOT_DIR)
    if is_patient_folder(os.path.join(ROOT_DIR, d))
]
if N_PATIENTS:
    patients = random.sample(patients, min(N_PATIENTS, len(patients)))
print(f"Processing {len(patients)} patients…")
os.makedirs(OUT_DIR, exist_ok=True)

global_alarms = set()
total_saved_windows = 0
total_final_alarms = 0

for pdir in patients:
    pid = os.path.basename(pdir)
    hea = numeric_header([f for f in os.listdir(pdir) if f.endswith(".hea")])
    if not hea:
        print(f"[{pid}] no numeric .hea – skip")
        continue

    rec_base = os.path.join(pdir, hea[:-4])
    try:
        rec = wfdb.rdrecord(rec_base)
    except Exception as e:
        print(f"[{pid}] rdrecord err: {e} – skip")
        continue

    try:
        i_hr, i_abp, i_resp = (
            chan_idx(rec.sig_name, "HR", "HEARTRATE"),
            chan_idx(rec.sig_name, "ABPMEAN", "ABPM"),
            chan_idx(rec.sig_name, "AWRR", "RESP"),
        )
    except ValueError:
        print(f"[{pid}] missing HR/ABP/RESP – skip")
        continue

    fs = rec.fs
    t  = np.arange(rec.p_signal.shape[0]) / fs
    hr, abp, resp = (
        rec.p_signal[:, i_hr].astype(float),
        rec.p_signal[:, i_abp].astype(float),
        rec.p_signal[:, i_resp].astype(float),
    )

    # 1) build spike mask windows
    mask = {ch: np.ones_like(hr, bool) for ch in ("HR","ABP","RESP")}
    win  = {ch: [] for ch in mask}
    for ch, arr in [("HR", hr), ("ABP", abp), ("RESP", resp)]:
        bad = np.where(arr > LIMITS[ch])[0]
        for idx in bad:
            tc = idx / fs
            t0, t1 = max(0, tc-FILTER_SPIKE_WINDOW), min(t[-1], tc+FILTER_SPIKE_WINDOW)
            win[ch].append((t0, t1))
            mask[ch] &= ~((t >= t0) & (t <= t1))

    # 2) apply mask & interpolate only -1 markers, preserve raw NaNs
    hr_marked   = np.where(mask["HR"],   hr,   -1.0)
    abp_marked  = np.where(mask["ABP"],  abp,  -1.0)
    resp_marked = np.where(mask["RESP"], resp, -1.0)

    hr_c   = interp_minus_one(hr_marked)
    abp_c  = interp_minus_one(abp_marked)
    resp_c = interp_minus_one(resp_marked)

    # 3) load alarms
    try:
        ann_al = wfdb.rdann(rec_base, "al")
        al_t   = np.array(ann_al.sample)/fs
        al_txt = ann_al.aux_note
    except:
        al_t, al_txt = np.array([]), []

    total_alarms = len(al_t)

    # 4) channel-specific muting
    kept_t, kept_txt = [], []
    muted = 0
    for ta, txt in zip(al_t, al_txt):
        m = False
        for ch in alarm_channels(txt):
            if any(t0 <= ta <= t1 for (t0, t1) in win[ch]):
                val = {"HR":hr, "ABP":abp, "RESP":resp}[ch][int(ta*fs)]
                if val > LIMITS[ch]:
                    m = True
                    break
        if m:
            muted += 1
        else:
            kept_t.append(ta)
            kept_txt.append(txt)

    # 5) split into DATA_WINDOW chunks
    win_len = int(DATA_WINDOW * fs)
    n_win   = len(hr) // win_len

    drop_count = {ch:0 for ch in ("HR","ABP","RESP")}
    dropped = saved = 0
    plotted_drop = False

    out_sub = os.path.join(OUT_DIR, pid)
    os.makedirs(out_sub, exist_ok=True)

    for w in range(n_win):
        b, e = w*win_len, (w+1)*win_len
        if e > len(hr): break

        # drop based on cleaned data runs of NaN or zeros
        drop_nan  = long_nan_run(hr_c[b:e], fs)   or long_nan_run(abp_c[b:e], fs)   or long_nan_run(resp_c[b:e], fs)
        drop_zero = long_zero_run(hr_c[b:e], fs) or long_zero_run(abp_c[b:e], fs) or long_zero_run(resp_c[b:e], fs)

        if drop_nan or drop_zero:
            for ch, arr in [("HR", hr_c), ("ABP", abp_c), ("RESP", resp_c)]:
                if long_nan_run(arr[b:e], fs) or long_zero_run(arr[b:e], fs):
                    drop_count[ch] += 1
            dropped += 1

            # plot first dropped window
            if not plotted_drop:
                plotted_drop = True
                fig, ax = plt.subplots(figsize=(10,3))
                ax.plot(t[b:e]-t[b], hr[b:e],   lw=1, label="HR")
                ax.plot(t[b:e]-t[b], abp[b:e], lw=1, label="ABP")
                ax.plot(t[b:e]-t[b], resp[b:e],lw=1, label="RESP")
                ax.set_title(f"Patient {pid} – First Dropped Window")
                ax.set_xlabel("Time (s)")
                ax.legend(fontsize=6)
                ax.grid(True)
                plt.tight_layout()
                plt.show()
            continue

        # save CSV
        df = pd.DataFrame({
            "Time":      t[b:e] - t[b],
            "HR":        hr_c[b:e],
            "ABPmean":   abp_c[b:e],
            "RESP":      resp_c[b:e]
        })
        fn = f"{pid}_{w:02d}.csv"
        df.to_csv(os.path.join(out_sub, fn), index=False)

        # alarms in predict window, skip "disconnect"
        ps, pe = e/fs, e/fs + PREDICT_S
        sel = [
            txt for ta, txt in zip(kept_t, kept_txt)
            if ps<=ta<=pe and "disconnect" not in txt.lower()
        ]

        cleaned = [clean_alarm_text(a) for a in sel]
        cleaned = [a for a in cleaned if a]
        uniq = sorted(dict.fromkeys(cleaned))

        for a in uniq:
            global_alarms.add(a)

        with open(os.path.join(out_sub, f"{pid}_{w:02d}_alarm.txt"), "w") as f:
            f.write("\n".join(uniq))

        saved += 1
        total_saved_windows += 1
        total_final_alarms  += len(uniq)

    # logs
    print(f"[{pid}] total_windows={n_win}, saved={saved}, dropped={dropped}")
    print(f"         drop_by_nan_zero → HR={drop_count['HR']}, "
          f"ABP={drop_count['ABP']}, RESP={drop_count['RESP']}")
    print(f"         total_alarms={total_alarms}, kept={len(kept_t)}, muted={muted}")

# write global alarm list
full = os.path.join(OUT_DIR, "full_alarm_list.txt")
with open(full, "w") as f:
    f.write("\n".join(sorted(global_alarms)))
print(f"\nGlobal alarms ({len(global_alarms)}) → {full}")

# final summary
print(f"\n[SUMMARY] Total saved windows: {total_saved_windows}")
print(f"[SUMMARY] Total alarms recorded (after cleaning): {total_final_alarms}")
