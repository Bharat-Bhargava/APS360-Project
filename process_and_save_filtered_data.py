#!/usr/bin/env python
# process_and_save_filtered_data.py – Filters signals and saves to CSV
# -----------------------------------------------------------------------------
import os, random, re, wfdb, numpy as np, matplotlib.pyplot as plt
import pandas as pd # REQUIRED for CSV export
plt.rcParams["figure.dpi"] = 110

# ───────────────────────── USER SETTINGS ──────────────────────────
# IMPORTANT: Set this to the root directory where your MIMIC data is located
ROOT_DIR   = r"/home/hydrogamer/Downloads/DESTINATION"
N_PATIENTS = 10               # how many patient folders to sample per run
WINDOW_S   = 30                # seconds to blank on either side of each fault
OUTPUT_DIR = "processed_csv_output" # Folder where cleaned CSVs will be saved

# Plausible physiological ranges used to decide if a reading is “bad”
LIMITS = {
    "HR":   (-1, 250),   # bpm
    "ABP":  (-1, 250),   # mmHg (mean)
    "RESP": (-1,  100)     # breaths/min
}
# ──────────────────────────────────────────────────────────────────

# Intervention & malfunction patterns
INTERVENTION_PATTERNS = [
    r"start", r"stop", r"bolus", r"infus", r"ventilat", r"intubat",
    r"extubat", r"sedat", r"nibp", r"abg", r"drug", r"dose", r"pressor"
]
MALFUNCTION_PATTERNS = [
    r"reduce size", r"non[- ]?pulsat", r"leads? off", r"lead off",
    r"erratic", "fault", "artifact", "alarm", "calibra", "noise",
    "disconnect", "obstruct", "out of range", "pressure bag",
    "reposition", "sensor"
]

def is_patient_folder(path: str) -> bool:
    """Return True if the folder name is all digits (e.g., '480')."""
    return os.path.isdir(path) and os.path.basename(path).isdigit()

def find_numeric_header(files):
    """
    Pick the numeric .hea file:
    • preference:   ####n.hea  (trailing 'n' before extension)
    • fallback:     first .hea file in the folder
    """
    numeric = [f for f in files if re.fullmatch(r"\d+n\.hea", f)]
    return numeric[0] if numeric else (files[0] if files else None)

def idx_for(sig_names, *candidates):
    """
    Return the first index whose (case-insensitive) name matches
    one of the candidate names; raise ValueError if none match.
    """
    for cand in candidates:
        for i, name in enumerate(sig_names):
            if name.lower() == cand.lower():
                return i
    raise ValueError(f"None of {candidates} found in {sig_names}")

# ──────────────────────────────────────────────────────────────────
# MAIN PROCESSING LOOP
# ──────────────────────────────────────────────────────────────────
# Get all patient folders
all_patient_folders = [d for d in os.listdir(ROOT_DIR)
                       if is_patient_folder(os.path.join(ROOT_DIR, d))]

if not all_patient_folders:
    print(f"No patient folders found in {ROOT_DIR}. Exiting.")
    exit()

# Randomly sample N_PATIENTS
random.seed(42) # for reproducibility
patients = [(os.path.join(ROOT_DIR, p), p) for p in
            random.sample(all_patient_folders, min(N_PATIENTS, len(all_patient_folders)))]

print(f"Processing {len(patients)} patient records...")

for path, record_name in patients:
    try:
        # Find the numeric header file
        record_files = os.listdir(path)
        header_file  = find_numeric_header(record_files)
        if not header_file:
            print(f"  Skipping record {record_name}: No numeric header found.")
            continue

        # Load the record
        record_path = os.path.join(path, header_file.replace(".hea", ""))
        record = wfdb.rdrecord(record_path)
        signals = np.array(record.p_signal)
        t       = np.array(range(len(signals))) / record.fs

        # Get signal indices (handling missing signals)
        try:
            idx_hr = idx_for(record.sig_name, "HR")
            hr_sig = signals[:, idx_hr]
        except ValueError:
            print(f"  Warning: 'HR' signal not found for record {record_name}. Using NaN for HR.")
            hr_sig = np.full_like(t, np.nan) # Fill with NaN if missing

        try:
            idx_abp = idx_for(record.sig_name, "ABPmean", "ABP")
            abp_sig = signals[:, idx_abp]
        except ValueError:
            print(f"  Warning: 'ABPmean' or 'ABP' signal not found for record {record_name}. Using NaN for ABP.")
            abp_sig = np.full_like(t, np.nan) # Fill with NaN if missing

        try:
            idx_resp = idx_for(record.sig_name, "RESP", "AWRR")
            resp_sig = signals[:, idx_resp]
        except ValueError:
            print(f"  Warning: 'RESP' or 'AWRR' signal not found for record {record_name}. Using NaN for RESP.")
            resp_sig = np.full_like(t, np.nan) # Fill with NaN if missing


        # Load annotations (handling missing files)
        # For .ml (malfunction) annotations
        ml_t = np.array([]) # Initialize as empty
        try:
            ml_ann = wfdb.rdann(record_path, "ml")
            ml_t = np.array([s for s, a in zip(ml_ann.sample, ml_ann.aux_note)
                             if any(re.search(p, a.lower()) for p in MALFUNCTION_PATTERNS)]) / record.fs
        except Exception:
            pass # Keep ml_t as empty if file not found or error


        # For .in (intervention) annotations
        inter_t = np.array([]) # Initialize as empty
        try:
            in_ann = wfdb.rdann(record_path, "in")
            inter_t = np.array([s for s, a in zip(in_ann.sample, in_ann.aux_note)
                                if any(re.search(p, a.lower()) for p in INTERVENTION_PATTERNS)]) / record.fs
        except Exception:
            pass # Keep inter_t as empty if file not found or error

        # For .al (alarm) annotations
        al_t = np.array([]) # Initialize as empty
        try:
            al_ann = wfdb.rdann(record_path, "al")
            al_t = np.array(al_ann.sample) / record.fs
        except Exception:
            pass # Keep al_t as empty if file not found or error


        # ───────────────────────── CLEANING LOGIC ───────────────────────
        hr, abp, rsp = hr_sig.copy(), abp_sig.copy(), resp_sig.copy()
        mask_hr, mask_abp, mask_resp = np.full_like(hr, True, dtype=bool), np.full_like(abp, True, dtype=bool), np.full_like(rsp, True, dtype=bool)

        # 1. Mask based on physiological limits
        mask_hr   = (hr >= LIMITS["HR"][0])   & (hr <= LIMITS["HR"][1])
        mask_abp  = (abp >= LIMITS["ABP"][0]) & (abp <= LIMITS["ABP"][1])
        mask_resp = (rsp >= LIMITS["RESP"][0])& (rsp <= LIMITS["RESP"][1])

        # 2. Mask around event times
        event_times = np.concatenate([inter_t, ml_t, al_t])
        event_sample_idx = (event_times * record.fs).astype(int)

        for event_idx in event_sample_idx:
            start_idx = max(0, int(event_idx - WINDOW_S * record.fs))
            end_idx   = min(len(signals), int(event_idx + WINDOW_S * record.fs))
            mask_hr[start_idx:end_idx] = False
            mask_abp[start_idx:end_idx] = False
            mask_resp[start_idx:end_idx] = False

        # Apply masks
        hr_cln   = np.where(mask_hr, hr, np.nan)
        abp_cln  = np.where(mask_abp, abp, np.nan)
        rsp_cln = np.where(mask_resp, rsp, np.nan)

        # ───────────────────────── CSV SAVING LOGIC ───────────────────────
        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Create a DataFrame from the cleaned data
        df = pd.DataFrame({
            'Time': t,
            'HR_Clean': hr_cln,
            'ABPmean_Clean': abp_cln,
            'RESP_Clean': rsp_cln
        })

        # Define the output CSV file path
        csv_filename = os.path.join(OUTPUT_DIR, f"{record_name}_cleaned_signals.csv")

        # Save the DataFrame to CSV
        df.to_csv(csv_filename, index=False) # index=False prevents writing DataFrame index as a column
        print(f"Saved cleaned data for {record_name} to {csv_filename}")
        # ──────────────────────────────────────────────────────────────────

        # ─────────────── plot (Optional: You can remove this section if you only need CSVs) ───────────────
        if not (np.all(np.isnan(hr)) and np.all(np.isnan(abp)) and np.all(np.isnan(rsp))):
            y_min = np.nanmin([hr, abp, rsp]) - 5
            y_max = np.nanmax([hr, abp, rsp]) + 5

            fig, (ax_raw, ax_cln) = plt.subplots(1, 2, figsize=(18,5), sharey=True)

            def panel(ax, h,a,r, title):
                ax.plot(t, h, lw=.7, label="HR")
                ax.plot(t, a, lw=.7, label="ABPmean")
                ax.plot(t, r, lw=.7, label="RESP")

                ax.scatter(al_t, [y_max]*len(al_t), c="red", s=20, marker="o", label="_alarms")
                ax.scatter(inter_t, [y_max-2]*len(inter_t), c="green", s=20, marker="o", label="_interv")
                ax.scatter(ml_t, [y_min]*len(ml_t), c="black", s=20, marker="o", label="_malfunc")

                ax.set_title(title)
                ax.set_xlim(t[0], t[-1])
                ax.set_ylim(y_min, y_max)
                ax.set_xlabel("Time (s)")
                ax.grid(True, ls=':')
                ax.legend(loc="upper right", ncol=3)

            panel(ax_raw, hr, abp, rsp, f"Patient {record_name} (RAW)")
            panel(ax_cln, hr_cln, abp_cln, rsp_cln, f"Patient {record_name} (CLEANED)")

            plt.tight_layout()
            plt.show()
        else:
            print(f"  No valid numeric data found for plotting record {record_name}.")

        # Print statistics
        hr_initial = np.count_nonzero(~np.isnan(hr))
        abp_initial = np.count_nonzero(~np.isnan(abp))
        resp_initial = np.count_nonzero(~np.isnan(rsp))

        hr_valid_range = np.count_nonzero((hr >= LIMITS["HR"][0]) & (hr <= LIMITS["HR"][1]))
        abp_valid_range = np.count_nonzero((abp >= LIMITS["ABP"][0]) & (abp <= LIMITS["ABP"][1]))
        resp_valid_range = np.count_nonzero((rsp >= LIMITS["RESP"][0]) & (rsp <= LIMITS["RESP"][1]))

        print(f"Record {record_name}: "
              f"initial(HR/ABP/RESP)={hr_initial:,}/{abp_initial:,}/{resp_initial:,} "
              f"valid(HR/ABP/RESP)={hr_valid_range:,}/{abp_valid_range:,}/{resp_valid_range:,} "
              f"masked(HR/ABP/RESP)={np.count_nonzero(~np.isnan(hr_cln)):,}/"
              f"{np.count_nonzero(~np.isnan(abp_cln)):,}/"
              f"{np.count_nonzero(~np.isnan(rsp_cln)):,}")

    except Exception as e:
        print(f"Error processing record {record_name}: {e}")
