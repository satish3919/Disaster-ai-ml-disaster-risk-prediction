
"""
train_flood_model_v3.py
=================================
PHASE A-5: Threshold tuning + regression-first approach.

BUILDS ON v2 (same leakage-safe labels, same ENSO+IOD features). Adds
two things your v2 report suggested trying next:

1. THRESHOLD SWEEP: v2 used the default 0.5 probability cutoff for
   "is this a flood year". For an early-warning use case, missing a
   real flood year (false negative) is usually worse than a false
   alarm, so this script sweeps thresholds and reports F1 AND recall
   at each one - so you can deliberately pick a lower threshold if you
   want higher recall, instead of accepting whatever 0.5 gives you.
   This uses out-of-fold predictions (cross_val_predict with the same
   GroupKFold leave-region-out splits), so the threshold choice itself
   is not tuned on data the model was trained on - no new leakage.

2. REGRESSION-FIRST: instead of predicting a binary flood/no-flood
   label directly, this predicts the actual JAS rainfall Z-SCORE
   (how many std-devs above/below that region's own history) as a
   continuous number, then thresholds the PREDICTED z-score afterward.
   Regression often extracts more signal than direct classification
   when the underlying phenomenon (rainfall amount) is continuous -
   you are not throwing away the magnitude information by collapsing
   to a label before training.

Both approaches are evaluated with the same leave-region-out GroupKFold
protocol as before, so results are directly comparable to your v2
report.

USAGE:
    python train_flood_model_v3.py

REQUIRES: flood_model_v2's feature engineering logic (duplicated here
so this script can run standalone) and the same rainfall CSV.
"""

import urllib.request
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    mean_squared_error, r2_score, confusion_matrix
)

# ============================================================
RAINFALL_CSV_PATH = "rainfall in india 1901-2015.csv"
ONI_URL = "https://psl.noaa.gov/data/correlation/oni.data"
ONI_LOCAL_FALLBACK = "oni.data"
IOD_URL = "https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data"
IOD_LOCAL_FALLBACK = "dmi.data"
FLOOD_THRESHOLD_STD = 1.0        # used only to LABEL flood years for evaluation
MIN_HISTORY_YEARS = 10
MIN_YEARS_PER_REGION = 15
RANDOM_SEED = 42
REPORT_OUTPUT_PATH = "flood_model_v3_report.txt"
# ============================================================


def _fetch_noaa_index(url, local_fallback, label):
    raw_text = None
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            raw_text = resp.read().decode("utf-8", errors="ignore")
        print(f"Downloaded live {label} data from NOAA.")
    except Exception as e:
        print(f"Could not download {label} data live ({e}).")
        with open(local_fallback, "r") as f:
            raw_text = f.read()
        print(f"Loaded {label} data from local file: {local_fallback}")

    rows = []
    for line in raw_text.strip().splitlines()[1:]:
        parts = line.split()
        if not parts:
            continue
        try:
            year = int(parts[0])
        except ValueError:
            break
        if year < 1800 or year > 2100 or len(parts) < 13:
            break
        try:
            values = [float(v) for v in parts[1:13]]
        except ValueError:
            break
        rows.append([year] + values)

    cols = ["YEAR", "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
            "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    out = pd.DataFrame(rows, columns=cols)
    out = out[(out[["MAR", "APR", "MAY"]].abs() < 90).all(axis=1)]
    print(f"{label} data loaded: {out['YEAR'].min():.0f}-{out['YEAR'].max():.0f} ({len(out)} years)")
    return out


def fetch_oni_data():
    oni = _fetch_noaa_index(ONI_URL, ONI_LOCAL_FALLBACK, "ONI")
    oni["ONI_PRE_MONSOON"] = oni[["MAR", "APR", "MAY"]].mean(axis=1)
    return oni[["YEAR", "ONI_PRE_MONSOON"]]


def fetch_iod_data():
    iod = _fetch_noaa_index(IOD_URL, IOD_LOCAL_FALLBACK, "IOD/DMI")
    iod["IOD_PRE_MONSOON"] = iod[["MAR", "APR", "MAY"]].mean(axis=1)
    return iod[["YEAR", "IOD_PRE_MONSOON"]]


def load_rainfall_data(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().upper() for c in df.columns]
    df["SUBDIVISION"] = df["SUBDIVISION"].astype(str).str.strip().str.upper()
    return df.sort_values(["SUBDIVISION", "YEAR"]).reset_index(drop=True)


def engineer_features(df, oni, iod):
    df = df.copy()
    counts = df.groupby("SUBDIVISION")["YEAR"].transform("count")
    df = df[counts >= MIN_YEARS_PER_REGION].copy()

    df["JAS"] = df["JUL"] + df["AUG"] + df["SEP"]
    df = df.sort_values(["SUBDIVISION", "YEAR"])
    grp = df.groupby("SUBDIVISION")["JAS"]
    df["JAS_HIST_MEAN"] = grp.apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=0, drop=True)
    df["JAS_HIST_STD"] = grp.apply(lambda s: s.shift(1).expanding().std()).reset_index(level=0, drop=True)
    df["JAS_HIST_COUNT"] = grp.cumcount()
    df = df[df["JAS_HIST_COUNT"] >= MIN_HISTORY_YEARS].copy()

    # Regression target: how many std-devs is this year's rainfall from
    # that region's own history (continuous, not binary)
    df["JAS_ZSCORE"] = (df["JAS"] - df["JAS_HIST_MEAN"]) / df["JAS_HIST_STD"]
    # Binary label (same definition as v2) - used only for evaluation/comparison
    df["FLOOD"] = (df["JAS_ZSCORE"] > FLOOD_THRESHOLD_STD).astype(int)

    df["PRE_MONSOON"] = df["MAR"] + df["APR"] + df["MAY"]
    df["JJAS_FULL"] = df["JUN"] + df["JUL"] + df["AUG"] + df["SEP"]
    df["PREV_JJAS"] = df.groupby("SUBDIVISION")["JJAS_FULL"].shift(1)
    df["RATE_OF_CHANGE"] = df["JUN"] - (df["PRE_MONSOON"] / 3.0)

    df = df.merge(oni, on="YEAR", how="left")
    df = df.merge(iod, on="YEAR", how="left")

    feature_cols = ["JUN", "PRE_MONSOON", "PREV_JJAS", "RATE_OF_CHANGE",
                     "ONI_PRE_MONSOON", "IOD_PRE_MONSOON",
                     "JAS_HIST_MEAN", "JAS_HIST_STD"]
    df = df.dropna(subset=feature_cols + ["JAS_ZSCORE"]).reset_index(drop=True)

    print(f"Regions included: {df['SUBDIVISION'].nunique()} | Rows: {len(df)}")
    print(f"Flood-year rate (z > {FLOOD_THRESHOLD_STD}): {100 * df['FLOOD'].mean():.1f}%")
    return df, feature_cols


def threshold_sweep_section(X, y, groups):
    """Approach 1: direct classifier, out-of-fold probabilities, sweep cutoff."""
    lines = ["=" * 65, "APPROACH 1: THRESHOLD SWEEP ON DIRECT CLASSIFIER", "=" * 65]

    model = HistGradientBoostingClassifier(
        max_depth=4, learning_rate=0.05, max_iter=300,
        random_state=RANDOM_SEED, class_weight="balanced"
    )
    n_splits = min(10, groups.nunique())
    gkf = GroupKFold(n_splits=n_splits)

    # Out-of-fold probabilities: each row's prediction comes from a model
    # that never saw that row's region during training - no leakage.
    oof_proba = cross_val_predict(
        model, X, y, cv=gkf, groups=groups, method="predict_proba"
    )[:, 1]

    lines.append(f"\nOut-of-fold ROC-AUC: {roc_auc_score(y, oof_proba):.3f}")
    lines.append("\nThreshold |  F1   | Precision | Recall | Flagged-as-flood %")
    lines.append("-" * 62)

    best_f1, best_thresh = 0, 0.5
    for thresh in np.arange(0.15, 0.65, 0.05):
        preds = (oof_proba >= thresh).astype(int)
        f1 = f1_score(y, preds, zero_division=0)
        prec = precision_score(y, preds, zero_division=0)
        rec = recall_score(y, preds, zero_division=0)
        flagged_pct = 100 * preds.mean()
        lines.append(f"  {thresh:.2f}   | {f1:.3f} |   {prec:.3f}   | {rec:.3f} |  {flagged_pct:.1f}%")
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh

    lines.append(f"\nBest F1 threshold: {best_thresh:.2f} (F1={best_f1:.3f})")
    lines.append(
        "\nHow to use this: if you want an early-warning system biased toward\n"
        "catching more real flood years (fewer misses, more false alarms),\n"
        "pick a threshold to the LEFT of the best-F1 one (lower cutoff =\n"
        "higher recall). If false alarms are costly, pick one to the right."
    )
    return "\n".join(lines)


def regression_first_section(df, feature_cols, groups):
    """Approach 2: predict the continuous z-score, then threshold afterward."""
    lines = ["\n" + "=" * 65, "APPROACH 2: REGRESSION-FIRST (predict rainfall z-score)", "=" * 65]

    X = df[feature_cols]
    y_continuous = df["JAS_ZSCORE"]
    y_binary = df["FLOOD"]

    reg_model = HistGradientBoostingRegressor(
        max_depth=4, learning_rate=0.05, max_iter=300, random_state=RANDOM_SEED
    )
    n_splits = min(10, groups.nunique())
    gkf = GroupKFold(n_splits=n_splits)

    oof_pred_zscore = cross_val_predict(reg_model, X, y_continuous, cv=gkf, groups=groups)

    rmse = mean_squared_error(y_continuous, oof_pred_zscore) ** 0.5
    r2 = r2_score(y_continuous, oof_pred_zscore)
    lines.append(f"\nOut-of-fold RMSE (z-score units): {rmse:.3f}")
    lines.append(f"Out-of-fold R^2:                  {r2:.3f}")
    lines.append(
        "(R^2 near 0 means the regression barely beats predicting the\n"
        " historical average every time; meaningfully above 0 means it's\n"
        " capturing real year-to-year variation.)"
    )

    # Convert the predicted continuous z-score back into a flood/no-flood
    # call, then compare to the same binary label used in Approach 1.
    derived_preds = (oof_pred_zscore > FLOOD_THRESHOLD_STD).astype(int)
    f1 = f1_score(y_binary, derived_preds, zero_division=0)
    prec = precision_score(y_binary, derived_preds, zero_division=0)
    rec = recall_score(y_binary, derived_preds, zero_division=0)
    lines.append(f"\nDerived flood classification (threshold z > {FLOOD_THRESHOLD_STD}):")
    lines.append(f"  F1: {f1:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f}")
    lines.append(f"  Confusion matrix:\n{confusion_matrix(y_binary, derived_preds)}")

    lines.append(
        "\nCompare this F1 to Approach 1's best-F1 row above. Whichever is\n"
        "higher is the better modeling strategy for this data - keep that\n"
        "one and discard the other rather than running both in production."
    )
    return "\n".join(lines)


def main():
    print("Fetching climate indices...")
    oni = fetch_oni_data()
    iod = fetch_iod_data()

    print(f"\nLoading rainfall data from {RAINFALL_CSV_PATH}...")
    raw = load_rainfall_data(RAINFALL_CSV_PATH)
    df, feature_cols = engineer_features(raw, oni, iod)

    X = df[feature_cols]
    y = df["FLOOD"]
    groups = df["SUBDIVISION"]

    section1 = threshold_sweep_section(X, y, groups)
    section2 = regression_first_section(df, feature_cols, groups)

    report = "\n".join([section1, section2])
    print("\n" + report)
    with open(REPORT_OUTPUT_PATH, "w") as f:
        f.write(report)
    print(f"\nReport saved to: {REPORT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()