
"""
train_flood_model_v2.py
=================================
PHASE A-4: Leakage-safe labels + IOD climate signal + gradient boosting.

WHAT CHANGED FROM THE PREVIOUS VERSION:
1. LABEL LEAKAGE FIX: the flood threshold (region mean + std of JAS
   rainfall) is now computed using an EXPANDING WINDOW of only prior
   years for each region-year, not the full 1901-2015 span. This makes
   the label a genuine "would this have been classifiable at the time"
   evaluation instead of a retrospective one.
2. Dropped region-identity one-hot dummies (they contribute nothing
   under leave-region-out CV - the held-out region's column is always
   zero in both train and test for that fold, so they're just noise).
   Replaced with numeric regional descriptors (that region's own
   historical mean/std, computed the same leakage-safe expanding way)
   so information can transfer ACROSS regions instead of encoding
   identity.
3. Added the real IOD (Indian Ocean Dipole / DMI) index from NOAA -
   a second, independent physical driver of Indian monsoon strength,
   alongside ENSO (ONI). Both are pre-monsoon (Mar-Apr-May) averages,
   known before June starts - genuine leakage-free forecasting inputs.
4. Added a rate-of-change feature (is rainfall accelerating or lagging
   heading into the monsoon).
5. Compares THREE models: Random Forest, Logistic Regression, and
   HistGradientBoostingClassifier (sklearn's built-in gradient boosting -
   no extra install needed, usually strong on imbalanced tabular data).
6. Reports PR-AUC (average precision) alongside ROC-AUC - PR-AUC is the
   more honest metric when positives are rare, since ROC-AUC can look
   good even when precision on the minority class is poor.

IMPORTANT - REALISTIC EXPECTATIONS:
Predicting an anomalously wet monsoon season from pre-season signals
alone is a genuinely hard, noisy geophysical problem. Published climate
science suggests ENSO/IOD explain a meaningful but partial fraction of
monsoon variability. A realistic, credible target for this task is
roughly F1 ~0.5-0.65 and ROC-AUC ~0.65-0.78 on truly unseen regions.
If you ever see numbers approaching 0.95+ here, that is a strong signal
of a leakage bug, not a breakthrough - re-check the label and feature
construction before trusting it.

USAGE:
    python train_flood_model_v2.py
"""

import urllib.request
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GroupKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score
)
import joblib

# ============================================================
RAINFALL_CSV_PATH = "rainfall in india 1901-2015.csv"  # <-- confirm matches your file name
ONI_URL = "https://psl.noaa.gov/data/correlation/oni.data"
ONI_LOCAL_FALLBACK = "oni.data"
IOD_URL = "https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data"
IOD_LOCAL_FALLBACK = "dmi.data"
FLOOD_THRESHOLD_STD = 1.0
MIN_HISTORY_YEARS = 10       # min prior years needed before a region-year gets a label
MIN_YEARS_PER_REGION = 15
MODEL_OUTPUT_PATH = "flood_model_v2.pkl"
REPORT_OUTPUT_PATH = "flood_model_v2_report.txt"
RANDOM_SEED = 42
# ============================================================


def _fetch_noaa_index(url: str, local_fallback: str, label: str) -> pd.DataFrame:
    """Generic fetcher for NOAA's fixed-width year-by-month index files (ONI, DMI/IOD)."""
    raw_text = None
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            raw_text = resp.read().decode("utf-8", errors="ignore")
        print(f"Downloaded live {label} data from NOAA.")
    except Exception as e:
        print(f"Could not download {label} data live ({e}).")
        try:
            with open(local_fallback, "r") as f:
                raw_text = f.read()
            print(f"Loaded {label} data from local file: {local_fallback}")
        except FileNotFoundError:
            raise RuntimeError(
                f"Could not download {label} data and no local fallback found.\n"
                f"Manually download it from {url} and save it as "
                f"'{local_fallback}' in this folder, then rerun."
            )

    rows = []
    lines = raw_text.strip().splitlines()
    for line in lines[1:]:
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
    out = out[(out[["MAR", "APR", "MAY"]].abs() < 90).all(axis=1)]  # drop sentinel/missing rows (e.g. -999, 99.9)
    print(f"{label} data loaded: {out['YEAR'].min():.0f}-{out['YEAR'].max():.0f} ({len(out)} years)")
    return out


def fetch_oni_data() -> pd.DataFrame:
    oni = _fetch_noaa_index(ONI_URL, ONI_LOCAL_FALLBACK, "ONI")
    oni["ONI_PRE_MONSOON"] = oni[["MAR", "APR", "MAY"]].mean(axis=1)
    return oni[["YEAR", "ONI_PRE_MONSOON"]]


def fetch_iod_data() -> pd.DataFrame:
    iod = _fetch_noaa_index(IOD_URL, IOD_LOCAL_FALLBACK, "IOD/DMI")
    iod["IOD_PRE_MONSOON"] = iod[["MAR", "APR", "MAY"]].mean(axis=1)
    return iod[["YEAR", "IOD_PRE_MONSOON"]]


def load_rainfall_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().upper() for c in df.columns]
    required_months = ["MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP"]
    missing = [c for c in ["SUBDIVISION", "YEAR"] + required_months if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}. Found: {list(df.columns)}")
    df["SUBDIVISION"] = df["SUBDIVISION"].astype(str).str.strip().str.upper()
    df = df.sort_values(["SUBDIVISION", "YEAR"]).reset_index(drop=True)
    return df


def engineer_features_and_labels(df: pd.DataFrame, oni: pd.DataFrame, iod: pd.DataFrame,
                                  threshold_std: float) -> pd.DataFrame:
    df = df.copy()
    counts = df.groupby("SUBDIVISION")["YEAR"].transform("count")
    df = df[counts >= MIN_YEARS_PER_REGION].copy()

    df["JAS"] = df["JUL"] + df["AUG"] + df["SEP"]

    # --- LEAKAGE-SAFE LABEL: expanding window of PRIOR years only ---
    df = df.sort_values(["SUBDIVISION", "YEAR"])
    grp = df.groupby("SUBDIVISION")["JAS"]
    # shift(1) excludes the current year, then expanding mean/std uses only years before it
    df["JAS_HIST_MEAN"] = grp.apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=0, drop=True)
    df["JAS_HIST_STD"] = grp.apply(lambda s: s.shift(1).expanding().std()).reset_index(level=0, drop=True)
    df["JAS_HIST_COUNT"] = grp.cumcount()  # number of prior years available

    df = df[df["JAS_HIST_COUNT"] >= MIN_HISTORY_YEARS].copy()
    df["FLOOD"] = (df["JAS"] > (df["JAS_HIST_MEAN"] + threshold_std * df["JAS_HIST_STD"])).astype(int)

    df["PRE_MONSOON"] = df["MAR"] + df["APR"] + df["MAY"]
    df["JJAS_FULL"] = df["JUN"] + df["JUL"] + df["AUG"] + df["SEP"]
    df["PREV_JJAS"] = df.groupby("SUBDIVISION")["JJAS_FULL"].shift(1)

    # Rate-of-change heading into the monsoon: is June rain running ahead of
    # or behind the pre-monsoon pace?
    df["RATE_OF_CHANGE"] = df["JUN"] - (df["PRE_MONSOON"] / 3.0)

    # Merge real climate indices (both are pre-monsoon, known before June)
    df = df.merge(oni, on="YEAR", how="left")
    df = df.merge(iod, on="YEAR", how="left")

    
    df = df.dropna(
        subset=["JUN", "PRE_MONSOON", "PREV_JJAS", "RATE_OF_CHANGE",
                "ONI_PRE_MONSOON", "IOD_PRE_MONSOON",
                "JAS_HIST_MEAN", "JAS_HIST_STD"]
    ).reset_index(drop=True)

    print(f"Regions included: {df['SUBDIVISION'].nunique()}")
    print(f"Total rows: {len(df)}")
    print(f"Flood-year rate: {100 * df['FLOOD'].mean():.1f}%")
    return df


def build_feature_matrix(df: pd.DataFrame):
    # Numeric-only features - NO region-identity dummies (they're dead
    # weight under leave-region-out CV). JAS_HIST_MEAN/STD act as a
    # leakage-safe numeric stand-in for "what kind of region is this",
    # since it transfers across regions instead of encoding identity.
    numeric_features = [
        "JUN", "PRE_MONSOON", "PREV_JJAS", "RATE_OF_CHANGE",
        "ONI_PRE_MONSOON", "IOD_PRE_MONSOON",
        "JAS_HIST_MEAN", "JAS_HIST_STD",
    ]
    X = df[numeric_features].copy()
    y = df["FLOOD"]
    groups = df["SUBDIVISION"]
    return X, y, groups, numeric_features


def evaluate_model(model_name, model, X, y, groups):
    lines = [f"\n=== {model_name} ==="]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    lines.append("--- WITHIN-REGION (random split) ---")
    lines.append(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
    lines.append(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.3f}")
    lines.append(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.3f}")
    lines.append(f"F1 score:  {f1_score(y_test, y_pred, zero_division=0):.3f}")
    if len(set(y_test)) > 1:
        lines.append(f"ROC-AUC:   {roc_auc_score(y_test, y_proba):.3f}")
        lines.append(f"PR-AUC:    {average_precision_score(y_test, y_proba):.3f}")
    lines.append(f"Confusion matrix:\n{confusion_matrix(y_test, y_pred)}")

    n_splits = min(10, groups.nunique())
    gkf = GroupKFold(n_splits=n_splits)
    f1_scores = cross_val_score(model, X, y, cv=gkf, groups=groups, scoring="f1")
    auc_scores = cross_val_score(model, X, y, cv=gkf, groups=groups, scoring="roc_auc")
    lines.append(f"\n--- LEAVE-REGION-OUT ({n_splits} folds) - the metric that matters most ---")
    lines.append(f"F1 per fold:      {np.round(f1_scores, 3)}")
    lines.append(f"Mean F1  on UNSEEN regions: {f1_scores.mean():.3f} (+/- {f1_scores.std():.3f})")
    lines.append(f"Mean AUC on UNSEEN regions: {auc_scores.mean():.3f} (+/- {auc_scores.std():.3f})")

    return "\n".join(lines), f1_scores.mean()


def main():
    print("Step 1: fetching real ENSO (ONI) and IOD (DMI) data from NOAA...")
    oni = fetch_oni_data()
    iod = fetch_iod_data()

    print(f"\nStep 2: loading rainfall data from {RAINFALL_CSV_PATH}...")
    raw = load_rainfall_data(RAINFALL_CSV_PATH)
    prepared = engineer_features_and_labels(raw, oni, iod, FLOOD_THRESHOLD_STD)
    X, y, groups, numeric_features = build_feature_matrix(prepared)

    rf_model = RandomForestClassifier(
        n_estimators=400, max_depth=6, min_samples_leaf=6,
        random_state=RANDOM_SEED, class_weight="balanced"
    )
    logreg_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED)
    )
    hgb_model = HistGradientBoostingClassifier(
        max_depth=4, learning_rate=0.05, max_iter=300,
        random_state=RANDOM_SEED, class_weight="balanced"
    )

    results = {}
    reports = []
    for name, model in [
        ("Random Forest", rf_model),
        ("Logistic Regression (baseline)", logreg_model),
        ("HistGradientBoosting", hgb_model),
    ]:
        report_text, mean_f1 = evaluate_model(name, model, X, y, groups)
        reports.append(report_text)
        results[name] = mean_f1

    best_name = max(results, key=results.get)
    print(f"\nBest model by leave-region-out F1: {best_name} ({results[best_name]:.3f})")

    rf_model.fit(X, y)
    importances = sorted(zip(X.columns, rf_model.feature_importances_), key=lambda x: -x[1])
    top_features = "\n".join(f"  {name:20s} {imp:.3f}" for name, imp in importances)

    report = "\n\n".join([
        "=" * 65,
        "FLOOD MODEL V2 - LEAKAGE-SAFE LABELS + ENSO + IOD - REPORT",
        "=" * 65,
        f"Regions: {groups.nunique()} | Rows: {len(X)}",
        *reports,
        f"\nBest model (leave-region-out F1): {best_name}",
        f"\nRandom Forest feature importances (all {len(importances)} features):\n{top_features}",
        "\nINTERPRETATION GUIDE:\n"
        "- Compare 'leave-region-out' numbers here to your previous run's.\n"
        "  Small, believable gains (a few points of F1/AUC) mean the extra\n"
        "  signal (IOD, rate-of-change, leakage-safe labels) is helping.\n"
        "- If ONI_PRE_MONSOON or IOD_PRE_MONSOON rank near the top of feature\n"
        "  importance, the climate signals are doing real work.\n"
        "- If JAS_HIST_STD ranks high, it likely means some regions are just\n"
        "  inherently more rainfall-variable than others, and that variability\n"
        "  itself is predictive of extreme-year risk - a legitimate finding.\n"
        "- Do NOT chase F1/AUC above roughly 0.75-0.8 on the leave-region-out\n"
        "  fold by adding more features - past that point you are far more\n"
        "  likely reintroducing leakage than finding real signal."
    ])
    print("\n" + report)
    with open(REPORT_OUTPUT_PATH, "w") as f:
        f.write(report)

    joblib.dump({
        "model": rf_model,
        "numeric_features": numeric_features,
        "all_columns": list(X.columns),
    }, MODEL_OUTPUT_PATH)
    print(f"\nModel saved to: {MODEL_OUTPUT_PATH}")
    print(f"Report saved to: {REPORT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()