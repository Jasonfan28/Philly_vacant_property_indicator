"""
Builder for 04b and 05 Python/Colab notebooks. Emits three .ipynb files:

  - 04b_model_validation.ipynb   ← spatial CV, LOGO CV, prediction CIs, sanity
  - 04b_vpi_comparison.ipynb     ← per-zip AUC, ward summary, equity audit
  - 05_output_analysis.ipynb     ← category & capacity analysis, spatial maps

All three CONSUME the outputs of 04a (PhillyStat360/data_py/) and the original
raw inputs (PhillyStat360/data/ and rawdata/). They WRITE to data_py/ so Python
and R outputs don't collide.

Re-run this script to regenerate the .ipynbs after edits.
"""

import json
import pathlib
import hashlib

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def make_cells():
    cells = []
    def md(text):
        cells.append({
            "cell_type": "markdown", "metadata": {},
            "source": text.splitlines(keepends=True),
        })
    def code(src):
        cells.append({
            "cell_type": "code", "metadata": {},
            "execution_count": None, "outputs": [],
            "source": src.splitlines(keepends=True),
        })
    return cells, md, code


def write_notebook(cells, name):
    out_path = pathlib.Path(__file__).parent / name
    nb = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "colab": {"name": name, "provenance": []},
        },
        "cells": cells,
    }
    # Add cell IDs
    for i, c in enumerate(nb["cells"]):
        c["id"] = f"cell-{i:03d}"
    out_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Wrote {out_path} ({len(cells)} cells)")


# --------------------------------------------------------------------------
# Shared boilerplate: Drive mount, path setup, package imports
# --------------------------------------------------------------------------
SETUP_DRIVE_MOUNT = """\
# Mount Google Drive (run once per Colab session)
from google.colab import drive
drive.mount('/content/drive')
"""

SETUP_PATHS = """\
import os
from pathlib import Path

ROOT       = Path('/content/drive/MyDrive/PhillyStat_R/PhillyStat360')
RAW_PATH   = ROOT / 'rawdata'         # original CSVs / geojsons
DATA_PATH  = ROOT / 'data'            # R-side outputs (features_residential.csv, ovs_residential.csv)
PY_PATH    = ROOT / 'data_py'         # 04a Python outputs (predictions, models, calibrators)
OUT_PATH   = ROOT / 'data_py'         # we WRITE here too
GRAPH_PATH = ROOT / 'graphs' / 'python'

OUT_PATH.mkdir(parents=True, exist_ok=True)
GRAPH_PATH.mkdir(parents=True, exist_ok=True)

assert (PY_PATH / 'all_predictions_rf.csv').exists(), \\
    f'04a outputs not found at {PY_PATH}. Run 04a_tidymodeling.ipynb first.'
print('All paths OK. PY_PATH =', PY_PATH)
"""

SETUP_IMPORTS_BASIC = """\
import json
import joblib
import warnings
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (roc_auc_score, average_precision_score,
                             roc_curve, brier_score_loss,
                             confusion_matrix, classification_report)

warnings.filterwarnings('ignore', category=UserWarning)
sns.set_theme(style='whitegrid', context='talk')
SEED = 42
"""

LOAD_PREDICTIONS = """\
preds = pd.read_csv(PY_PATH / 'all_predictions_rf.csv', low_memory=False)
print(f"Loaded predictions: {len(preds):,} rows × {preds.shape[1]} cols")

# Sanity check on expected columns from 04a (post-ensemble)
expected = ['parcel_number', 'ovs', 'ensemble_prob', 'ensemble_prob_raw',
            'risk_score', 'qtile_tier', 'ensemble_flag',
            'rf_prob', 'logit_prob']
missing = [c for c in expected if c not in preds.columns]
if missing:
    print(f"  [!] Missing expected columns: {missing}")
    print("  Re-run 04a_tidymodeling.ipynb — this notebook expects the post-ensemble export.")
else:
    print("  All expected columns present.")
"""


# ==========================================================================
# 04b_model_validation.ipynb
# ==========================================================================
def build_04b_model_validation():
    cells, md, code = make_cells()

    md("""# PhillyStat360 — 04b: Model Validation (Python / Colab port)

Port of `code/04b_model_validation.Rmd`. Extends 04a with four validation
components:

1. **Spatial cross-validation** by ZIP code (`GroupKFold`) — prevents the
   spatial-autocorrelation leakage that inflates standard k-fold estimates.
2. **LOGO CV** (`LeaveOneGroupOut`) — holds out one entire ZIP at a time;
   the hardest generalization test for citywide deployment.
3. **Prediction confidence intervals** — for `RandomForestClassifier`,
   computed from the variance across individual decision trees (a Python
   analog to ranger's infinitesimal jackknife).
4. **Sanity checks** — feature importance, calibration curve, partial
   dependence, known-vacant scoring.

> **Production model is the ensemble (Logit + RF), `ensemble_prob` from
> 04a.** Spatial CV / LOGO / CIs in this notebook are computed on the RF
> half because (a) only RF has tree variance for CI estimation and (b)
> spatial CV on RF is conservative — Logit is a simpler model that
> generalizes more smoothly, so the ensemble's spatial AUC will be at
> least as good as RF alone. Calibration plots, known-vacant scoring,
> and the high-uncertainty filter all use **`ensemble_prob`** (production
> score) so the validation reflects what the dashboard actually shows.

> **Runtime warning.** Spatial CV refits 10 RF models, LOGO refits one per
> ZIP (~45). Both default to a 10% subsample (sample flags below) so the
> notebook completes in ~10 min. Set the flags to `False` for a final pass.
""")

    md("""## 0. Setup""")
    code(SETUP_DRIVE_MOUNT)
    code(SETUP_PATHS)
    code(SETUP_IMPORTS_BASIC + """\

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.model_selection import (GroupKFold, LeaveOneGroupOut,
                                     train_test_split)
from sklearn.calibration import calibration_curve
from sklearn.inspection import partial_dependence

TRAIN_CUTOFF = pd.Timestamp('2025-10-01')
""")

    md("""## 1. Load data""")
    code("""\
features_df = pd.read_csv(DATA_PATH / 'features_residential.csv', low_memory=False)
preds       = pd.read_csv(PY_PATH   / 'all_predictions_rf.csv',  low_memory=False)
preds['parcel_number'] = preds['parcel_number'].astype(str)
thresholds  = pd.read_csv(PY_PATH   / 'model_thresholds.csv')

# Load 04a's pre-fit logit and ensemble calibrator so we can score test parcels
# with the production ensemble without refitting.
logit_fit  = joblib.load(PY_PATH / 'model_logit_final.joblib')
calibrators = joblib.load(PY_PATH / 'calibrators.joblib')
cal_ens     = calibrators['ensemble']
print('Loaded 04a artifacts: model_logit_final.joblib + calibrators.joblib')

ens_thresh = float(thresholds.loc[thresholds['model'] == 'ensemble', 'threshold'].iloc[0])
print(f"features_residential: {len(features_df):,} rows  |  OVS=1: {features_df['ovs'].mean():.1%}")
print(f"predictions:          {len(preds):,} rows")
print(f"Ensemble Youden threshold from 04a: {ens_thresh:.4f} (calibrated)")

# Verify ensemble columns are present
required = ['ensemble_prob', 'ensemble_prob_raw', 'ensemble_flag', 'risk_score']
missing  = [c for c in required if c not in preds.columns]
assert not missing, (
    f"Missing ensemble columns {missing} in all_predictions_rf.csv. "
    f"Re-run 04a_tidymodeling.ipynb."
)
""")

    md("""## 2. Rebuild model dataset (mirrors 04a)

Same `model_vars`, same imputation, same split. Only used to refit RF for the
spatial / LOGO CV folds.
""")
    code("""\
# Mirrors the 04a model_vars (post-2026-04-28 audit, with C&S history)
model_vars = [
    'n_violations_total', 'n_violations_recent',
    'n_violations_2yr',   'n_violations_3yr',   'n_violations_5yr',
    'n_distinct_codes',
    'viol_trend_3v5', 'viol_accel_2v3',
    'n_repeat_codes', 'resolution_rate',
    'has_fire_safety_code',
    'days_since_last_viol',
    'license_lapse_rate',
    'exterior_condition', 'building_age',
    'log_livable_area',   'is_poor_condition',
    'years_since_sale',
    'n_cs_total', 'cs_span_days', 'days_since_last_cs',
    'n_transfers_total', 'n_transfers_5yr', 'n_transfers_3yr',
    'n_deed_transfers',
    'had_sheriff_sale', 'sheriff_sale_recent', 'n_sheriff_sales',
    'log_price_change',
    'days_since_last_transfer',
    'nbr_ovs_rate_zip', 'nbr_ovs_rate_tract',
    'nbr_n_vacant_zip', 'nbr_n_vacant_tract',
]
print(f"Total features in model_vars: {len(model_vars)}")
""")

    code("""\
model_df = features_df.dropna(subset=['exterior_condition', 'building_age']).copy()

for col in ['days_since_last_viol', 'days_since_last_transfer']:
    if col in model_df.columns and model_df[col].isna().any():
        model_df[col] = model_df[col].fillna(model_df[col].median()).astype(float)

if 'days_oldest_open_viol' in model_df.columns:
    model_df['days_oldest_open_viol'] = model_df['days_oldest_open_viol'].fillna(0).astype(int)

assert 'zip_code' in model_df.columns, \\
    'zip_code missing — required for spatial CV'

n_zips = model_df['zip_code'].nunique(dropna=True)
print(f"model_df: {len(model_df):,} parcels  |  {n_zips} zip codes  |  OVS=1: {model_df['ovs'].mean():.1%}")
""")

    code("""\
# Replicate 04a's stratified 70/30 split
train_df, test_df = train_test_split(
    model_df, test_size=0.30, random_state=SEED,
    stratify=model_df['ovs'].astype(int).values,
)
train_df = train_df.copy(); test_df = test_df.copy()
print(f"Train: {len(train_df):,}  |  Test: {len(test_df):,}")
""")

    md("""## 3. Recipe / spec (mirrors 04a)

`SimpleImputer(median) → VarianceThreshold(0) → RandomForestClassifier(class_weight='balanced')`.
No SMOTE — matches the post-2026-04-28 production pipeline.
""")

    code("""\
# Reuse RF tune params from 04a if available; otherwise sensible defaults.
rf_tune_csv = PY_PATH / 'rf_tune_results.csv'
if rf_tune_csv.exists():
    s = pd.read_csv(rf_tune_csv)
    BEST_MTRY  = int(s['mtry'].iloc[0])
    BEST_MIN_N = int(s['min_n'].iloc[0])
    print(f"Loaded RF params from 04a tune: mtry={BEST_MTRY} | min_n={BEST_MIN_N}")
else:
    BEST_MTRY  = int(np.floor(np.sqrt(len(model_vars))))
    BEST_MIN_N = 5
    print(f"No saved tune — defaults: mtry={BEST_MTRY} | min_n={BEST_MIN_N}")

def make_rf_pipeline(n_estimators=500, **rf_kwargs):
    return SkPipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('vt',     VarianceThreshold(0.0)),
        ('rf',     RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_features=BEST_MTRY,
                        min_samples_leaf=BEST_MIN_N,
                        class_weight='balanced',
                        random_state=SEED, n_jobs=-1, **rf_kwargs)),
    ])
""")

    md("""## 4. Spatial Cross-Validation by ZIP

`GroupKFold` with 10 folds, grouped by `zip_code`. Each fold holds out roughly
10% of zip codes — every parcel from a given ZIP is either entirely in train
or entirely in test, so the model never sees a parcel's neighbors during
training.

> **Sample flag.** `SAMPLE_FOR_CV = True` runs on a 10% stratified sample (~1
> minute). Set to `False` for a final report.
""")

    code("""\
SAMPLE_FOR_CV = True   # 10% sample — flip to False for final pass

CV_TREES = 200  # CV-only RF, fewer trees for speed; full fit later uses 500.

if SAMPLE_FOR_CV:
    cv_idx = (
        train_df.groupby('zip_code', observed=True, group_keys=False)
                .apply(lambda g: g.sample(frac=0.10, random_state=SEED))
                .index
    )
    cv_data = train_df.loc[cv_idx]
    print(f"Spatial CV on 10% sample: {len(cv_data):,} rows  |  "
          f"{cv_data['zip_code'].nunique()} zip codes")
else:
    cv_data = train_df
    print(f"Spatial CV on full train: {len(cv_data):,} rows  |  "
          f"{cv_data['zip_code'].nunique()} zip codes")
""")

    code("""\
def cv_zip_metrics(cv_data, n_splits=10, n_estimators=200):
    \"\"\"Run group-k-fold CV grouped by zip_code; return per-fold metrics dataframe.\"\"\"
    X = cv_data[model_vars]
    y = cv_data['ovs'].astype(int).values
    groups = cv_data['zip_code'].astype(str).fillna('NA').values

    gkf = GroupKFold(n_splits=n_splits)

    rows = []
    for fold_i, (tr, va) in enumerate(gkf.split(X, y, groups=groups)):
        pipe = make_rf_pipeline(n_estimators=n_estimators)
        pipe.fit(X.iloc[tr], y[tr])
        p = pipe.predict_proba(X.iloc[va])[:, 1]
        # Youden best
        fpr, tpr, _ = roc_curve(y[va], p)
        rows.append({
            'fold':     fold_i + 1,
            'n_train':  len(tr),
            'n_test':   len(va),
            'n_groups_held_out': len(set(groups[va])),
            'roc_auc':  roc_auc_score(y[va], p),
            'pr_auc':   average_precision_score(y[va], p),
            'j_index':  float((tpr - fpr).max()),
        })
    return pd.DataFrame(rows)

print(f"Running spatial CV (10 folds × {CV_TREES}-tree RF) — this takes a few minutes…")
spatial_cv = cv_zip_metrics(cv_data, n_splits=10, n_estimators=CV_TREES)
print('Done.')
spatial_cv.round(4)
""")

    code("""\
spatial_cv_summary = spatial_cv[['roc_auc', 'pr_auc', 'j_index']].agg(['mean', 'std']).T
spatial_cv_summary['std_err'] = spatial_cv_summary['std'] / np.sqrt(len(spatial_cv))
spatial_cv_summary[['mean', 'std_err']].round(4)
""")

    code("""\
# Per-fold visual
fig, ax = plt.subplots(figsize=(10, 4.5))
for col, color in [('roc_auc', 'steelblue'), ('j_index', 'tomato')]:
    ax.plot(spatial_cv['fold'], spatial_cv[col], '-o', color=color, label=col, lw=1.5)
    ax.axhline(spatial_cv[col].mean(), ls='--', color=color, alpha=0.4)
ax.set_xlabel('Fold')
ax.set_ylabel('Metric value')
ax.set_title('Spatial CV: Per-Fold Performance by ZIP-Code Group')
ax.legend()
ax.set_xticks(spatial_cv['fold'])
plt.tight_layout()
plt.savefig(GRAPH_PATH / 'spatial_cv_performance.png', dpi=200, bbox_inches='tight')
plt.show()
""")

    md("""## 5. LOGO Cross-Validation (Leave-One-ZIP-Out)

`LeaveOneGroupOut`: each ZIP held out once. Hardest generalization test —
simulates deploying to a brand-new neighborhood with no training history.

> **Sample flag.** `SAMPLE_FOR_LOGO = True` uses a random subset of 15 ZIPs
> instead of all ~45. Default ON for ~5-min runtime.
""")

    code("""\
SAMPLE_FOR_LOGO = True
N_LOGO_ZIPS = 15

logo_data = cv_data  # reuse the spatial-CV sample (already 10% if flag on)

if SAMPLE_FOR_LOGO:
    rng = np.random.default_rng(SEED)
    sampled_zips = rng.choice(logo_data['zip_code'].dropna().unique(),
                              size=min(N_LOGO_ZIPS, logo_data['zip_code'].nunique()),
                              replace=False)
    logo_data = logo_data[logo_data['zip_code'].isin(sampled_zips)]
    print(f"LOGO on {len(sampled_zips)} sampled ZIPs ({len(logo_data):,} parcels)")
else:
    print(f"LOGO on all {logo_data['zip_code'].nunique()} ZIPs ({len(logo_data):,} parcels)")
""")

    code("""\
def logo_zip_metrics(data, n_estimators=200):
    \"\"\"LeaveOneGroupOut by zip_code. Returns per-zip metrics.\"\"\"
    X = data[model_vars]
    y = data['ovs'].astype(int).values
    groups = data['zip_code'].astype(str).fillna('NA').values

    logo = LeaveOneGroupOut()
    rows = []
    n_folds = logo.get_n_splits(groups=groups)
    for fold_i, (tr, va) in enumerate(logo.split(X, y, groups=groups)):
        zip_held = groups[va][0]
        n_pos    = int(y[va].sum())
        if n_pos < 2 or len(va) < 50:
            # Skip ZIPs too small/sparse to compute reliable AUC
            continue
        pipe = make_rf_pipeline(n_estimators=n_estimators)
        pipe.fit(X.iloc[tr], y[tr])
        p = pipe.predict_proba(X.iloc[va])[:, 1]
        rows.append({
            'fold':     fold_i + 1,
            'zip':      zip_held,
            'n':        len(va),
            'n_vacant': n_pos,
            'roc_auc':  roc_auc_score(y[va], p),
            'pr_auc':   average_precision_score(y[va], p) if n_pos > 0 else np.nan,
        })
        if (fold_i + 1) % 5 == 0:
            print(f"  fold {fold_i+1}/{n_folds}: zip={zip_held}, AUC={rows[-1]['roc_auc']:.3f}")
    return pd.DataFrame(rows)

logo_results = logo_zip_metrics(logo_data, n_estimators=CV_TREES)
print(f"\\nLOGO complete: {len(logo_results)} ZIPs evaluated")
print(f"  Mean AUC:   {logo_results['roc_auc'].mean():.4f}")
print(f"  Median AUC: {logo_results['roc_auc'].median():.4f}")
print(f"  ZIPs AUC < 0.70: {int((logo_results['roc_auc'] < 0.70).sum())}")
""")

    code("""\
fig, ax = plt.subplots(figsize=(10, max(6, 0.3 * len(logo_results))))
df_plot = logo_results.sort_values('roc_auc')
colors = ['tomato' if a < 0.70 else 'steelblue' for a in df_plot['roc_auc']]
ax.barh(df_plot['zip'].astype(str), df_plot['roc_auc'], color=colors)
ax.axvline(logo_results['roc_auc'].mean(), ls='--', color='gray',
           label=f"mean AUC = {logo_results['roc_auc'].mean():.3f}")
ax.set_xlim(0, 1)
ax.set_xlabel('ROC-AUC')
ax.set_title('LOGO CV: AUC by Held-Out ZIP\\n(red bars = AUC < 0.70; generalization concern)')
ax.legend()
plt.tight_layout()
plt.savefig(GRAPH_PATH / 'logo_cv_by_zip.png', dpi=200, bbox_inches='tight')
plt.show()
""")

    md("""## 6. Prediction Confidence Intervals

sklearn's `RandomForestClassifier` exposes the underlying `estimators_` (one
fit per tree). We compute a per-parcel SE as the std of tree-level probability
predictions divided by sqrt(n_trees) — a simpler analog to ranger's
infinitesimal jackknife. The 95% CI is `prob ± 1.96 × SE`.

A parcel with `prob = 0.6` and `ci_width = 0.05` is much more confidently
flagged than one with `prob = 0.6` and `ci_width = 0.30`.
""")

    code("""\
# Refit the production RF (500 trees) on the full train set so we can pull
# per-tree predictions for SE estimation.
print('Refitting RF (500 trees) for CI estimation…')
rf_full = make_rf_pipeline(n_estimators=500)
rf_full.fit(train_df[model_vars], train_df['ovs'].astype(int).values)
print('Done.')
""")

    code("""\
def rf_predict_with_ci(pipe, X_df):
    \"\"\"Predict with per-parcel CIs from the variance across trees.

    Returns a DataFrame with rf_prob, rf_se, ci_lower, ci_upper, ci_width.
    \"\"\"
    # Feed X through the pre-RF pipeline steps
    X_imputed = pipe.named_steps['impute'].transform(X_df)
    X_vt      = pipe.named_steps['vt'].transform(X_imputed)
    rf_clf    = pipe.named_steps['rf']

    # Each tree predicts a 0/1 vote (or proba if using predict_proba per tree).
    # Use predict_proba on each tree, take prob_class_1.
    tree_preds = np.stack([
        tree.predict_proba(X_vt)[:, 1] for tree in rf_clf.estimators_
    ])  # shape: (n_trees, n_samples)

    rf_prob = tree_preds.mean(axis=0)
    # SE: sample std of tree probs / sqrt(n_trees) — analog to bootstrap SE
    rf_se   = tree_preds.std(axis=0, ddof=1) / np.sqrt(len(rf_clf.estimators_))
    ci_lower = np.maximum(0, rf_prob - 1.96 * rf_se)
    ci_upper = np.minimum(1, rf_prob + 1.96 * rf_se)
    return pd.DataFrame({
        'rf_prob':  rf_prob,
        'rf_se':    rf_se,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'ci_width': ci_upper - ci_lower,
    }, index=X_df.index)

print('Computing CIs on test set…')
ci_df = rf_predict_with_ci(rf_full, test_df[model_vars])
test_with_ci = pd.concat([test_df.reset_index(drop=True),
                          ci_df.reset_index(drop=True)], axis=1)

# Score test parcels with the production ensemble too (logit + RF, calibrated).
# logit raw probs come from the saved logit; rf raw probs come from rf_full
# (this notebook's refit, slightly different from 04a's due to different RF seed
# behavior across spatial fold splits — but the calibration shape holds).
test_with_ci['logit_prob_raw']    = logit_fit.predict_proba(test_df[model_vars])[:, 1]
test_with_ci['rf_prob_raw']       = test_with_ci['rf_prob']
test_with_ci['ensemble_prob_raw'] = 0.5 * test_with_ci['logit_prob_raw'] + 0.5 * test_with_ci['rf_prob_raw']
test_with_ci['ensemble_prob']     = cal_ens.transform(test_with_ci['ensemble_prob_raw'].values)
# Pull the production ensemble_flag (top 1% by raw rank) for the high-uncertainty
# review below — that's what shows up in the city dashboard, not the calibrated
# Youden threshold (which would flag ~14% of parcels).
ens_preds_for_test = preds[['parcel_number', 'ensemble_flag', 'qtile_tier']].copy()
test_with_ci = test_with_ci.merge(
    ens_preds_for_test, on='parcel_number', how='left',
)
print(f"Ensemble scored on test set: AUC = "
      f"{roc_auc_score(test_with_ci['ovs'].astype(int), test_with_ci['ensemble_prob']):.4f}")

summary = pd.DataFrame({
    'metric': ['mean P(vacant)', 'mean SE', 'mean CI width',
               '% CI < 0.10 (high confidence)', '% CI > 0.30 (uncertain)'],
    'value': [
        f"{test_with_ci['rf_prob'].mean():.4f}",
        f"{test_with_ci['rf_se'].mean():.4f}",
        f"{test_with_ci['ci_width'].mean():.4f}",
        f"{(test_with_ci['ci_width'] < 0.10).mean():.1%}",
        f"{(test_with_ci['ci_width'] > 0.30).mean():.1%}",
    ],
})
summary
""")

    code("""\
# CI width vs predicted probability — where is the model uncertain?
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

ax = axes[0]
binned = test_with_ci.assign(prob_bin=pd.cut(test_with_ci['rf_prob'],
                                             bins=np.arange(0, 1.05, 0.05)))
agg = binned.groupby('prob_bin', observed=True).agg(
    mean_prob=('rf_prob', 'mean'),
    mean_ci=('ci_width', 'mean'),
    n=('rf_prob', 'size'),
).reset_index().dropna()
ax.plot(agg['mean_prob'], agg['mean_ci'], '-', color='steelblue', lw=2)
ax.scatter(agg['mean_prob'], agg['mean_ci'], color='steelblue',
           s=20 + agg['n'].rank(pct=True) * 100, alpha=0.8, zorder=3)
ax.set_xlabel('Predicted P(vacant)')
ax.set_ylabel('Mean CI width')
ax.set_title('CI width vs predicted probability')

ax = axes[1]
for label, color in [(0, 'steelblue'), (1, 'tomato')]:
    sub = test_with_ci.loc[test_with_ci['ovs'] == label, 'ci_width']
    ax.hist(sub, bins=40, alpha=0.5, density=True,
            label=f"OVS = {label}", color=color)
ax.set_xlabel('CI width')
ax.set_ylabel('Density')
ax.set_title('CI width distribution by true OVS')
ax.legend()

plt.tight_layout()
plt.savefig(GRAPH_PATH / 'prediction_ci_analysis.png', dpi=200, bbox_inches='tight')
plt.show()
""")

    code("""\
# High-uncertainty flagged parcels — these are the "we flagged but unsure"
# cases the city should treat with extra care. We use the PRODUCTION flag
# (ensemble_flag = top 1% by raw ensemble rank) — that's what the dashboard
# shows. CIs themselves are still RF-only (only RF has tree variance).
high_unc_mask = (test_with_ci['ci_width'] > 0.30) & (test_with_ci['ensemble_flag'] == 1)
print(f"Parcels in production flag set (top 1% by ensemble): "
      f"{int(test_with_ci['ensemble_flag'].sum()):,}")
print(f"Of those, with CI > 0.30 (high RF uncertainty): "
      f"{int(high_unc_mask.sum()):,}")

high_unc = (
    test_with_ci[high_unc_mask]
    .sort_values('ci_width', ascending=False)
    [['parcel_number', 'ovs', 'ensemble_prob', 'rf_prob', 'rf_se',
      'ci_lower', 'ci_upper', 'ci_width', 'qtile_tier']]
    .head(20)
    .round(4)
)
high_unc
""")

    md("""## 7. Sanity Checks

### 7a. Feature importance""")

    code("""\
rf_clf = rf_full.named_steps['rf']
vt     = rf_full.named_steps['vt']
surviving = [f for f, k in zip(model_vars, vt.get_support()) if k]
imp_df = pd.DataFrame({
    'feature':    surviving,
    'importance': rf_clf.feature_importances_,
}).sort_values('importance', ascending=False).head(20)

fig, ax = plt.subplots(figsize=(8, 7))
ax.barh(imp_df['feature'][::-1], imp_df['importance'][::-1], color='steelblue')
ax.set_title('Top 20 features by impurity importance')
plt.tight_layout()
plt.savefig(GRAPH_PATH / 'rf_vip_04b.png', dpi=200, bbox_inches='tight')
plt.show()
imp_df.round(4)
""")

    md("""### 7b. Calibration curve""")

    code("""\
y_true = test_with_ci['ovs'].astype(int).values

# Calibration for RF (raw) and ensemble (calibrated, what the dashboard shows)
prob_pred_rf,  prob_true_rf  = calibration_curve(
    y_true, test_with_ci['rf_prob'].values, n_bins=10, strategy='quantile')
prob_pred_ens, prob_true_ens = calibration_curve(
    y_true, test_with_ci['ensemble_prob'].values, n_bins=10, strategy='quantile')

fig, ax = plt.subplots(figsize=(8, 6.5))
lim = max(prob_pred_rf.max(), prob_pred_ens.max(),
          prob_true_rf.max(), prob_true_ens.max()) * 1.05
ax.plot([0, lim], [0, lim], '--', color='gray', label='Perfect calibration')
ax.plot(prob_pred_rf,  prob_true_rf,  '-o', color='steelblue', lw=1.5,
        label='RF (raw)', alpha=0.7)
ax.plot(prob_pred_ens, prob_true_ens, '-o', color='black',     lw=2.5,
        label='Vacancy Risk Score (ensemble, calibrated)')
ax.set_xlabel('Mean predicted P(vacant)')
ax.set_ylabel('Observed vacancy rate')
ax.set_title('Calibration: ensemble (production) vs RF alone')
ax.legend()
plt.tight_layout()
plt.savefig(GRAPH_PATH / 'calibration_curve.png', dpi=200, bbox_inches='tight')
plt.show()
""")

    md("""### 7c. Known-vacant scoring check

If `cs_truly_active`, `had_vacancy_license`, etc. are present in the test set
(they're excluded from `model_vars` as leaks but still in features_residential),
we can check the model assigns higher scores to those parcels. If it doesn't,
the model is ignoring its strongest non-leakage proxies.
""")

    code("""\
def safe_col(df, col, default=0):
    return df[col] if col in df.columns else pd.Series(default, index=df.index)

groups = pd.Series('No strong signal', index=test_with_ci.index)
groups[safe_col(test_with_ci, 'cs_truly_active') == 1] = 'Clean & Seal active (strongest)'
groups[(groups == 'No strong signal') & (safe_col(test_with_ci, 'had_vacancy_license') == 1)] = 'Had vacancy license'
groups[(groups == 'No strong signal') & (safe_col(test_with_ci, 'has_open_vacancy_kw') == 1)] = 'Open vacancy-kw violation'

# Sanity scoring uses the PRODUCTION ensemble probability, and flag rate uses
# the production ensemble_flag (top 1% by raw rank), so the table answers
# "how does the dashboard actually treat parcels in each known-signal group?"
sanity = (
    test_with_ci.assign(group=groups)
    .groupby('group', observed=True)
    .agg(n=('ensemble_prob', 'size'),
         ovs1_rate=('ovs', lambda s: float((s == 1).mean())),
         mean_ensemble_prob=('ensemble_prob', 'mean'),
         mean_rf_prob=('rf_prob', 'mean'),
         pct_flagged=('ensemble_flag', lambda s: float(s.mean())))
    .sort_values('mean_ensemble_prob', ascending=False)
    .reset_index()
)
sanity['ovs1_rate']          = sanity['ovs1_rate'].map('{:.1%}'.format)
sanity['pct_flagged']        = sanity['pct_flagged'].map('{:.1%}'.format)
sanity['mean_ensemble_prob'] = sanity['mean_ensemble_prob'].round(4)
sanity['mean_rf_prob']       = sanity['mean_rf_prob'].round(4)
sanity['n']                  = sanity['n'].map('{:,}'.format)
sanity
""")

    md("""### 7d. Partial dependence: `n_violations_total`""")

    code("""\
# sklearn partial_dependence — works on the full pipeline.
# NB: in older sklearn the result key is 'values' instead of 'grid_values'.
pdp = partial_dependence(
    rf_full, X=train_df[model_vars].sample(min(20000, len(train_df)), random_state=SEED),
    features=['n_violations_total'],
    grid_resolution=30, kind='average', percentiles=(0.01, 0.99),
)
# Handle both old and new sklearn API (values vs grid_values)
xs = pdp.get('grid_values', pdp.get('values'))[0]
ys = pdp['average'][0]

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(xs, ys, color='steelblue', lw=2)
ax.set_xlabel('n_violations_total')
ax.set_ylabel('Marginal P(vacant)')
ax.set_title('Partial dependence: P(vacant) vs n_violations_total\\n'
             '(other features held at their distribution; expect monotone increase)')
plt.tight_layout()
plt.savefig(GRAPH_PATH / 'partial_dependence_violations.png', dpi=200, bbox_inches='tight')
plt.show()
""")

    md("""## 8. Export validation results""")

    code("""\
# Export test predictions with CIs — includes ensemble_prob (production score)
# alongside RF stats so downstream consumers don't have to join back.
ci_export_cols = ['parcel_number', 'ovs',
                  'ensemble_prob', 'ensemble_prob_raw', 'ensemble_flag',
                  'rf_prob', 'rf_se', 'ci_lower', 'ci_upper', 'ci_width']
test_with_ci[ci_export_cols].to_csv(OUT_PATH / 'predictions_with_ci.csv', index=False)
print(f"Exported predictions_with_ci.csv: {len(test_with_ci):,} rows")

# Export CV metric tables
spatial_cv.assign(method='spatial_zip').to_csv(OUT_PATH / 'spatial_cv_metrics.csv', index=False)
logo_results.assign(method='logo_zip').to_csv(OUT_PATH / 'logo_cv_metrics.csv', index=False)
print('Exported spatial_cv_metrics.csv, logo_cv_metrics.csv')

# Test-set ensemble metrics — what the dashboard actually deploys.
y_te = test_with_ci['ovs'].astype(int).values
ens_p = test_with_ci['ensemble_prob'].values
ens_test_auc    = roc_auc_score(y_te, ens_p)
ens_test_pr_auc = average_precision_score(y_te, ens_p)

# Validation summary — RF spatial CV is the conservative lower bound; ensemble
# row reports the actual test-set headline number.
summary_tbl = pd.DataFrame({
    'check': [
        'Production model (test set) — ROC-AUC',
        'Production model (test set) — PR-AUC',
        'Spatial CV mean ROC-AUC (RF only, conservative)',
        'Spatial CV mean PR-AUC (RF only, conservative)',
        'LOGO CV mean ROC-AUC (RF only, conservative)',
        'LOGO CV mean PR-AUC (RF only, conservative)',
        'Test mean CI width (RF tree-variance)',
        '% production-flagged parcels with CI > 0.30',
    ],
    'value': [
        f"{ens_test_auc:.4f}",
        f"{ens_test_pr_auc:.4f}",
        f"{spatial_cv['roc_auc'].mean():.4f}",
        f"{spatial_cv['pr_auc'].mean():.4f}",
        f"{logo_results['roc_auc'].mean():.4f}",
        f"{logo_results['pr_auc'].mean():.4f}",
        f"{test_with_ci['ci_width'].mean():.4f}",
        f"{((test_with_ci['ci_width'] > 0.30) & (test_with_ci['ensemble_flag'] == 1)).mean():.2%}",
    ],
})
summary_tbl.to_csv(OUT_PATH / 'validation_summary.csv', index=False)
summary_tbl
""")

    md("""---
**Done.** Outputs in `data_py/`. To run a final pass with no sampling, set
`SAMPLE_FOR_CV = False` and `SAMPLE_FOR_LOGO = False` at the top of §4 / §5
(expect ~1–2 hours).
""")

    write_notebook(cells, '04b_model_validation.ipynb')


# ==========================================================================
# 04b_vpi_comparison.ipynb
# ==========================================================================
def build_04b_vpi_comparison():
    cells, md, code = make_cells()

    md("""# PhillyStat360 — 04b: VPI Comparison & Spatial Analysis (Python / Colab port)

Port of `code/04b_vpi_comparison.Rmd`. Joins predictions to OPA metadata,
reports per-ZIP and per-building-type AUC, ward-level summaries, and an
equity audit by census-tract poverty quintile.

> **Equity audit note.** The R version pulls ACS poverty rates via
> `tidycensus`. The Python version expects either:
> - A pre-saved `acs_poverty_phl.csv` in `data/` (one row per census tract
>   with `census_tract_key` and `poverty_rate` columns), OR
> - The `census` Python package + a Census API key in `CENSUS_API_KEY`.
>
> If neither is available, the equity audit gracefully skips.
""")

    md("""## 0. Setup""")
    code(SETUP_DRIVE_MOUNT)
    code(SETUP_PATHS)
    code(SETUP_IMPORTS_BASIC + """\

# geopandas only used for the optional spatial map at the end (skipped here in 04b)
try:
    import geopandas as gpd
    HAS_GPD = True
except ImportError:
    HAS_GPD = False
""")

    md("""## 1. Load data""")
    code("""\
preds = pd.read_csv(PY_PATH / 'all_predictions_rf.csv', low_memory=False)
preds['parcel_number'] = preds['parcel_number'].astype(str)
print(f"Predictions: {len(preds):,} rows")

features_full = pd.read_csv(DATA_PATH / 'features_residential.csv', low_memory=False)
features_full['parcel_number'] = features_full['parcel_number'].astype(str)
features = features_full[['parcel_number', 'zip_code', 'census_tract',
                          'exterior_condition', 'building_age']].copy()

# OPA metadata: ward + category
opa_path = RAW_PATH / 'opa_properties_public.csv'
if not opa_path.exists():
    opa_path = DATA_PATH / 'opa_properties_public.csv'
opa = pd.read_csv(opa_path, low_memory=False)
opa.columns = [c.lower().strip().replace(' ', '_') for c in opa.columns]
opa['parcel_number'] = opa['parcel_number'].astype(str)
opa = opa[['parcel_number', 'geographic_ward',
           'category_code', 'category_code_description',
           'building_code_description']].drop_duplicates(subset='parcel_number')
print(f"OPA: {len(opa):,} parcels")
""")

    code("""\
# Five-tier mapping based on the production ensemble probability
TIER_LEVELS = ['Very Unlikely (0–0.2)', 'Unlikely (0.2–0.4)',
               'Maybe (0.4–0.6)',       'Likely (0.6–0.8)',
               'Very Likely (0.8–1.0)']

def to_tier(p):
    if p < 0.2:  return TIER_LEVELS[0]
    if p < 0.4:  return TIER_LEVELS[1]
    if p < 0.6:  return TIER_LEVELS[2]
    if p < 0.8:  return TIER_LEVELS[3]
    return TIER_LEVELS[4]

# Use ensemble_prob (calibrated) as the primary probability column.
prob_col = 'ensemble_prob' if 'ensemble_prob' in preds.columns else 'rf_prob'
print(f"Using `{prob_col}` as the primary probability column")

df = (
    preds
    .merge(features, on='parcel_number', how='left')
    .merge(opa,      on='parcel_number', how='left')
)
df['prob_tier'] = pd.Categorical(
    df[prob_col].map(to_tier), categories=TIER_LEVELS, ordered=True,
)
print(f"Assembled: {len(df):,} parcels")
print(f"  zip_code coverage:        {df['zip_code'].notna().mean():.1%}")
print(f"  geographic_ward coverage: {df['geographic_ward'].notna().mean():.1%}")
""")

    md("""## 2. Per-ZIP AUC

Where does the model generalize well? Where does it struggle? Filter to ZIPs
with at least 50 parcels and 5 positives so AUC is meaningful.
""")

    code("""\
def safe_auc(group):
    if group['ovs'].nunique() < 2 or len(group) < 10:
        return np.nan
    return roc_auc_score(group['ovs'].astype(int), group[prob_col])

zip_auc = (
    df.dropna(subset=['zip_code'])
    .groupby('zip_code')
    .filter(lambda g: len(g) >= 50 and (g['ovs'] == 1).sum() >= 5)
    .groupby('zip_code')
    .apply(lambda g: pd.Series({
        'n_parcels': len(g),
        'n_vacant':  int((g['ovs'] == 1).sum()),
        'prev_rate': float((g['ovs'] == 1).mean()),
        'auc':       safe_auc(g),
    }))
    .reset_index()
    .sort_values('auc')
)
print(f"ZIPs evaluated: {len(zip_auc)}")
print(f"  Median AUC:       {zip_auc['auc'].median():.3f}")
print(f"  5th percentile:   {zip_auc['auc'].quantile(0.05):.3f}")
print(f"  ZIPs AUC < 0.70:  {int((zip_auc['auc'] < 0.70).sum())}")
zip_auc.head(10).round(3)
""")

    code("""\
fig, ax = plt.subplots(figsize=(13, 5))
df_plot = zip_auc.sort_values('auc')
colors = plt.cm.RdYlBu(df_plot['auc'].rank(pct=True))
ax.bar(range(len(df_plot)), df_plot['auc'], color=colors)
ax.axhline(0.70, color='tomato', ls='--', label='AUC = 0.70')
ax.set_xticks(range(len(df_plot)))
ax.set_xticklabels(df_plot['zip_code'].astype(str), rotation=90, fontsize=7)
ax.set_ylim(0, 1)
ax.set_ylabel('AUC')
ax.set_title(f'Model AUC by ZIP code  ({len(df_plot)} ZIPs, min 50 parcels & 5 positives)')
ax.legend()
plt.tight_layout()
plt.savefig(GRAPH_PATH / 'auc_by_zip.png', dpi=200, bbox_inches='tight')
plt.show()
""")

    md("""## 3. Per-building-type AUC""")

    code("""\
bldg_auc = (
    df.dropna(subset=['category_code_description'])
    .groupby('category_code_description')
    .filter(lambda g: len(g) >= 100 and (g['ovs'] == 1).sum() >= 10)
    .groupby('category_code_description')
    .apply(lambda g: pd.Series({
        'n_parcels': len(g),
        'n_vacant':  int((g['ovs'] == 1).sum()),
        'prev_rate': float((g['ovs'] == 1).mean()),
        'auc':       safe_auc(g),
    }))
    .reset_index()
    .sort_values('n_parcels', ascending=False)
)
bldg_auc.round(3)
""")

    md("""## 4. Tier distribution""")

    code("""\
TIER_COLORS = {
    'Very Unlikely (0–0.2)': '#4575b4',
    'Unlikely (0.2–0.4)':    '#91bfdb',
    'Maybe (0.4–0.6)':       '#ffffbf',
    'Likely (0.6–0.8)':      '#fc8d59',
    'Very Likely (0.8–1.0)': '#d73027',
}

tier_summary = (
    df.groupby('prob_tier', observed=True)
    .agg(n_parcels=('ovs', 'size'),
         n_ovs=('ovs', lambda s: int((s == 1).sum())),
         ovs_rate=('ovs', lambda s: float((s == 1).mean())),
         mean_prob=(prob_col, 'mean'))
    .reset_index()
)
tier_summary['pct'] = tier_summary['n_parcels'] / tier_summary['n_parcels'].sum()
tier_summary.round(4)
""")

    code("""\
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, ycol, ylabel, fmt in [
    (axes[0], 'n_parcels', 'N parcels',     lambda v: f"{int(v):,}"),
    (axes[1], 'ovs_rate',  'OVS=1 rate',    lambda v: f"{v:.1%}"),
]:
    bars = ax.bar(tier_summary['prob_tier'].astype(str), tier_summary[ycol],
                  color=[TIER_COLORS[t] for t in tier_summary['prob_tier'].astype(str)])
    for bar, val in zip(bars, tier_summary[ycol]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), fmt(val),
                ha='center', va='bottom', fontsize=9)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='x', rotation=20, labelsize=8)
fig.suptitle('Tier distribution', y=1.02)
plt.tight_layout()
plt.savefig(GRAPH_PATH / 'tier_distribution.png', dpi=200, bbox_inches='tight')
plt.show()
""")

    md("""## 5. Ward-level summary""")

    code("""\
ward_summary = (
    df.dropna(subset=['geographic_ward'])
    .groupby('geographic_ward')
    .apply(lambda g: pd.Series({
        'n_parcels':      len(g),
        'n_ovs':          int((g['ovs'] == 1).sum()),
        'ovs_rate':       float((g['ovs'] == 1).mean()),
        'mean_prob':      float(g[prob_col].mean()),
        'n_very_likely':  int((g['prob_tier'] == 'Very Likely (0.8–1.0)').sum()),
        'n_high_risk':    int(g['prob_tier'].isin(['Likely (0.6–0.8)', 'Very Likely (0.8–1.0)']).sum()),
        'high_risk_rate': float(g['prob_tier'].isin(['Likely (0.6–0.8)', 'Very Likely (0.8–1.0)']).mean()),
    }))
    .sort_values('mean_prob', ascending=False)
    .reset_index()
)
ward_summary.head(15).round(4)
""")

    code("""\
top20 = ward_summary.nlargest(20, 'mean_prob')
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(top20['geographic_ward'].astype(str)[::-1], top20['mean_prob'][::-1], color='#d73027')
for i, v in enumerate(top20['mean_prob'][::-1]):
    ax.text(v, i, f' {v:.3f}', va='center', fontsize=9)
ax.set_xlabel('Mean P(vacant)')
ax.set_title('Top 20 wards by mean ensemble probability')
plt.tight_layout()
plt.savefig(GRAPH_PATH / 'ward_mean_prob.png', dpi=200, bbox_inches='tight')
plt.show()
""")

    md("""## 6. Equity audit: AUC by census-tract poverty quintile

Tries three sources for ACS poverty data, in order:
1. Pre-saved `data/acs_poverty_phl.csv` (preferred — no API call needed)
2. Census API via the `census` package (requires `CENSUS_API_KEY` env var)
3. Skip with a clear message

If any neighborhood-poverty-by-tract data is available, the model is
audited for performance disparity across poverty quintiles.
""")

    code("""\
acs_poverty = None
acs_path = DATA_PATH / 'acs_poverty_phl.csv'

if acs_path.exists():
    acs_poverty = pd.read_csv(acs_path)
    if 'census_tract_key' not in acs_poverty.columns:
        # Try to derive from GEOID
        if 'GEOID' in acs_poverty.columns or 'geoid' in acs_poverty.columns:
            geoid_col = 'GEOID' if 'GEOID' in acs_poverty.columns else 'geoid'
            acs_poverty['census_tract_key'] = acs_poverty[geoid_col].astype(str).str[-6:]
    # Force string dtype + 6-char zero-pad so merge keys align with df_eq below
    acs_poverty['census_tract_key'] = (
        acs_poverty['census_tract_key'].astype(str).str.zfill(6)
    )
    print(f"ACS poverty loaded from {acs_path}: {len(acs_poverty)} tracts")
else:
    api_key = os.environ.get('CENSUS_API_KEY')
    if api_key:
        try:
            from census import Census
            c = Census(api_key)
            data = c.acs5.state_county_tract(
                ('NAME', 'S1701_C03_001E'), state_fips='42', county_fips='101',
                tract='*', year=2022,
            )
            acs_poverty = pd.DataFrame(data).rename(columns={'S1701_C03_001E': 'poverty_rate'})
            acs_poverty['census_tract_key'] = (acs_poverty['state'] + acs_poverty['county']
                                                + acs_poverty['tract']).str[-6:]
            print(f"ACS poverty fetched via Census API: {len(acs_poverty)} tracts")
        except Exception as e:
            print(f"Census API call failed: {e}")
            acs_poverty = None
    else:
        print('No CENSUS_API_KEY set and no acs_poverty_phl.csv found — equity audit will be skipped.')
""")

    code("""\
if acs_poverty is not None:
    # Pad census_tract to 6 digits to align with the Census format (e.g. 153 → 015300)
    df_eq = df.copy()
    df_eq['census_tract_key'] = (
        df_eq['census_tract'].fillna(0).astype(int).mul(100).astype(str).str.zfill(6)
    )
    df_eq = df_eq.merge(acs_poverty[['census_tract_key', 'poverty_rate']],
                         on='census_tract_key', how='left')
    coverage = df_eq['poverty_rate'].notna().mean()
    print(f"Parcels with poverty_rate: {df_eq['poverty_rate'].notna().sum():,} "
          f"of {len(df_eq):,} ({coverage:.1%})")
else:
    df_eq = df.assign(poverty_rate=np.nan)
    coverage = 0
""")

    code("""\
if acs_poverty is not None and coverage > 0.5:
    df_eq['poverty_quintile'] = pd.qcut(
        df_eq['poverty_rate'], 5,
        labels=['Q1 (lowest)', 'Q2', 'Q3', 'Q4', 'Q5 (highest)'],
        duplicates='drop',
    )
    eq = (
        df_eq.dropna(subset=['poverty_quintile'])
        .groupby('poverty_quintile', observed=True)
        .apply(lambda g: pd.Series({
            'n_parcels':  len(g),
            'n_vacant':   int((g['ovs'] == 1).sum()),
            'obs_rate':   float((g['ovs'] == 1).mean()),
            'mean_pred':  float(g[prob_col].mean()),
            'auc':        safe_auc(g),
        }))
        .reset_index()
    )
    eq.round(4)
else:
    print('Equity audit skipped (insufficient ACS coverage)')
    eq = None
""")

    code("""\
if eq is not None:
    fig, ax = plt.subplots(figsize=(9, 4))
    qs = eq['poverty_quintile'].astype(str)
    x = np.arange(len(qs))
    ax.bar(x - 0.2, eq['obs_rate'],  width=0.4, label='Observed vacancy rate', color='#2166ac')
    ax.bar(x + 0.2, eq['mean_pred'], width=0.4, label='Mean predicted P(vacant)', color='#d6604d')
    ax.set_xticks(x)
    ax.set_xticklabels(qs)
    ax.set_ylabel('Rate')
    ax.set_title('Observed vs predicted vacancy rate by census-tract poverty quintile')
    ax.legend()
    plt.tight_layout()
    plt.savefig(GRAPH_PATH / 'equity_poverty_quintile.png', dpi=200, bbox_inches='tight')
    plt.show()
""")

    md("""## 7. Export""")

    code("""\
# Per-parcel predictions with ward + category metadata
export_cols = ['parcel_number', 'ovs', 'data_split',
               prob_col, 'prob_tier',
               'zip_code', 'geographic_ward', 'census_tract',
               'category_code', 'category_code_description']
df[export_cols].to_csv(OUT_PATH / 'predictions_04b.csv', index=False)
print(f"Exported predictions_04b.csv: {len(df):,} rows")

ward_summary.to_csv(OUT_PATH / 'ward_summary_04b.csv', index=False)
zip_auc.to_csv(    OUT_PATH / 'zip_auc_04b.csv',     index=False)
print('Exported ward_summary_04b.csv, zip_auc_04b.csv')

if eq is not None:
    eq.to_csv(OUT_PATH / 'equity_poverty_auc.csv', index=False)
    print('Exported equity_poverty_auc.csv')
""")

    md("""---
**Done.** Outputs in `data_py/`. The optional spatial map step (heavy — needs
PWD parcel polygons) lives in 05; this notebook focuses on tabular validation.
""")

    write_notebook(cells, '04b_vpi_comparison.ipynb')


# ==========================================================================
# 05_output_analysis.ipynb
# ==========================================================================
def build_05_output_analysis():
    cells, md, code = make_cells()

    md("""# PhillyStat360 — 05: Output Analysis (Python / Colab port)

Port of `code/05_output_analysis_v3.Rmd`. Stakeholder-facing analysis of the
final ensemble predictions:

- §2 — Distribution by OPA property category (tables + density plots)
- §3 — Spatial summary by ZIP code (bar chart + choropleth)
- §4 — Capacity-based threshold table — answers "if we can inspect N parcels,
   what threshold should we use?"
- §5 — GeoJSON export for dashboards (optional, heavy)

Uses the **ensemble** probability column from `data_py/all_predictions_rf.csv`
as the primary score throughout.
""")

    md("""## 0. Setup""")
    code(SETUP_DRIVE_MOUNT)
    code(SETUP_PATHS)
    code("""\
%pip install -q geopandas folium pygris
""")
    code(SETUP_IMPORTS_BASIC + """\

# Optional spatial libraries
try:
    import geopandas as gpd
    HAS_GPD = True
except ImportError:
    HAS_GPD = False
    print('geopandas missing — spatial steps will be skipped')

try:
    import folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False
    print('folium missing — interactive map will be skipped')
""")

    md("""## 1. Load predictions & metadata""")

    code("""\
preds = pd.read_csv(PY_PATH / 'all_predictions_rf.csv', low_memory=False)
preds['parcel_number'] = preds['parcel_number'].astype(str)
print(f"Predictions: {len(preds):,} rows")

features_df = pd.read_csv(DATA_PATH / 'features_residential.csv', low_memory=False)
features_df['parcel_number'] = features_df['parcel_number'].astype(str)

ovs_meta = pd.read_csv(DATA_PATH / 'ovs_residential.csv', low_memory=False)
ovs_meta['parcel_number'] = ovs_meta['parcel_number'].astype(str)
ovs_meta = ovs_meta[['parcel_number', 'category_code',
                     'category_code_description', 'zip_code',
                     'census_tract', 'zoning']].drop_duplicates(subset='parcel_number')

prob_col     = 'ensemble_prob'     if 'ensemble_prob'     in preds.columns else 'rf_prob'
prob_raw_col = 'ensemble_prob_raw' if 'ensemble_prob_raw' in preds.columns else 'rf_prob'
print(f"Primary probability column: {prob_col}  (raw: {prob_raw_col})")

analysis = (
    preds
    .merge(features_df[['parcel_number', 'exterior_condition', 'building_age',
                        'log_livable_area']],
           on='parcel_number', how='left')
    .merge(ovs_meta, on='parcel_number', how='left')
)
print(f"analysis_df: {len(analysis):,} parcels")
print(f"  OVS=1 (observed): {int((analysis['ovs'] == 1).sum()):,}")
print(f"  category_code_description present for "
      f"{analysis['category_code_description'].notna().sum():,} parcels")
""")

    md("""## 2. Distribution by OPA category

### 2a. Per-category table""")

    code("""\
type_summary = (
    analysis.dropna(subset=['category_code_description'])
    .groupby('category_code_description')
    .agg(n_parcels=('parcel_number', 'size'),
         n_observed_vac=('ovs', lambda s: int((s == 1).sum())),
         obs_rate=('ovs', lambda s: float((s == 1).mean())),
         mean_pred_prob=(prob_col, 'mean'),
         median_pred_prob=(prob_col, 'median'))
    .reset_index()
    .sort_values('mean_pred_prob', ascending=False)
)
type_summary.round(4)
""")

    md("""### 2b. Density plot by major categories""")

    code("""\
major_cats = type_summary[type_summary['n_parcels'] >= 500]['category_code_description'].tolist()
plot_df = analysis[analysis['category_code_description'].isin(major_cats)]

# Use raw probabilities (better spread for visualization than calibrated)
nonzero = plot_df[plot_df[prob_raw_col] > 0]
print(f"Plotting {len(nonzero):,} of {len(plot_df):,} parcels (raw prob > 0)")

fig, ax = plt.subplots(figsize=(11, 6))
for cat in major_cats:
    sub = nonzero[nonzero['category_code_description'] == cat]
    if len(sub) >= 20:
        sns.kdeplot(sub[prob_raw_col], ax=ax, label=cat, linewidth=1.5)
ax.set_xlim(0, 0.4)
ax.set_xlabel(f'Raw P(vacant) [{prob_raw_col}]')
ax.set_ylabel('Density')
ax.set_title('Probability distribution by OPA category (non-zero predictions)')
ax.legend(loc='upper right', fontsize=8)
plt.tight_layout()
plt.savefig(GRAPH_PATH / 'prob_distribution_by_category.png', dpi=200, bbox_inches='tight')
plt.show()
""")

    md("""### 2c. Mean P(vacant) and tier breakdown by category""")

    code("""\
mean_prob_cat = (
    analysis[analysis['category_code_description'].isin(major_cats)]
    .groupby('category_code_description')
    .agg(mean_prob=(prob_col, 'mean'),
         obs_rate=('ovs', lambda s: float((s == 1).mean())))
    .reset_index()
    .sort_values('mean_prob')
)

fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(mean_prob_cat))))
y = np.arange(len(mean_prob_cat))
ax.barh(y, mean_prob_cat['mean_prob'], color='#2166ac', alpha=0.85, label='Mean predicted')
ax.scatter(mean_prob_cat['obs_rate'], y, color='#d73027', s=80, zorder=5,
           marker='D', label='Observed OVS rate')
for i, v in enumerate(mean_prob_cat['mean_prob']):
    ax.text(v, i, f' {v:.2%}', va='center', fontsize=8)
ax.set_yticks(y)
ax.set_yticklabels(mean_prob_cat['category_code_description'])
ax.set_xlabel('Probability / Rate')
ax.set_title('Mean predicted P(vacant) by OPA category')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(GRAPH_PATH / 'mean_prob_by_category.png', dpi=200, bbox_inches='tight')
plt.show()
""")

    md("""### 2d. Category calibration plot""")

    code("""\
calib = type_summary[type_summary['n_parcels'] >= 200].copy()

fig, ax = plt.subplots(figsize=(8, 7))
ax.plot([0, calib['obs_rate'].max() * 1.2], [0, calib['obs_rate'].max() * 1.2],
        '--', color='gray', label='Perfect calibration')
ax.scatter(calib['mean_pred_prob'], calib['obs_rate'],
           s=calib['n_parcels'] / 200, alpha=0.6, color='steelblue',
           edgecolors='white')
for _, row in calib.iterrows():
    ax.annotate(row['category_code_description'][:25],
                (row['mean_pred_prob'], row['obs_rate']),
                fontsize=7, alpha=0.8, xytext=(4, 4), textcoords='offset points')
ax.set_xlabel('Mean predicted P(vacant)')
ax.set_ylabel('Observed vacancy rate')
ax.set_title('Category calibration (above diagonal = under-predicted)')
ax.legend()
plt.tight_layout()
plt.savefig(GRAPH_PATH / 'category_calibration.png', dpi=200, bbox_inches='tight')
plt.show()
""")

    md("""## 3. Spatial summary by ZIP""")

    code("""\
zip_summary = (
    analysis.dropna(subset=['zip_code'])
    .groupby('zip_code')
    .agg(n_parcels=('parcel_number', 'size'),
         n_obs_vacant=('ovs', lambda s: int((s == 1).sum())),
         obs_rate=('ovs', lambda s: float((s == 1).mean())),
         mean_pred_prob=(prob_col, 'mean'))
    .reset_index()
    .sort_values('mean_pred_prob', ascending=False)
)
zip_summary.head(20).round(4)
""")

    code("""\
sub = zip_summary[zip_summary['n_parcels'] >= 100].sort_values('mean_pred_prob')
fig, ax = plt.subplots(figsize=(10, max(7, 0.25 * len(sub))))
y = np.arange(len(sub))
colors = plt.cm.Blues(sub['mean_pred_prob'].rank(pct=True))
ax.barh(y, sub['mean_pred_prob'], color=colors, label='Mean P(vacant)')
ax.scatter(sub['obs_rate'], y, color='black', marker='D', s=40, zorder=5,
           label='Observed rate')
ax.set_yticks(y)
ax.set_yticklabels(sub['zip_code'].astype(str), fontsize=8)
ax.set_xlabel('Probability / Rate')
ax.set_title('Mean predicted P(vacant) by ZIP code (≥100 parcels)')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(GRAPH_PATH / 'zip_mean_prob_bar.png', dpi=200, bbox_inches='tight')
plt.show()
""")

    md("""### 3c. Choropleth map (optional)

Uses `pygris` (Census TIGER) to fetch ZIP boundaries for Philly. Skips
gracefully if `pygris` isn't installed or the API call fails.
""")

    code("""\
zcta_gdf = None
try:
    from pygris import zctas
    print('Fetching ZCTA boundaries via pygris (Census TIGER)…')
    z = zctas(year=2020, cb=True, cache=True)
    z.columns = [c.upper() for c in z.columns]
    zcta_col = 'ZCTA5CE20' if 'ZCTA5CE20' in z.columns else \\
               ('ZCTA5CE10' if 'ZCTA5CE10' in z.columns else 'GEOID20')
    z = z[z[zcta_col].astype(str).str.startswith('191')]
    z = z.merge(zip_summary.assign(zip_code=lambda d: d['zip_code'].astype(str)),
                left_on=zcta_col, right_on='zip_code', how='inner')
    zcta_gdf = z
    print(f"  {len(zcta_gdf)} ZIP polygons matched")
except Exception as e:
    print(f'pygris ZCTA fetch failed — choropleth will be skipped: {e}')
""")

    code("""\
if zcta_gdf is not None and len(zcta_gdf) > 0:
    fig, ax = plt.subplots(figsize=(9, 9))
    zcta_gdf.plot(column='mean_pred_prob', cmap='Blues', legend=True,
                  edgecolor='white', linewidth=0.4, ax=ax,
                  legend_kwds={'label': 'Mean P(vacant)', 'shrink': 0.6})
    ax.set_axis_off()
    ax.set_title('Mean predicted vacancy probability by ZIP code')
    plt.tight_layout()
    plt.savefig(GRAPH_PATH / 'spatial_zip_choropleth.png', dpi=200, bbox_inches='tight')
    plt.show()
""")

    md("""### 3d. Interactive Folium map (optional)""")

    code("""\
if zcta_gdf is not None and HAS_FOLIUM:
    map_data = zcta_gdf.to_crs(epsg=4326).copy()
    # Find the ZCTA column name (varies by tigris vintage)
    zcta_col = next((c for c in ['ZCTA5CE20', 'ZCTA5CE10', 'GEOID20', 'GEOID']
                     if c in map_data.columns), None)
    if zcta_col is None:
        print('Could not identify ZCTA column — folium map skipped.')
    else:
        # Standardize for folium: it needs a key column accessible via
        # feature.properties.<key>
        map_data['zcta_id'] = map_data[zcta_col].astype(str)

        m = folium.Map(location=[39.9526, -75.1652], zoom_start=11,
                       tiles='CartoDB positron')

        folium.Choropleth(
            geo_data=map_data.__geo_interface__,
            data=map_data[['zcta_id', 'mean_pred_prob']],
            columns=['zcta_id', 'mean_pred_prob'],
            key_on='feature.properties.zcta_id',
            fill_color='Blues', fill_opacity=0.7, line_opacity=0.3,
            legend_name='Mean P(vacant)',
        ).add_to(m)

        # Add hover tooltips with the actual numbers
        folium.GeoJson(
            map_data,
            style_function=lambda x: {'fillOpacity': 0, 'color': 'transparent'},
            tooltip=folium.GeoJsonTooltip(
                fields=['zcta_id', 'n_parcels', 'n_obs_vacant',
                        'obs_rate', 'mean_pred_prob'],
                aliases=['ZIP:', 'Parcels:', 'Observed vacant:',
                         'Observed rate:', 'Mean P(vacant):'],
                localize=True,
            ),
        ).add_to(m)

        out_html = OUT_PATH / 'vacancy_risk_map.html'
        m.save(str(out_html))
        print(f"Interactive map saved to {out_html}")
        m
""")

    md("""## 4. Capacity-based threshold table

Translates "we can inspect N properties" → recommended probability cutoff,
along with expected precision and recall.
""")

    code("""\
# Use raw ensemble probability so the threshold table covers the full 0–1 range.
# (Calibrated probs are squashed below ~0.6 so a "threshold = 0.5" lookup is
# meaningless on calibrated.)
labeled = analysis.dropna(subset=['ovs', prob_raw_col])
score = labeled[prob_raw_col].values
y     = labeled['ovs'].astype(int).values
print(f"Capacity table built on {len(labeled):,} labeled parcels  "
      f"({int((y == 1).sum()):,} positives)")

thresholds_seq = np.arange(0.01, 0.99, 0.01)
rows = []
for t in thresholds_seq:
    flagged = score >= t
    n_flag  = int(flagged.sum())
    if n_flag == 0:
        continue
    n_tp = int(((flagged) & (y == 1)).sum())
    rows.append({
        'threshold': round(float(t), 3),
        'n_flagged': n_flag,
        'precision': n_tp / n_flag,
        'recall':    n_tp / max(int((y == 1).sum()), 1),
    })
capacity_df = pd.DataFrame(rows)
capacity_df.head()
""")

    code("""\
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(capacity_df['n_flagged'], capacity_df['precision'], label='Precision',
        color='steelblue', lw=2)
ax.plot(capacity_df['n_flagged'], capacity_df['recall'],    label='Recall',
        color='tomato', lw=2)
ax.set_xscale('log')
ax.set_xlabel('Parcels flagged (inspection capacity, log scale)')
ax.set_ylabel('Rate')
ax.set_ylim(0, 1)
ax.set_title('Capacity-based threshold curve\\n'
             '(left = high threshold, fewer flags; right = low threshold, more flags)')
ax.legend()
plt.tight_layout()
plt.savefig(GRAPH_PATH / 'capacity_threshold_curve.png', dpi=200, bbox_inches='tight')
plt.show()
""")

    code("""\
capacity_targets = [100, 250, 500, 1000, 2000, 5000, 10000]
rows = []
for cap in capacity_targets:
    candidate = capacity_df[capacity_df['n_flagged'] >= cap]
    if len(candidate) == 0:
        rows.append({'target_capacity': cap, 'n_flagged': np.nan,
                     'threshold': np.nan, 'precision': np.nan, 'recall': np.nan})
        continue
    row = candidate.loc[candidate['n_flagged'].idxmin()].to_dict()
    row['target_capacity'] = cap
    rows.append(row)
capacity_lookup = pd.DataFrame(rows)[
    ['target_capacity', 'n_flagged', 'threshold', 'precision', 'recall']
]
capacity_lookup['precision'] = capacity_lookup['precision'].map(
    lambda v: f"{v:.1%}" if pd.notnull(v) else '—')
capacity_lookup['recall']    = capacity_lookup['recall'].map(
    lambda v: f"{v:.1%}" if pd.notnull(v) else '—')
capacity_lookup
""")

    md("""## 5. Export summaries""")

    code("""\
zip_summary.to_csv(OUT_PATH / 'output_zip_summary.csv', index=False)
type_summary.to_csv(OUT_PATH / 'output_category_summary.csv', index=False)
capacity_df.to_csv(OUT_PATH / 'capacity_threshold_curve.csv', index=False)
print('Exported output_zip_summary.csv, output_category_summary.csv, capacity_threshold_curve.csv')

print(f"\\n--- 05 Output Analysis Summary ---")
print(f"Parcels analyzed:        {len(analysis):,}")
print(f"ZIP codes covered:       {analysis['zip_code'].nunique(dropna=True)}")
print(f"Property categories:     {analysis['category_code_description'].nunique(dropna=True)}")
print(f"Mean ensemble P(vacant): {analysis[prob_col].mean():.4f}")
""")

    md("""## 6. GeoJSON export (optional — heavy)

Joins predictions onto PWD parcel polygons (`PWD_PARCELS.geojson`, ~420 MB)
and writes three GeoJSON files: full, flagged-only (top 1%), simplified.

> **Default: SKIP this section.** It needs ~30–60 GB of RAM and takes 15+
> minutes. Set `RUN_GEOJSON_EXPORT = True` only when you actually need the
> spatial export for the dashboard team.
""")

    code("""\
RUN_GEOJSON_EXPORT = False  # set True only when you want the heavy spatial export

if RUN_GEOJSON_EXPORT and HAS_GPD:
    pwd_path = RAW_PATH / 'PWD_PARCELS.geojson'
    if not pwd_path.exists():
        print(f"PWD_PARCELS.geojson not found at {pwd_path} — skipping export")
    else:
        print(f"Loading PWD_PARCELS.geojson ({pwd_path.stat().st_size / 1e6:.0f} MB)…")
        parcels_sf = gpd.read_file(pwd_path)
        parcels_sf.columns = [c.lower() for c in parcels_sf.columns]
        if 'brt_id' in parcels_sf.columns:
            parcels_sf = parcels_sf.rename(columns={'brt_id': 'parcel_number'})
        parcels_sf['parcel_number'] = parcels_sf['parcel_number'].astype(str)
        print(f"Loaded {len(parcels_sf):,} polygons")

        # Join
        export_cols = ['parcel_number', 'ovs', 'data_split',
                       prob_col, prob_raw_col, 'risk_score',
                       'qtile_tier', 'ensemble_flag',
                       'rf_prob', 'logit_prob', 'xgb_prob', 'lgb_prob',
                       'rf_flag', 'logit_flag', 'xgb_flag', 'lgb_flag']
        export_cols = [c for c in export_cols if c in analysis.columns]

        sf_out = parcels_sf[['parcel_number', 'geometry']].merge(
            analysis[export_cols], on='parcel_number', how='inner',
        ).to_crs(epsg=4326)

        full_path = OUT_PATH / 'vacancy_predictions.geojson'
        sf_out.to_file(full_path, driver='GeoJSON')
        print(f"Exported {full_path} ({len(sf_out):,} parcels, "
              f"{full_path.stat().st_size / 1e6:.0f} MB)")

        # Flagged-only
        if 'ensemble_flag' in sf_out.columns:
            flagged = sf_out[sf_out['ensemble_flag'] == 1]
            flagged_path = OUT_PATH / 'vacancy_predictions_flagged.geojson'
            flagged.to_file(flagged_path, driver='GeoJSON')
            print(f"Exported {flagged_path} ({len(flagged):,} parcels, "
                  f"{flagged_path.stat().st_size / 1e6:.0f} MB)")
else:
    print('GeoJSON export skipped (RUN_GEOJSON_EXPORT=False or geopandas missing).')
""")

    md("""---
**Done.** All summary CSVs written to `data_py/`. The optional choropleth and
interactive Folium map are in `graphs/python/` and `data_py/`. To produce the
heavy parcel-polygon GeoJSON, set `RUN_GEOJSON_EXPORT = True` in §6.
""")

    write_notebook(cells, '05_output_analysis.ipynb')


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
if __name__ == '__main__':
    build_04b_model_validation()
    build_04b_vpi_comparison()
    build_05_output_analysis()
