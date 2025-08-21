#!/usr/bin/env python3
"""
step1.py
Final end-to-end pipeline for Taiwan credit default dataset (.xls/.xlsx)

What it does:
 - Load Excel
 - EDA (summary, histograms, correlations)
 - Feature engineering
 - Preprocessing (numeric impute+scale, categorical impute+onehot)
 - SMOTE on transformed features
 - Train RandomForest (GridSearchCV) + LogisticRegression baseline
 - Calibrate and threshold-tune (sklearn compatibility handled)
 - SHAP explainability (optional)
 - Fairness checks (SEX, age buckets) — index-safe
 - Save artifacts and model card in outputs/
"""

import warnings, inspect, json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, precision_score, recall_score, f1_score
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
import joblib

# Optional SHAP
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

# ---------------- USER CONFIG ----------------
DATA_PATH = r"C:\Users\saadk\Desktop\Codealpha project\default of credit card clients.xls"
OUTPUT_DIR = Path("outputs"); OUTPUT_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42
TEST_SIZE = 0.20
VALID_SIZE = 0.20
CV_FOLDS = 4
RF_PARAM_GRID = {"n_estimators": [100, 200], "max_depth": [8, 12, None]}
TARGET_CANDIDATES = ["default payment next month", "default.payment.next.month", "Y", "target"]
# ---------------------------------------------

# ---------- Helpers ----------
def load_excel(path: str, header_row: int = 1) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"DATA_PATH not found: {p}")
    suf = p.suffix.lower()
    if suf == ".xls":
        try:
            import xlrd  # noqa
        except Exception:
            raise ImportError("Install xlrd to read .xls files: pip install xlrd")
        df = pd.read_excel(p, engine="xlrd", header=header_row)
    elif suf == ".xlsx":
        try:
            import openpyxl  # noqa
        except Exception:
            raise ImportError("Install openpyxl to read .xlsx files: pip install openpyxl")
        df = pd.read_excel(p, engine="openpyxl", header=header_row)
    else:
        raise ValueError("DATA_PATH must be .xls or .xlsx")
    df.columns = [str(c).strip() for c in df.columns]
    return df

def standardize_columns_and_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = list(df.columns)
    # map generic X1..Y names to UCI order if needed
    if (len(cols) >= 24 and any(str(c).upper().startswith("X") for c in cols[:3])) or ("Y" in cols):
        standard = [
            'ID','LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE',
            'PAY_1','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6',
            'BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6',
            'PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6','Y'
        ]
        mapping = {}
        for i, c in enumerate(cols):
            if i < len(standard):
                mapping[c] = standard[i]
        df = df.rename(columns=mapping)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    # detect target
    target_col = None
    for cand in TARGET_CANDIDATES:
        if cand in df.columns:
            target_col = cand
            break
    if target_col is None:
        target_col = df.columns[-1]
        print(f"[WARN] Falling back to last column as target: {target_col}")
    df = df.rename(columns={target_col: 'target'})
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])
    df['target'] = pd.to_numeric(df['target'], errors='coerce').fillna(0).astype(int)
    return df

def run_eda(df: pd.DataFrame):
    print("Running EDA...")
    # basics
    summary = {
        "shape": df.shape,
        "dtypes": df.dtypes.apply(lambda x: x.name).to_dict(),
        "missing_values": df.isna().sum().sort_values(ascending=False).to_dict(),
        "class_balance": df['target'].value_counts(normalize=True).to_dict()
    }
    df.head(10).to_csv(OUTPUT_DIR / "eda_head.csv", index=False)
    df.describe(include='all').to_csv(OUTPUT_DIR / "eda_describe.csv")
    # hist examples
    for col in ['LIMIT_BAL','AGE','BILL_AMT1','PAY_AMT1']:
        if col in df.columns:
            plt.figure(figsize=(6,4)); sns.histplot(df[col].dropna(), bins=50); plt.title(col)
            plt.tight_layout(); plt.savefig(OUTPUT_DIR / f"hist_{col}.png"); plt.close()
    # correlation heatmap
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] > 1:
        plt.figure(figsize=(10,8)); sns.heatmap(num.corr(), center=0, cmap='coolwarm'); plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "corr_heatmap.png"); plt.close()
    with open(OUTPUT_DIR / "eda_summary.json", "w") as f:
        json.dump(summary, f, default=str, indent=2)
    print("EDA saved to outputs/")

def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # convert non-target cols to numeric
    for c in df.columns:
        if c != 'target':
            df[c] = pd.to_numeric(df[c], errors='coerce')
    bill_cols = [c for c in df.columns if str(c).upper().startswith('BILL_AMT')]
    payamt_cols = [c for c in df.columns if str(c).upper().startswith('PAY_AMT')]
    pay_cols = [c for c in df.columns if str(c).upper().startswith('PAY_') and not str(c).upper().startswith('PAY_AMT')]
    if bill_cols:
        df['bill_sum'] = df[bill_cols].sum(axis=1); df['bill_mean'] = df[bill_cols].mean(axis=1)
    else:
        df['bill_sum'] = 0; df['bill_mean'] = 0
    if payamt_cols:
        df['pay_sum'] = df[payamt_cols].sum(axis=1); df['pay_mean'] = df[payamt_cols].mean(axis=1)
    else:
        df['pay_sum'] = 0; df['pay_mean'] = 0
    df['payment_rate'] = df['pay_sum'] / df['bill_sum'].replace({0: np.nan}); df['payment_rate'] = df['payment_rate'].fillna(0)
    if 'LIMIT_BAL' in df.columns:
        df['debt_ratio'] = df['bill_sum'] / df['LIMIT_BAL'].replace({0: np.nan}); df['debt_ratio'] = df['debt_ratio'].fillna(0)
    else:
        df['debt_ratio'] = 0
    if pay_cols:
        df['num_delinq_pos'] = (df[pay_cols] > 0).sum(axis=1); df['max_delinq'] = df[pay_cols].max(axis=1)
    else:
        df['num_delinq_pos'] = 0; df['max_delinq'] = 0
    if 'AGE' in df.columns:
        df['age_bucket'] = pd.cut(df['AGE'], bins=[0,25,35,50,200], labels=['<25','25-35','35-50','50+'])
    return df

def build_preprocessor(X: pd.DataFrame):
    categorical_cols = [c for c in ['SEX','EDUCATION','MARRIAGE'] if c in X.columns]
    numeric_cols = [c for c in X.select_dtypes(include=[np.number]).columns.tolist() if c not in categorical_cols and c != 'target']
    numeric_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    # OneHotEncoder compat (sparse_output vs sparse)
    ohe_kwargs = {}
    sig = inspect.signature(OneHotEncoder).parameters
    if 'sparse_output' in sig:
        ohe_kwargs['sparse_output'] = False
    elif 'sparse' in sig:
        ohe_kwargs['sparse'] = False
    categorical_transformer = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', **ohe_kwargs))])
    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_cols), ('cat', categorical_transformer, categorical_cols)], remainder='drop')
    return preprocessor, numeric_cols, categorical_cols

def get_feature_names(col_transformer: ColumnTransformer, numeric_cols, categorical_cols):
    try:
        names = col_transformer.get_feature_names_out()
        names = [str(n).split('__')[-1] for n in names]
        return list(names)
    except Exception:
        names = []
        names.extend(numeric_cols)
        try:
            cat_trans = col_transformer.named_transformers_['cat']
            ohe = cat_trans.named_steps['onehot']
            cats = ohe.categories_
            for col, cats_i in zip(categorical_cols, cats):
                for cat in cats_i:
                    names.append(f"{col}_{cat}")
        except Exception:
            names.extend(categorical_cols)
        return names

def save_monitoring_stats(X_train: pd.DataFrame):
    stats = {}
    for col in X_train.columns:
        s = X_train[col].dropna()
        if s.dtype.kind in 'bifc':
            stats[col] = {'mean': float(s.mean()), 'std': float(s.std()), 'min': float(s.min()), '25%': float(s.quantile(0.25)), '50%': float(s.median()), '75%': float(s.quantile(0.75)), 'max': float(s.max()), 'n': int(len(s))}
        else:
            stats[col] = {'top_counts': s.astype(str).value_counts().head(10).to_dict(), 'n': int(len(s))}
    with open(OUTPUT_DIR / "monitoring_stats.json", "w") as f:
        json.dump({'created_at': datetime.utcnow().isoformat(), 'stats': stats}, f, indent=2)
    print("Saved monitoring stats.")

# --------- Fairness check (index-safe) ----------
def fairness_checks(df_subset: pd.DataFrame, y_true, y_proba, threshold: float = 0.5):
    """
    df_subset: features corresponding to test set (any index) -> will be reset inside
    y_true: true labels (array/series) corresponding to test rows
    y_proba: predicted probabilities corresponding to test rows
    """
    df_local = df_subset.reset_index(drop=True)
    y_true_local = pd.Series(y_true).reset_index(drop=True)
    y_proba_local = pd.Series(y_proba).reset_index(drop=True)
    y_pred_local = (y_proba_local >= threshold).astype(int)

    # age buckets (ensure exist)
    if 'AGE' in df_local.columns:
        df_local['age_bucket'] = pd.cut(df_local['AGE'], bins=[0,25,35,50,200], labels=['<25','25-35','35-50','50+'])
    checks = {}
    if 'SEX' in df_local.columns:
        sex_stats = {}
        for g in sorted(df_local['SEX'].dropna().unique()):
            mask = df_local['SEX'] == g
            if mask.sum() < 10:
                continue
            rep = classification_report(y_true_local[mask], y_pred_local[mask], output_dict=True, zero_division=0)
            sex_stats[str(g)] = {'support': int(mask.sum()), 'precision_pos': rep.get('1',{}).get('precision'), 'recall_pos': rep.get('1',{}).get('recall'), 'f1_pos': rep.get('1',{}).get('f1-score')}
        checks['SEX'] = sex_stats
    if 'age_bucket' in df_local.columns:
        age_stats = {}
        for g in df_local['age_bucket'].cat.categories:
            mask = df_local['age_bucket'] == g
            if mask.sum() < 10:
                continue
            rep = classification_report(y_true_local[mask], y_pred_local[mask], output_dict=True, zero_division=0)
            age_stats[str(g)] = {'support': int(mask.sum()), 'precision_pos': rep.get('1',{}).get('precision'), 'recall_pos': rep.get('1',{}).get('recall'), 'f1_pos': rep.get('1',{}).get('f1-score')}
        checks['age_bucket'] = age_stats
    with open(OUTPUT_DIR / "fairness_checks.json", "w") as f:
        json.dump(checks, f, indent=2, default=str)
    print("Saved fairness checks.")

# ---------- Calibration & threshold tuning ----------
def calibrate_and_tune(best_clf, X_val_trans, y_val, prefer_metric='f1'):
    sig = inspect.signature(CalibratedClassifierCV).parameters
    if 'estimator' in sig:
        calib = CalibratedClassifierCV(estimator=best_clf, cv='prefit', method='sigmoid')
    else:
        calib = CalibratedClassifierCV(base_estimator=best_clf, cv='prefit', method='sigmoid')
    calib.fit(X_val_trans, y_val)
    y_val_proba = calib.predict_proba(X_val_trans)[:,1]
    thresholds = np.linspace(0.01, 0.99, 99)
    best_thr, best_score, best_metrics = 0.5, -1, None
    for thr in thresholds:
        y_pred = (y_val_proba >= thr).astype(int)
        if prefer_metric == 'f1':
            score = f1_score(y_val, y_pred, zero_division=0)
        else:
            score = f1_score(y_val, y_pred, zero_division=0)
        if score > best_score:
            best_score = score; best_thr = thr
            best_metrics = {'precision': float(precision_score(y_val, y_pred, zero_division=0)), 'recall': float(recall_score(y_val, y_pred, zero_division=0)), 'f1': float(f1_score(y_val, y_pred, zero_division=0))}
    # calibration curve (optional)
    try:
        from sklearn.calibration import calibration_curve
        prob_true, prob_pred = calibration_curve(y_val, y_val_proba, n_bins=10)
        plt.figure(figsize=(6,6)); plt.plot(prob_pred, prob_true, marker='o'); plt.plot([0,1],[0,1],'k--'); plt.xlabel('Pred'); plt.ylabel('True'); plt.title('Calibration (val)'); plt.tight_layout(); plt.savefig(OUTPUT_DIR / "calibration_curve.png"); plt.close()
    except Exception:
        pass
    return calib, best_thr, best_metrics, y_val_proba

# ---------- SHAP (optional) ----------
def shap_explain(best_rf, X_sample_df, feat_names):
    if not HAS_SHAP:
        print("SHAP not installed; skipping SHAP explainability.")
        return
    try:
        explainer = shap.TreeExplainer(best_rf)
        shap_vals = explainer.shap_values(X_sample_df)
        if isinstance(shap_vals, list) and len(shap_vals) >= 2:
            shap.summary_plot(shap_vals[1], X_sample_df, show=False); plt.tight_layout(); plt.savefig(OUTPUT_DIR / "shap_summary_pos.png"); plt.close()
        else:
            shap.summary_plot(shap_vals, X_sample_df, show=False); plt.tight_layout(); plt.savefig(OUTPUT_DIR / "shap_summary.png"); plt.close()
        print("Saved SHAP plots.")
    except Exception as e:
        print("SHAP failed:", e)

# ---------- Model card ----------
def write_model_card(metadata: dict, results: list, path: Path = OUTPUT_DIR / "model_card.txt"):
    lines = []
    lines.append("Model card - Taiwan Credit Default")
    lines.append(f"Created: {datetime.utcnow().isoformat()} UTC")
    lines.append("\nDataset:")
    lines.append(f"- path: {metadata.get('data_path')}")
    lines.append(f"- rows: {metadata.get('n_rows')}, cols: {metadata.get('n_cols')}")
    lines.append("\nSample features:")
    for f in metadata.get('features_sample', [])[:100]:
        lines.append(f"- {f}")
    lines.append("\nEvaluation:")
    for r in results:
        lines.append(f"- {r['model']}: ROC-AUC={r.get('roc_auc')}, precision={r.get('precision')}, recall={r.get('recall')}, f1={r.get('f1')}")
    lines.append("\nLimitations:\n- Historic dataset; monitor drift.\n- SMOTE used during training.")
    path.write_text("\n".join(lines))
    print("Model card saved.")

# ---------------------- Main pipeline ----------------------------
def main():
    print("Loading dataset...")
    df_raw = load_excel(DATA_PATH)
    df = standardize_columns_and_target(df_raw)
    print("Loaded shape:", df.shape)
    df.to_csv(OUTPUT_DIR / "raw_loaded.csv", index=False)

    # EDA
    run_eda(df)

    # Feature engineering
    df = feature_engineer(df)
    df.to_csv(OUTPUT_DIR / "engineered.csv", index=False)

    # Prepare X,y
    y = df['target'].astype(int)
    X = df.drop(columns=['target'])

    # Splits
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=VALID_SIZE, stratify=y_train_full, random_state=RANDOM_STATE)

    # Monitoring stats on train
    save_monitoring_stats(X_train)

    # Preprocessor
    preproc, numeric_cols, categorical_cols = build_preprocessor(X_train)
    preproc.fit(X_train)

    feat_names = get_feature_names(preproc, numeric_cols, categorical_cols)
    print("Transformed feature count:", len(feat_names))

    # Transform sets (to dense arrays)
    X_train_trans = preproc.transform(X_train)
    X_val_trans = preproc.transform(X_val)
    X_test_trans = preproc.transform(X_test)
    if hasattr(X_train_trans, "toarray"):
        X_train_trans = X_train_trans.toarray()
        X_val_trans = X_val_trans.toarray()
        X_test_trans = X_test_trans.toarray()

    # SMOTE on transformed training data
    sm = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = sm.fit_resample(X_train_trans, y_train)
    print("After SMOTE:", X_train_res.shape, " positives:", int(sum(y_train_res)))

    # Grid search RandomForest
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(rf, RF_PARAM_GRID, scoring='roc_auc', cv=cv, n_jobs=-1, verbose=1)
    print("Running GridSearchCV for RandomForest...")
    gs.fit(X_train_res, y_train_res)
    best_rf = gs.best_estimator_
    print("Best RF params:", gs.best_params_)

    # Logistic baseline
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_res, y_train_res)

    # Calibrate and tune threshold on validation set
    try:
        calib_clf, best_thr, best_metrics, y_val_proba = calibrate_and_tune(best_rf, X_val_trans, y_val, prefer_metric='f1')
        print("Calibration done. Best threshold:", best_thr, "metrics:", best_metrics)
    except Exception as e:
        print("Calibration failed, using uncalibrated RF with default threshold 0.5:", e)
        best_rf.fit(X_train_res, y_train_res)
        calib_clf = best_rf
        best_thr = 0.5
        y_val_proba = best_rf.predict_proba(X_val_trans)[:,1]

    # Evaluate on test
    if hasattr(calib_clf, 'predict_proba'):
        y_test_proba = calib_clf.predict_proba(X_test_trans)[:,1]
    else:
        y_test_proba = best_rf.predict_proba(X_test_trans)[:,1]
    y_test_pred = (y_test_proba >= best_thr).astype(int)

    print("\nTest classification report:\n", classification_report(y_test, y_test_pred, zero_division=0))
    roc = roc_auc_score(y_test, y_test_proba)
    print("Test ROC AUC:", roc)

    # Save results
    results = [{'model': 'RandomForest (calibrated)' if hasattr(calib_clf, 'predict_proba') else 'RandomForest', 'roc_auc': float(roc), 'precision': float(precision_score(y_test, y_test_pred, zero_division=0)), 'recall': float(recall_score(y_test, y_test_pred, zero_division=0)), 'f1': float(f1_score(y_test, y_test_pred, zero_division=0))}]
    pd.DataFrame(results).to_csv(OUTPUT_DIR / "model_results.csv", index=False)

    # ROC plot
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    plt.figure(figsize=(6,5)); plt.plot(fpr, tpr, label=f"AUC={roc:.3f}"); plt.plot([0,1],[0,1],'k--'); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC (test)"); plt.legend(); plt.tight_layout(); plt.savefig(OUTPUT_DIR / "roc_curve.png"); plt.close()

    # Save pipeline (preprocessor + calibrated classifier)
    if hasattr(calib_clf, 'predict_proba'):
        final_pipeline = Pipeline([('pre', preproc), ('clf', calib_clf)])
    else:
        final_pipeline = Pipeline([('pre', preproc), ('clf', best_rf)])
    joblib.dump(final_pipeline, OUTPUT_DIR / "best_model.joblib")
    print("Saved final pipeline to outputs/best_model.joblib")

    # SHAP explainability (optional)
    if HAS_SHAP:
        try:
            sample_n = min(1000, X_test_trans.shape[0])
            X_shap_df = pd.DataFrame(X_test_trans[:sample_n], columns=feat_names)
            shap_explain(best_rf, X_shap_df, feat_names)
        except Exception as e:
            print("SHAP explainability error:", e)
    else:
        print("SHAP not installed; to enable, run: pip install shap")

    # Fairness checks - pass test features & aligned arrays (function resets indices internally)
    fairness_checks(X_test, y_test, y_test_proba, threshold=best_thr)

    # Save monitoring stats (train)
    save_monitoring_stats(X_train)

    # Write model card
    metadata = {'data_path': str(DATA_PATH), 'n_rows': int(df.shape[0]), 'n_cols': int(df.shape[1]), 'features_sample': feat_names}
    write_model_card = globals()['write_model_card'] if 'write_model_card' in globals() else None
    if write_model_card is None:
        # small inline writer if not present
        with open(OUTPUT_DIR / "model_card.txt", "w") as f:
            f.write(f"Model created: {datetime.utcnow().isoformat()}\nRows: {df.shape[0]}\nCols: {df.shape[1]}\n")
    else:
        write_model_card(metadata, results, path=OUTPUT_DIR / "model_card.txt")

    print("Done. All artifacts are in the 'outputs' folder.")

if __name__ == "__main__":
    main()
