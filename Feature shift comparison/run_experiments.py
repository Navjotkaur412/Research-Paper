
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

DATASETS = {
    "BreastCancer": dict(         
        file="data.csv", target="diagnosis", positive="M", drop=["id"]),
    "PimaDiabetes": dict(         
        file="diabetes.csv", target="Outcome", positive="1",
        zero_as_nan=["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]),
    "AdultIncome": dict(          
        file="adult.csv", target="income", positive=">50K", drop=["fnlwgt"]),
    "CreditDefault": dict(        
        file="UCI_Credit_Card.csv", target="default.payment.next.month",
        positive="1", drop=["ID"], categorical=["SEX", "EDUCATION", "MARRIAGE"]),
}

FINE_DELTAS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
SUBSET_DELTAS = [0.5, 1.0, 2.0]
RANDOM_FRACS = [0.1, 0.25, 0.5]
TOPBOTTOM_FRAC = 0.25
TEST_SIZE = 0.3


def load_dataset(name, data_dir, max_rows):
    cfg = DATASETS[name]
    path = Path(data_dir) / cfg["file"]
    if not path.exists():  # search subfolders too (e.g. data/<unzipped-folder>/file.csv)
        hits = [p for p in Path(data_dir).rglob("*")
                if p.is_file() and p.name.lower() == cfg["file"].lower()]
        if not hits:
            found = sorted(str(p.relative_to(data_dir)) for p in Path(data_dir).rglob("*") if p.is_file())
            raise FileNotFoundError(f"'{cfg['file']}' not found anywhere under '{data_dir}'. "
                                    f"Files found: {found}. Fix the file name in DATASETS['{name}'].")
        path = hits[0]
    print(f"[{name}] using file: {path}")
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df = df.drop(columns=[c for c in cfg.get("drop", []) if c in df.columns])
    df = df.replace(r"^\s*\?\s*$", np.nan, regex=True)
    for c in cfg.get("zero_as_nan", []):
        df[c] = df[c].replace(0, np.nan)
    df = df.drop_duplicates().reset_index(drop=True)

    y = (df[cfg["target"]].astype(str).str.strip().str.rstrip(".") == cfg["positive"]).astype(int)
    X = df.drop(columns=[cfg["target"]])

    cat_cols = [c for c in X.columns
                if (not pd.api.types.is_numeric_dtype(X[c])) or c in cfg.get("categorical", [])]
    num_cols = [c for c in X.columns if c not in cat_cols]
    for c in cat_cols:
        X[c] = X[c].where(X[c].isna(), X[c].astype(str))
    X[num_cols] = X[num_cols].astype(float)

    if len(X) > max_rows:  # fixed stratified subsample so SVM stays fast
        X, _, y, _ = train_test_split(X, y, train_size=max_rows, stratify=y, random_state=0)
        X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    return X, y, num_cols, cat_cols


# --------------------------------------------------------------------------
# 2. MODELS / PREPROCESSING / SHIFTS
# --------------------------------------------------------------------------
def make_models(seed):
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=seed),
        "DecisionTree": DecisionTreeClassifier(random_state=seed),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1),
        "SVM_RBF": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=seed),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                 eval_metric="logloss", random_state=seed,
                                 n_jobs=1, verbosity=0),
    }


def make_preprocessor(num_cols, cat_cols):
    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    parts = [("num", num, num_cols)]
    if cat_cols:
        cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
        parts.append(("cat", cat, cat_cols))
    return ColumnTransformer(parts)


def n_select(frac, n):
    return max(1, int(round(frac * n)))


def informative_order(num_cols, X_tr, y_tr):
    """Numeric columns sorted from most to least informative (mutual information)."""
    Xn = X_tr[num_cols]
    mi = mutual_info_classif(Xn.fillna(Xn.median()), y_tr, random_state=0)
    return [num_cols[i] for i in np.argsort(mi)[::-1]]


def get_features(mode, frac, num_cols, perm, order):
    if mode == "all":
        return list(num_cols)
    k = n_select(frac, len(num_cols))
    if mode == "random":
        return perm[:k]          # nested random subsets (10% subset is inside the 25% subset...)
    if mode == "top":
        return order[:k]
    return order[-k:]            # bottom


def apply_shift(X_te, shift_type, delta, cols, mu, sd, signs):
    Xs = X_te.copy()
    if shift_type == "mean":
        Xs[cols] = Xs[cols] + delta * sd[cols]
    elif shift_type == "mean_signed":
        Xs[cols] = Xs[cols] + delta * sd[cols] * signs[cols]
    elif shift_type == "variance":
        Xs[cols] = mu[cols] + (1.0 + delta) * (Xs[cols] - mu[cols])
    return Xs


def score(model, X, y):
    pred = model.predict(X)
    s = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X)
    return dict(accuracy=accuracy_score(y, pred),
                precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred, zero_division=0),
                f1=f1_score(y, pred, zero_division=0),
                roc_auc=roc_auc_score(y, s),
                pred_pos_rate=float(np.mean(pred)))


def build_conditions():
    """(shift_type, feature_mode, fraction_of_numeric_features, delta)"""
    conds = [("none", "all", 1.0, 0.0)]
    conds += [("mean", "all", 1.0, d) for d in FINE_DELTAS]
    conds += [("mean_signed", "all", 1.0, d) for d in FINE_DELTAS]
    conds += [("variance", "all", 1.0, d) for d in FINE_DELTAS]
    conds += [("mean", "random", f, d) for f in RANDOM_FRACS for d in SUBSET_DELTAS]
    conds += [("mean", fs, TOPBOTTOM_FRAC, d) for fs in ("top", "bottom") for d in SUBSET_DELTAS]
    return conds


# --------------------------------------------------------------------------
# 3. ONE (dataset, seed) RUN
# --------------------------------------------------------------------------
def run_one(name, X, y, num_cols, cat_cols, seed, conds):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=seed)

    pre = make_preprocessor(num_cols, cat_cols)       # fit on TRAIN only
    Xtr = pre.fit_transform(X_tr)
    models = make_models(seed)
    for m in models.values():
        m.fit(Xtr, y_tr)

    mu = X_tr[num_cols].mean()
    sd = X_tr[num_cols].std().replace(0, 1).fillna(1)
    rng = np.random.default_rng(seed)
    perm = list(rng.permutation(num_cols))
    signs = pd.Series(rng.choice([-1.0, 1.0], size=len(num_cols)), index=num_cols)
    order = informative_order(num_cols, X_tr, y_tr)

    rows = []
    for stype, fs, frac, d in conds:
        cols = get_features(fs, frac, num_cols, perm, order)
        Xte = pre.transform(apply_shift(X_te, stype, d, cols, mu, sd, signs))
        for mname, m in models.items():
            rows.append(dict(dataset=name, seed=seed, model=mname, shift_type=stype,
                             feature_set=fs, frac=frac, delta=d, n_shifted=len(cols),
                             **score(m, Xte, y_te)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="results_v2")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--max-rows", type=int, default=5000)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    data, info = {}, []
    for n in args.datasets:
        X, y, num_cols, cat_cols = load_dataset(n, args.data_dir, args.max_rows)
        data[n] = (X, y, num_cols, cat_cols)
        info.append(dict(dataset=n, rows_used=len(X), numeric_features=len(num_cols),
                         categorical_features=len(cat_cols),
                         positive_rate=round(float(y.mean()), 4)))
        print(f"Loaded {n}: {X.shape}, numeric={len(num_cols)}, categorical={len(cat_cols)}, "
              f"positive rate={y.mean():.3f}")
    pd.DataFrame(info).to_csv(out / "dataset_summary.csv", index=False)

    hp = [dict(model=k, params=str(v.get_params())) for k, v in make_models(0).items()]
    pd.DataFrame(hp).to_csv(out / "hyperparameters.csv", index=False)

    conds = build_conditions()
    tasks = [(n, s) for n in args.datasets for s in range(args.seeds)]
    res = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(run_one)(n, *data[n], s, conds) for n, s in tasks)
    df = pd.DataFrame([r for chunk in res for r in chunk])
    df.to_csv(out / "results_raw.csv", index=False)
    print(f"\nSaved {len(df)} rows to {out/'results_raw.csv'}")


if __name__ == "__main__":
    main()
