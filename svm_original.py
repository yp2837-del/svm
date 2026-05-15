from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR, SVR

from annotated_features import (
    FEATURE_COLUMNS,
    FORWARD_HORIZON,
    TARGET_COLUMN,
    build_features,
    prediction_frame,
    training_frame,
)
from annotated_xgboost import (
    DEFAULT_TOP_K,
    EMBARGO_DAYS,
    VAL_DAYS,
    _time_splits,
    build_portfolio,
)

DATA_DIR = Path(__file__).parent / "data"
# LinearSVR on ~100k+ rows is very slow; subsample by default (still SVM, same features).
DEFAULT_TRAIN_SUBSAMPLE_LINEAR = 50_000


def rank_ic(y_true: np.ndarray, y_pred: np.ndarray, dates: np.ndarray) -> float:
    """Daily cross-sectional Spearman correlation, averaged over dates."""
    ics = []
    for d in np.unique(dates):
        mask = dates == d
        if mask.sum() < 20:
            continue
        rho, _ = spearmanr(y_true[mask], y_pred[mask])
        if not np.isnan(rho):
            ics.append(rho)
    return float(np.mean(ics)) if ics else float("nan")


def _subsample_train(
    train_df: pd.DataFrame,
    max_rows: int | None,
    random_state: int,
) -> pd.DataFrame:
    if max_rows is None or len(train_df) <= max_rows:
        return train_df
    return train_df.sample(n=max_rows, random_state=random_state)


def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,  # noqa: ARG001 — kept for API parity with annotated_xgboost
    *,
    kernel: str = "linear",
    c: float = 1.0,
    epsilon: float = 0.01,
    train_subsample: int | None = None,
    random_state: int = 42,
) -> Pipeline:
    """Fit StandardScaler + SVR on training rows only (val used only for IC reporting)."""
    if kernel == "linear" and train_subsample is None and len(train_df) > DEFAULT_TRAIN_SUBSAMPLE_LINEAR:
        train_subsample = DEFAULT_TRAIN_SUBSAMPLE_LINEAR

    fit_df = _subsample_train(train_df, train_subsample, random_state)
    if kernel == "linear":
        # 110k rows × max_iter=20000 can take 10+ minutes; 5k + subsample is usually enough.
        reg = LinearSVR(
            C=c,
            epsilon=epsilon,
            max_iter=5_000,
            tol=1e-3,
            random_state=random_state,
        )
    elif kernel == "rbf":
        reg = SVR(kernel="rbf", C=c, epsilon=epsilon, gamma="scale")
    else:
        raise ValueError(f"unsupported kernel: {kernel!r} (use 'linear' or 'rbf')")

    pipe = Pipeline([("scaler", StandardScaler()), ("svr", reg)])
    if len(train_df) > len(fit_df):
        print(
            f"   fitting SVM on {len(fit_df):,} rows (subsampled from {len(train_df):,}) …",
            flush=True,
        )
    else:
        print(f"   fitting SVM on {len(fit_df):,} rows …", flush=True)
    t0 = time.perf_counter()
    pipe.fit(fit_df[FEATURE_COLUMNS], fit_df[TARGET_COLUMN])
    print(f"   fit done in {time.perf_counter() - t0:.1f}s", flush=True)
    return pipe


def _normalize_pred_df(pred_df: pd.DataFrame) -> pd.DataFrame:
    if "stock_code" not in pred_df.columns:
        if pred_df.index.name == "stock_code":
            pred_df = pred_df.reset_index()
        elif isinstance(pred_df.index, pd.MultiIndex) and "stock_code" in pred_df.index.names:
            pred_df = pred_df.reset_index()
        elif "code" in pred_df.columns:
            pred_df = pred_df.rename(columns={"code": "stock_code"})
        else:
            raise RuntimeError(
                "prediction_frame() returned no 'stock_code'. "
                f"columns={list(pred_df.columns)}, index_names={pred_df.index.names}"
            )
    return pred_df


def main():
    p = argparse.ArgumentParser(description="CSI500 portfolio via annotated features + SVM")
    p.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    p.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    p.add_argument("--as-of", default=None, help="YYYYMMDD; defaults to latest date in data")
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--test-days", type=int, default=20)
    p.add_argument("--kernel", choices=("linear", "rbf"), default="linear")
    p.add_argument("--c", type=float, default=1.0, help="SVM regularization parameter C")
    p.add_argument("--epsilon", type=float, default=0.01, help="epsilon-tube width")
    p.add_argument(
        "--train-subsample",
        type=int,
        default=None,
        help=(
            f"max training rows (linear default: {DEFAULT_TRAIN_SUBSAMPLE_LINEAR} when train is larger; "
            "rbf default: 40000). Use 0 to disable subsampling."
        ),
    )
    p.add_argument("--random-state", type=int, default=42)
    args = p.parse_args()

    if args.train_subsample == 0:
        args.train_subsample = None
    if args.kernel == "rbf" and args.train_subsample is None:
        print(">> Note: RBF SVR is slow on full data; using train_subsample=40000 by default.")
        args.train_subsample = 40_000

    def _log(msg: str) -> None:
        print(msg, flush=True)

    _log(f">> Loading {args.prices}")
    prices = pd.read_parquet(args.prices)
    _log(
        f"   {len(prices):,} rows, {prices['stock_code'].nunique()} stocks, "
        f"dates {prices['date'].min().date()} to {prices['date'].max().date()}"
    )

    index_path = Path(args.index)
    if not index_path.exists():
        raise FileNotFoundError(
            f"Index file not found: {index_path}. Run download_data.py first."
        )
    index_df = pd.read_parquet(index_path)
    _log(f">> Loading {index_path} ({len(index_df):,} rows)")

    _log(">> Building features (annotated_features) — usually ~5–30s …")
    t_feat = time.perf_counter()
    panel = build_features(prices, index_df=index_df)
    _log(f"   features built in {time.perf_counter() - t_feat:.1f}s")

    as_of_ts = pd.Timestamp(args.as_of) if args.as_of else panel["date"].max()
    trading_dates = np.sort(panel["date"].unique())
    as_of_idx = int(np.searchsorted(trading_dates, np.datetime64(as_of_ts)))
    cutoff_idx = max(0, as_of_idx - FORWARD_HORIZON)
    train_cutoff = pd.Timestamp(trading_dates[cutoff_idx])
    train_pool = training_frame(panel, max_date=train_cutoff)

    all_dates = np.sort(train_pool["date"].unique())
    min_need = VAL_DAYS + EMBARGO_DAYS + 20 + (
        args.test_days + EMBARGO_DAYS if args.test_days > 0 else 0
    )
    if len(all_dates) < min_need:
        raise RuntimeError("Not enough dates to train; download more history or reduce --test-days.")

    train_end, val_start, val_end, test_start, test_end = _time_splits(
        all_dates, VAL_DAYS, EMBARGO_DAYS, args.test_days,
    )
    train_df = train_pool[train_pool["date"] <= train_end]
    val_df = train_pool[(train_pool["date"] >= val_start) & (train_pool["date"] <= val_end)]
    _log(f"   train: {len(train_df):,} rows up to {train_end.date()}")
    _log(f"   embargo: {EMBARGO_DAYS} trading days (discarded) before val")
    _log(f"   val:   {len(val_df):,} rows {val_start.date()} to {val_end.date()}")
    if args.test_days > 0:
        _log(f"   embargo: {EMBARGO_DAYS} trading days (discarded) before test")
        test_df = train_pool[
            (train_pool["date"] >= test_start) & (train_pool["date"] <= test_end)
        ]
        _log(f"   test:  {len(test_df):,} rows {test_start.date()} to {test_end.date()} (held out)")
    else:
        test_df = pd.DataFrame()

    _log(f">> Training SVM (kernel={args.kernel}, C={args.c})")
    model = train_model(
        train_df,
        val_df,
        kernel=args.kernel,
        c=args.c,
        epsilon=args.epsilon,
        train_subsample=args.train_subsample,
        random_state=args.random_state,
    )

    val_pred = model.predict(val_df[FEATURE_COLUMNS])
    ic = rank_ic(val_df[TARGET_COLUMN].to_numpy(), val_pred, val_df["date"].to_numpy())
    _log(f"   validation rank IC: {ic:.4f}")

    if args.test_days > 0 and not test_df.empty:
        test_pred = model.predict(test_df[FEATURE_COLUMNS])
        ic_test = rank_ic(
            test_df[TARGET_COLUMN].to_numpy(), test_pred, test_df["date"].to_numpy(),
        )
        _log(f"   test rank IC:       {ic_test:.4f}")

    _log(">> Predicting portfolio")
    pred_df = prediction_frame(panel, as_of=args.as_of)
    if pred_df.empty:
        raise RuntimeError(f"No rows available for as_of={args.as_of}. Check data.")
    pred_df = _normalize_pred_df(pred_df)
    pred_date = pred_df["date"].iloc[0]
    _log(f"   as of {pred_date.date()}, scoring {len(pred_df)} stocks")

    pred_df = pred_df.assign(score=model.predict(pred_df[FEATURE_COLUMNS]))
    scores = pred_df.set_index("stock_code")["score"]
    weights = build_portfolio(scores, top_k=args.top_k)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"stock_code": weights.index, "weight": weights.values})
    out.to_csv(out_path, index=False)
    _log(f">> Wrote {len(out)} names to {out_path}")
    _log(
        f"   weight summary: min={out['weight'].min():.4f} "
        f"max={out['weight'].max():.4f} sum={out['weight'].sum():.4f}"
    )


if __name__ == "__main__":
    main()
