
"""
Equity Factor Pipeline: end‑to‑end functions to load prices, join Fama‑French
factors, run FF5 regressions, and build trading strategy summaries.

Quick start
-----------

from equity_factor_pipeline import run_analysis

result = run_analysis(
    ticker="NVDA",
    ff5_path="F-F_Research_Data_5_Factors_2x3_daily.csv",
    mom_path="F-F_Momentum_Factor_daily.csv",
    start="2010-01-01",
    end=None,
)

print(result["stats"])            # AnnRet/AnnVol/Sharpe table for strategies
print(result["contrib_table"])    # Monthly + annualized (~x12) contribution
print(result["params"])           # OLS params (alpha + betas)
# result["summary_text"]          # Full statsmodels summary as text
# result["curves"].plot()         # Equity curves (requires matplotlib)

Notes
-----
- Factors are expected in DAILY frequency and PERCENT in the raw CSV/TXT; this
  loader converts them to DECIMALS and DateTimeIndex.
- Uses Yahoo Finance via yfinance for prices.
- All joins are on calendar dates; we restrict to common intersections.
- Strategies are illustrative and operate on monthly excess returns.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import re
import statsmodels.api as sm

# Optional dependency used in load; keep import local in function to avoid hard req
# import yfinance as yf

# =============================================================================
# Utilities
# =============================================================================

def _ensure_simple(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DatetimeIndex and sensible, flat column names.

    Special handling for yfinance MultiIndex columns:
    - If columns are MultiIndex like (Field, Ticker),
      * If there's only one unique ticker, keep Field names (e.g., 'Close').
      * Else join as 'Field_Ticker' (e.g., 'Close_NVDA').
    """
    df = df.copy()
    # Index -> Datetime
    if isinstance(df.index, pd.MultiIndex):
        df.index = df.index.get_level_values(-1)
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"

    # Columns -> flatten smartly
    if getattr(df, "columns", None) is not None and getattr(df.columns, "nlevels", 1) > 1:
        lvl0 = df.columns.get_level_values(0).astype(str)
        lvl1 = df.columns.get_level_values(1).astype(str)
        uniq0 = pd.unique(lvl0)
        uniq1 = pd.unique(lvl1)
        if len(uniq1) == 1:  # single ticker -> keep price field names
            df.columns = lvl0
        elif len(uniq0) == 1:  # single field -> keep ticker names
            df.columns = lvl1
        else:  # multiple tickers & fields
            df.columns = [f"{a}_{b}" for a, b in zip(lvl0, lvl1)]
    else:
        # If columns are tuples (rare), fall back to last element
        df.columns = [c if not isinstance(c, tuple) else c[-1] for c in df.columns]

    df.columns = [str(c) for c in df.columns]
    return df


def pick_price_column(df: pd.DataFrame) -> str:
    """Pick a reasonable price column from OHLCV tables.

    Preference order: 'Adj Close', 'Close'; otherwise heuristic fallbacks.
    Works with flattened names like 'Close_NVDA' as well.
    """
    candidates = [
        "Adj Close", "AdjClose", "adj close", "adj_close",
        "Close", "close",
    ]
    # direct hits
    for c in candidates:
        if c in df.columns:
            return c
    # pattern hits (e.g., 'Close_NVDA')
    for c in df.columns:
        s = str(c).lower().replace(" ", "")
        if s.startswith("adjclose") or s.startswith("close"):
            return c
    # single-column DF
    if df.shape[1] == 1:
        return df.columns[0]
    raise KeyError("No suitable price column found. Consider passing price_col explicitly.")


def compound_return(s: pd.Series) -> float:
    """Compound returns over a period: prod(1+r) - 1, ignoring NaNs."""
    s = pd.to_numeric(s, errors="coerce").dropna()
    return (1.0 + s).prod() - 1.0 if not s.empty else np.nan


def cum_index(r: pd.Series) -> pd.Series:
    """Cumulative index from returns (start = 1)."""
    return (1.0 + r.fillna(0.0)).cumprod()


def perf_summary(r: pd.Series, name: str = "strategy") -> pd.Series:
    r = r.dropna()
    if r.empty:
        return pd.Series({"AnnRet": np.nan, "AnnVol": np.nan, "Sharpe": np.nan}, name=name)
    ann_ret = (1 + r).prod() ** (12 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(12)
    sharpe = (r.mean() * 12) / ann_vol if ann_vol > 0 else np.nan
    return pd.Series({"AnnRet": ann_ret, "AnnVol": ann_vol, "Sharpe": sharpe}, name=name)


# =============================================================================
# Data loading
# =============================================================================

def load_stock_data_yf(
    ticker: str,
    start: str = "2010-01-01",
    end: Optional[str] = None,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Download OHLCV from Yahoo Finance. Returns a DataFrame indexed by date.

    Parameters
    ----------
    ticker : str
    start, end : str or None
    auto_adjust : bool
    """
    import yfinance as yf  # local import to keep module importable without yfinance

    data = yf.download(ticker, start=start, end=end, auto_adjust=auto_adjust)
    if data.empty:
        raise ValueError(f"No data downloaded for {ticker}. Check symbol or dates.")
    data = _ensure_simple(data)
    return data


def compute_returns_and_momentum(
    stock_data: pd.DataFrame,
    price_col: Optional[str] = None,
    freq: str = "D",
    momentum_window: int = 252,
) -> pd.DataFrame:
    """Compute price, simple returns, and 12M price momentum.

    If price_col is None, the function auto-detects a sensible price column
    (e.g., 'Adj Close', 'Close', or a flattened variant like 'Close_NVDA').

    momentum_window: trading days (252 ~ 12 months)
    freq: 'D' keeps daily; otherwise resample (e.g., 'W', 'M', 'ME').
    """
    # Auto-select price column if not provided
    if price_col is None:
        price_col = pick_price_column(stock_data)
    try:
        prices = stock_data[price_col].copy()
    except KeyError:
        # Retry with auto-detect in case the provided name is missing after flattening
        price_col = pick_price_column(stock_data)
        prices = stock_data[price_col].copy()
    if freq and freq != "D":
        prices = prices.resample(freq).last()
    ret = prices.pct_change()
    mom = prices.pct_change(periods=momentum_window).shift(1)
    out = pd.concat([prices, ret, mom], axis=1)
    out.columns = ["Price", "Return", "PriceMomentum"]
    return _ensure_simple(out)


# ----- Fama–French parsers (flexible, CSV/TXT with commas OR whitespace) -----

def sniff_ff_file(file_path: str, preview: int = 5) -> None:
    """Print a quick sniff of rows beginning with date tokens (YYYYMM or YYYYMMDD)."""
    with open(file_path, "r") as f:
        lines = f.readlines()
    data_lines = [ln for ln in lines if re.match(r"^\s*\d{6,8}(\s|,)", ln)]
    print(f"Total lines: {len(lines)} | Data lines detected: {len(data_lines)}")
    for ln in data_lines[:preview]:
        toks = re.split(r"[,\s]+", ln.strip())
        print(f"{ln.strip()}\n -> tokens={len(toks)} {toks}\n")


def load_ff5_daily(file_path: str) -> pd.DataFrame:
    """Load FF5 daily factors from raw CSV/TXT into DECIMALS with columns:
    ['MKT_RF','SMB','HML','RMW','CMA','RF'] (and 'MOM' if present).
    """
    with open(file_path, "r") as f:
        lines = f.readlines()
    data_lines = [ln for ln in lines if re.match(r"^\s*\d{6,8}(\s|,)", ln)]
    if not data_lines:
        raise ValueError("No data lines detected. Inspect the file or use sniff_ff_file().")

    df = pd.read_csv(StringIO("".join(data_lines)), sep=r"[,\s]+", header=None, engine="python")

    first_date = str(df.iloc[0, 0])
    date_fmt = "%Y%m%d" if len(first_date) == 8 else "%Y%m"

    ncol = df.shape[1]
    if ncol == 7:
        cols = ["Date", "MKT_RF", "SMB", "HML", "RMW", "CMA", "RF"]
    elif ncol == 8:
        cols = ["Date", "MKT_RF", "SMB", "HML", "RMW", "CMA", "RF", "MOM"]
    else:
        raise ValueError(f"Unexpected FF5 column count: {ncol}")

    df.columns = cols
    df["Date"] = pd.to_datetime(df["Date"].astype(str), format=date_fmt)
    df = df.set_index("Date").sort_index()

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0
    return _ensure_simple(df)


def load_momentum_daily(file_path: str) -> pd.DataFrame:
    """Load Fama–French Momentum (Daily) into DECIMALS with 'MOM' column."""
    with open(file_path, "r") as f:
        lines = f.readlines()
    data_lines = [ln for ln in lines if re.match(r"^\s*\d{6,8}(\s|,)", ln)]
    if not data_lines:
        raise ValueError("No data lines detected in momentum file.")

    df = pd.read_csv(StringIO("".join(data_lines)), sep=r"[,\s]+", header=None, engine="python")

    first_date = str(df.iloc[0, 0])
    date_fmt = "%Y%m%d" if len(first_date) == 8 else "%Y%m"

    if df.shape[1] == 2:
        df.columns = ["Date", "MOM"]
    elif df.shape[1] == 3:
        df.columns = ["Date", "MOM", "RF_mom"]  # rarely present; not used
    else:
        raise ValueError(f"Unexpected momentum column count: {df.shape[1]}")

    df["Date"] = pd.to_datetime(df["Date"].astype(str), format=date_fmt)
    df = df.set_index("Date").sort_index()

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0
    df = df[["MOM"]]
    return _ensure_simple(df)


# =============================================================================
# Macro data loaders
# =============================================================================

@dataclass
class MacroConfig:
    # Built-in fetchers (Yahoo Finance)
    use_vix: bool = True
    vix_ticker: str = "^VIX"                 # CBOE VIX Index
    include_oil: bool = True
    oil_ticker: str = "CL=F"                 # WTI Crude Futures
    include_fx_vol: bool = True
    fx_ticker: str = "INR=X"                 # USD/INR (Yahoo format)
    fx_vol_window: int = 21                   # trading days ~ 1 month

    # User-provided CSVs (optional)
    fo_path: Optional[str] = None             # F&O positioning (e.g., net longs or PCR)
    fo_date_col: str = "Date"
    fo_value_col: str = "Signal"

    earnings_sent_path: Optional[str] = None  # Earnings sentiment index
    es_date_col: str = "Date"
    es_value_col: str = "Sentiment"

    gpr_path: Optional[str] = None            # Geopolitical Risk index
    gpr_date_col: str = "Date"
    gpr_value_col: str = "GPR"

    tpu_path: Optional[str] = None            # Trade Policy Uncertainty index
    tpu_date_col: str = "Date"
    tpu_value_col: str = "TPU"

    # Aggregation frequency
    freq: str = "ME"                          # target: Month-End


def _fetch_yf_close_series(ticker: str, start: Optional[str], end: Optional[str]) -> pd.Series:
    import yfinance as yf
    df = yf.download(ticker, start=start, end=end, auto_adjust=False)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    df = _ensure_simple(df)
    # Prefer 'Adj Close' if present; else 'Close'
    col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    s = df[col].rename(ticker)
    s.index = pd.to_datetime(s.index)
    return s


def macro_vix_change(cfg: MacroConfig, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    if not cfg.use_vix:
        return pd.DataFrame()
    s = _fetch_yf_close_series(cfg.vix_ticker, start, end)
    # Monthly change (level diff) and pct change
    vix_m = s.resample(cfg.freq).last()
    out = pd.DataFrame({
        'dVIX': vix_m.diff(),
        'dVIX_pct': vix_m.pct_change(),
        'VIX': vix_m,
    })
    return out


def macro_oil_return(cfg: MacroConfig, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    if not cfg.include_oil:
        return pd.DataFrame()
    s = _fetch_yf_close_series(cfg.oil_ticker, start, end)
    o = s.resample(cfg.freq).last().pct_change().rename('OilRet')
    return o.to_frame()


def macro_fx_realized_vol(cfg: MacroConfig, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    if not cfg.include_fx_vol or not cfg.fx_ticker:
        return pd.DataFrame()
    s = _fetch_yf_close_series(cfg.fx_ticker, start, end)
    logret = np.log(s).diff()
    # 21d rolling realized vol annualized
    rv = logret.rolling(cfg.fx_vol_window).std() * np.sqrt(252)
    fxv_m = rv.resample(cfg.freq).last().rename('FXVol')
    return fxv_m.to_frame()


def _load_csv_macro(path: str, date_col: str, value_col: str, out_name: str, freq: str = 'ME') -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    df = pd.read_csv(path)
    if date_col not in df.columns or value_col not in df.columns:
        raise ValueError(f"CSV {path} must contain columns '{date_col}' and '{value_col}'")
    # Try multiple date formats automatically
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    df = df.set_index(date_col).sort_index()
    s = pd.to_numeric(df[value_col], errors='coerce').dropna().rename(out_name)
    # If higher frequency than monthly, aggregate to month-end (mean)
    s_m = s.resample(freq).mean()
    return s_m.to_frame()


def build_macro_monthly(cfg: MacroConfig, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    parts = []
    try:
        parts.append(macro_vix_change(cfg, start, end))
    except Exception:
        pass
    try:
        parts.append(macro_oil_return(cfg, start, end))
    except Exception:
        pass
    try:
        parts.append(macro_fx_realized_vol(cfg, start, end))
    except Exception:
        pass
    # User-provided files
    parts.append(_load_csv_macro(cfg.fo_path, cfg.fo_date_col, cfg.fo_value_col, 'FO_Position', cfg.freq) if cfg.fo_path else pd.DataFrame())
    parts.append(_load_csv_macro(cfg.earnings_sent_path, cfg.es_date_col, cfg.es_value_col, 'EarningsSent', cfg.freq) if cfg.earnings_sent_path else pd.DataFrame())
    parts.append(_load_csv_macro(cfg.gpr_path, cfg.gpr_date_col, cfg.gpr_value_col, 'GPR', cfg.freq) if cfg.gpr_path else pd.DataFrame())
    parts.append(_load_csv_macro(cfg.tpu_path, cfg.tpu_date_col, cfg.tpu_value_col, 'TPU', cfg.freq) if cfg.tpu_path else pd.DataFrame())

    parts = [p for p in parts if isinstance(p, pd.DataFrame) and not p.empty]
    if not parts:
        return pd.DataFrame()
    macro = pd.concat(parts, axis=1)
    macro = _ensure_simple(macro)
    return macro


# =============================================================================
# Dataset assembly (daily and monthly)
# =============================================================================

def build_daily_dataset(
    stock_data: pd.DataFrame,
    ff5_daily: pd.DataFrame,
    mom_daily: Optional[pd.DataFrame] = None,
    price_col: Optional[str] = None,
    freq: str = "D",
    momentum_window: int = 252,
) -> pd.DataFrame:
    """Join price/returns with FF factors at daily (or resampled) frequency.

    Returns columns: ['Price','Return','PriceMomentum','MKT_RF','SMB','HML','RMW','CMA','RF',('MOM')]
    plus 'Excess_Return' (stock Return − RF).
    """
    pr = compute_returns_and_momentum(stock_data, price_col=price_col, freq=freq, momentum_window=momentum_window)
    factors = ff5_daily.copy()
    if freq and freq != "D":
        # resample factors to the same freq via last business day in period then compound
        # For returns, compounding is needed; here we keep daily and resample via last
        # but to remain precise, we will join at daily then downsample later at monthly
        pass  # keep factors daily; we'll handle compounding in to_monthly()
    if mom_daily is not None and "MOM" in mom_daily.columns:
        factors = factors.join(mom_daily[["MOM"]], how="left")
    df = pr.join(factors, how="inner").dropna(subset=["Return", "RF"])  # need RF
    df["Excess_Return"] = df["Return"] - df["RF"]
    return _ensure_simple(df)


def to_monthly(data_daily: pd.DataFrame) -> pd.DataFrame:
    """Convert daily dataset to month-end using compounding for returns/factors.

    - Stock monthly return: last Close pct_change OR compound of daily Return.
    - Factor monthly returns: compound each factor column.
    - RF is compounded as (1+RF).prod()-1.
    """
    dd = _ensure_simple(data_daily)

    # Price-derived monthly return: prefer compounding of daily Return for robustness
    stock_m = dd["Return"].resample("ME").apply(compound_return).to_frame(name="Return")

    # Compound factor returns by month
    factor_cols = [c for c in ["MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"] if c in dd.columns]
    fac_m = dd[factor_cols].resample("ME").apply(compound_return)

    # Risk-free compounded over the month
    rf_m = dd[["RF"]].resample("ME").apply(compound_return)

    out = stock_m.join(fac_m, how="inner").join(rf_m, how="inner").dropna()
    out["Excess_Return"] = out["Return"] - out["RF"]
    return _ensure_simple(out)


# =============================================================================
# Regression & decomposition (monthly)
# =============================================================================

def ff5_regression_monthly(
    data_m: pd.DataFrame,
    factor_cols: Iterable[str] = ("MKT_RF", "SMB", "HML", "RMW", "CMA"),
) -> Tuple[sm.regression.linear_model.RegressionResultsWrapper, pd.Series, pd.Series, pd.Series]:
    """Run OLS of Excess_Return ~ const + FF5 factors (monthly).

    Returns
    -------
    res : statsmodels results
    fitted_excess : Series of fitted excess returns
    residual_return : y - fitted
    beta_term : beta·f (ex‑alpha fitted component)
    """
    cols = [c for c in factor_cols if c in data_m.columns]
    X = sm.add_constant(data_m[cols])
    y = data_m["Excess_Return"]
    res = sm.OLS(y, X, missing="drop").fit()

    beta = res.params[cols]
    alpha_const = float(res.params.get("const", 0.0))

    beta_term = data_m[cols].mul(beta, axis=1).sum(axis=1)
    fitted_excess = alpha_const + beta_term
    residual_return = y - fitted_excess
    return res, fitted_excess.rename("Fitted_Excess"), residual_return.rename("Residual"), beta_term.rename("BetaTerm")


def contribution_table_monthly_and_annual(
    res: sm.regression.linear_model.RegressionResultsWrapper,
    data_m: pd.DataFrame,
    factor_cols: Iterable[str] = ("MKT_RF", "SMB", "HML", "RMW", "CMA"),
) -> pd.DataFrame:
    params = res.params
    means = data_m[list(factor_cols)].mean()

    contrib_m = pd.Series({"alpha": params.get("const", 0.0)}, name="Monthly")
    for k in factor_cols:
        if k in params.index:
            contrib_m.loc[k] = params[k] * means[k]
    contrib_a = (contrib_m * 12.0).rename("Annualized (~×12)")
    table = pd.concat([contrib_m, contrib_a], axis=1)
    table.loc["TOTAL"] = [contrib_m.sum(), contrib_a.sum()]
    return table


# =============================================================================
# Extended regression with macro factors (monthly)
# =============================================================================

def extended_regression_monthly(
    data_m: pd.DataFrame,
    macro_m: pd.DataFrame,
    factor_cols: Iterable[str] = ("MKT_RF", "SMB", "HML", "RMW", "CMA"),
    macro_cols: Optional[Iterable[str]] = None,
):
    """OLS of Excess_Return ~ const + FF factors + Macro indicators.

    Returns
    -------
    res : statsmodels results
    fitted_excess : Series alpha + beta·f + gamma·m
    residual_return : y - fitted
    beta_term : beta·f (factors only)
    gamma_term : gamma·m (macros only)
    explanatory_term : beta_term + gamma_term
    """
    macro_cols = list(macro_cols) if macro_cols is not None else list(macro_m.columns)

    # Align indices and build X
    joined = data_m.join(macro_m[macro_cols], how="inner")
    cols_f = [c for c in factor_cols if c in joined.columns]
    cols_m = [c for c in macro_cols if c in joined.columns]

    X = sm.add_constant(joined[cols_f + cols_m])
    y = joined["Excess_Return"]

    res = sm.OLS(y, X, missing="drop").fit()

    beta = res.params[cols_f] if cols_f else pd.Series(dtype=float)
    gamma = res.params[cols_m] if cols_m else pd.Series(dtype=float)
    alpha_const = float(res.params.get("const", 0.0))

    beta_term = joined[cols_f].mul(beta, axis=1).sum(axis=1) if cols_f else pd.Series(0.0, index=joined.index)
    gamma_term = joined[cols_m].mul(gamma, axis=1).sum(axis=1) if cols_m else pd.Series(0.0, index=joined.index)
    fitted_excess = alpha_const + beta_term + gamma_term
    residual_return = y - fitted_excess

    return (
        res,
        fitted_excess.rename("Fitted_Excess_ext"),
        residual_return.rename("Residual_ext"),
        beta_term.rename("BetaTerm"),
        gamma_term.rename("GammaTerm"),
        (beta_term + gamma_term).rename("ExplanatoryTerm"),
    )


def contribution_table_extended(
    res: sm.regression.linear_model.RegressionResultsWrapper,
    data_m: pd.DataFrame,
    macro_m: pd.DataFrame,
    factor_cols: Iterable[str] = ("MKT_RF", "SMB", "HML", "RMW", "CMA"),
    macro_cols: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    macro_cols = list(macro_cols) if macro_cols is not None else list(macro_m.columns)
    joined = data_m.join(macro_m[macro_cols], how="inner")

    params = res.params
    means_f = joined[[c for c in factor_cols if c in joined.columns]].mean()
    means_m = joined[[c for c in macro_cols if c in joined.columns]].mean()

    contrib_m = pd.Series({"alpha": params.get("const", 0.0)}, name="Monthly")
    for k in means_f.index:
        if k in params.index:
            contrib_m.loc[k] = params[k] * means_f[k]
    for k in means_m.index:
        if k in params.index:
            contrib_m.loc[k] = params[k] * means_m[k]

    contrib_a = (contrib_m * 12.0).rename("Annualized (~×12)")
    table = pd.concat([contrib_m, contrib_a], axis=1)
    table.loc["TOTAL"] = [contrib_m.sum(), contrib_a.sum()]
    return table


# =============================================================================
# Strategies (monthly)
# =============================================================================

def strategy_alpha_residual_mr(
    residual_return: pd.Series,
    excess_return: pd.Series,
    beta_term: pd.Series,
    lookback_z: int = 12,
    enter_long_z: float = -1.5,
    enter_short_z: float = 1.5,
    exit_z: float = 0.5,
    time_stop_m: int = 3,
    cost_per_change: float = 0.001,
    risk_target: Optional[float] = None,  # annualized vol target for sizing
) -> Tuple[pd.Series, pd.Series]:
    """Residual mean‑reversion alpha trade on monthly data.

    Returns (alpha_trade_returns, position_series)
    """
    # Position via residual z-score
    z = (residual_return - residual_return.rolling(lookback_z).mean()) / residual_return.rolling(lookback_z).std()

    pos = pd.Series(0, index=residual_return.index, dtype=int)
    in_pos = 0
    bars = 0
    for t in residual_return.index:
        zt = z.loc[t]
        if np.isnan(zt):
            pos.loc[t] = in_pos
            continue
        if in_pos == 0:
            if zt <= enter_long_z:
                in_pos, bars = 1, 0
            elif zt >= enter_short_z:
                in_pos, bars = -1, 0
        else:
            bars += 1
            if (abs(zt) < exit_z) or (bars >= time_stop_m):
                in_pos, bars = 0, 0
        pos.loc[t] = in_pos

    # Hedged leg = Excess − beta·f  (i.e., alpha + residual)
    hedged_leg = excess_return - beta_term
    ret = hedged_leg * pos

    # Costs on position changes
    turnover = pos.diff().abs().fillna(0)
    ret = ret - cost_per_change * turnover

    if risk_target is not None and risk_target > 0:
        resid_vol_12m = residual_return.rolling(12).std() * np.sqrt(12)
        size = (risk_target / resid_vol_12m).clip(upper=1.0).fillna(0.0)
        ret = (hedged_leg * pos * size) - cost_per_change * turnover

    return ret.rename("AlphaTrade_residualMR"), pos.rename("Alpha_Position")


def strategy_beta_trade(
    fitted_excess: pd.Series,
    excess_return: pd.Series,
    ma: int = 6,
) -> pd.Series:
    """Own factors when fitted trend is up and residual trend is down; else own stock excess.
    Returns the monthly return series for the strategy.
    """
    resid = excess_return - fitted_excess
    factor_trend_up = fitted_excess.rolling(ma).mean() > 0
    resid_trend_down = resid.rolling(ma).mean() < 0
    hold_factor = (factor_trend_up & resid_trend_down)
    beta_term_only = fitted_excess - fitted_excess.index.to_series().map(lambda _: 0)  # identity; clarity
    # Equivalent to beta_term; but we simply switch between factor fit and actual excess
    out = pd.Series(np.where(hold_factor, fitted_excess, excess_return), index=excess_return.index)
    return out.rename("BetaTrade_factorWhenResidualWeak")


def strategy_momentum_confirm(
    fitted_excess: pd.Series,
    residual_return: pd.Series,
    excess_return: pd.Series,
    trend_len: int = 6,
) -> pd.Series:
    """Long excess only when both fitted and residual indices trend up over `trend_len` months."""
    fitted_idx = cum_index(fitted_excess)
    resid_idx = cum_index(residual_return)
    fitted_up = fitted_idx > fitted_idx.shift(trend_len)
    resid_up = resid_idx > resid_idx.shift(trend_len)
    long_when_both = fitted_up & resid_up
    out = excess_return.where(long_when_both, 0.0)
    return out.rename("MomentumConfirm_longWhenBothUp")


# =============================================================================
# Orchestrator
# =============================================================================

def plot_equity_curves(curves: pd.DataFrame, title: str = "Equity Curves", out_path: Optional[str] = None) -> None:
    """Plot cumulative equity indices for strategies and optionally save to file."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    for col in curves.columns:
        plt.plot(curves.index, curves[col], label=col)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Index (start = 1)")
    plt.legend()
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
    plt.close()


def clean_symbol(sym: str) -> str:
    """Best-effort clean-up for ticker symbols for yfinance.

    - Trim whitespace
    - Convert dots to dashes (e.g., BRK.B -> BRK-B)
    - Keep alphanumerics and dashes only
    """
    s = str(sym).strip().upper()
    s = s.replace(".", "-")
    s = re.sub(r"[^A-Z0-9\-]", "", s)
    return s


def run_analysis(
    ticker: str,
    ff5_path: str,
    mom_path: Optional[str] = None,
    start: str = "2010-01-01",
    end: Optional[str] = None,
    price_col: Optional[str] = None,
) -> Dict[str, object]:
    """High-level: load everything, run regression & strategies, return artifacts.

    Returns dict with keys:
      - data_daily, data_monthly
      - params (Series), summary_text (str)
      - contrib_table (DataFrame)
      - fitted_excess, residual_return, beta_term (Series)
      - strategies returns (DataFrame) and stats (DataFrame)
      - curves (DataFrame) cumulative equity indices for strategies
    """
    stock = load_stock_data_yf(ticker, start=start, end=end)

    ff5 = load_ff5_daily(ff5_path)
    mom = load_momentum_daily(mom_path) if mom_path else None

    daily = build_daily_dataset(stock, ff5, mom, price_col=price_col)
    monthly = to_monthly(daily)

    res, fitted_excess, residual_return, beta_term = ff5_regression_monthly(monthly)
    contrib_table = contribution_table_monthly_and_annual(res, monthly)

    # Strategies
    alpha_ret, alpha_pos = strategy_alpha_residual_mr(
        residual_return=residual_return,
        excess_return=monthly["Excess_Return"],
        beta_term=beta_term,
    )
    alpha_sized_ret, _ = strategy_alpha_residual_mr(
        residual_return=residual_return,
        excess_return=monthly["Excess_Return"],
        beta_term=beta_term,
        risk_target=0.10,
    )
    beta_ret = strategy_beta_trade(fitted_excess, monthly["Excess_Return"])
    mom_ret = strategy_momentum_confirm(fitted_excess, residual_return, monthly["Excess_Return"])

    # Stats
    stats = pd.concat(
        [
            perf_summary(alpha_ret, "AlphaTrade_residualMR"),
            perf_summary(beta_ret, "BetaTrade_factorWhenResidualWeak"),
            perf_summary(mom_ret, "MomentumConfirm_longWhenBothUp"),
            perf_summary(alpha_sized_ret, "AlphaTrade_residualMR_sized"),
        ],
        axis=1,
    ).T

    # Curves
    curves = pd.DataFrame(
        {
            "AlphaTrade": cum_index(alpha_ret.fillna(0)),
            "BetaTrade": cum_index(beta_ret.fillna(0)),
            "MomConfirm": cum_index(mom_ret.fillna(0)),
        }
    )

    out = {
        "data_daily": daily,
        "data_monthly": monthly,
        "params": res.params,
        "summary_text": str(res.summary()),
        "contrib_table": contrib_table,
        "fitted_excess": fitted_excess,
        "residual_return": residual_return,
        "beta_term": beta_term,
        "strategies": pd.DataFrame({
            "AlphaTrade_residualMR": alpha_ret,
            "AlphaTrade_residualMR_sized": alpha_sized_ret,
            "BetaTrade_factorWhenResidualWeak": beta_ret,
            "MomentumConfirm_longWhenBothUp": mom_ret,
        }),
        "stats": stats,
        "curves": curves,
    }
    return out


# =============================================================================
# Orchestrator (with macro)
# =============================================================================

def run_analysis_with_macro(
    ticker: str,
    ff5_path: str,
    mom_path: Optional[str] = None,
    macro_cfg: Optional[MacroConfig] = None,
    start: str = "2010-01-01",
    end: Optional[str] = None,
    price_col: Optional[str] = None,
) -> Dict[str, object]:
    """High-level pipeline with macro indicators added to FF factors.

    Returns dict with everything from `run_analysis` plus extended regression
    and strategies that hedge both factor and macro terms.
    """
    base = run_analysis(ticker=ticker, ff5_path=ff5_path, mom_path=mom_path, start=start, end=end, price_col=price_col)
    monthly = base["data_monthly"]

    cfg = macro_cfg or MacroConfig()
    macro_m = build_macro_monthly(cfg, start, end)

    if macro_m.empty:
        # No macros available; fall back to base
        return {**base, "macro_monthly": macro_m}

    res_ext, fitted_excess_ext, residual_ext, beta_term, gamma_term, expl_term = extended_regression_monthly(
        monthly, macro_m
    )
    contrib_ext = contribution_table_extended(res_ext, monthly, macro_m)

    # Strategies using extended fit (hedge by full explanatory term)
    alpha_ext, alpha_pos = strategy_alpha_residual_mr(
        residual_return=residual_ext,
        excess_return=monthly["Excess_Return"],
        beta_term=expl_term,  # pass full term as hedge
    )
    alpha_ext_sized, _ = strategy_alpha_residual_mr(
        residual_return=residual_ext,
        excess_return=monthly["Excess_Return"],
        beta_term=expl_term,
        risk_target=0.10,
    )
    beta_trade_ext = strategy_beta_trade(fitted_excess_ext, monthly["Excess_Return"])  # same logic
    mom_conf_ext = strategy_momentum_confirm(fitted_excess_ext, residual_ext, monthly["Excess_Return"])  # same logic

    stats_ext = pd.concat(
        [
            perf_summary(alpha_ext, "AlphaTrade_residualMR_ext"),
            perf_summary(beta_trade_ext, "BetaTrade_factorWhenResidualWeak_ext"),
            perf_summary(mom_conf_ext, "MomentumConfirm_longWhenBothUp_ext"),
            perf_summary(alpha_ext_sized, "AlphaTrade_residualMR_sized_ext"),
        ],
        axis=1,
    ).T

    curves_ext = pd.DataFrame(
        {
            "AlphaTrade_ext": cum_index(alpha_ext.fillna(0)),
            "BetaTrade_ext": cum_index(beta_trade_ext.fillna(0)),
            "MomConfirm_ext": cum_index(mom_conf_ext.fillna(0)),
        }
    )

    extended = {
        **base,
        "macro_monthly": macro_m,
        "extended_params": res_ext.params,
        "extended_summary_text": str(res_ext.summary()),
        "extended_contrib_table": contrib_ext,
        "fitted_excess_ext": fitted_excess_ext,
        "residual_ext": residual_ext,
        "beta_term_ext": beta_term,
        "gamma_term_ext": gamma_term,
        "explanatory_term_ext": expl_term,
        "strategies_ext": pd.DataFrame({
            "AlphaTrade_residualMR_ext": alpha_ext,
            "AlphaTrade_residualMR_sized_ext": alpha_ext_sized,
            "BetaTrade_factorWhenResidualWeak_ext": beta_trade_ext,
            "MomentumConfirm_longWhenBothUp_ext": mom_conf_ext,
        }),
        "stats_ext": stats_ext,
        "curves_ext": curves_ext,
    }
    return extended


# =============================================================================
# Batch runner across many tickers
# =============================================================================

def batch_run_analysis(
    tickers: Iterable[str],
    ff5_path: str,
    mom_path: Optional[str] = None,
    start: str = "2010-01-01",
    end: Optional[str] = None,
    out_dir: str = "factor_batch_out",
    max_workers: int = 3,
    save_plots: bool = True,
) -> Dict[str, object]:
    """Run the full pipeline for many tickers and save artifacts per ticker.

    Outputs per ticker (under `out_dir`):
      - <TICKER>_stats.csv
      - <TICKER>_params.csv
      - <TICKER>_contrib.csv
      - <TICKER>_equity_curves.png  (if save_plots)

    Returns a dict with combined frames:
      - results: dict[ticker] -> single-ticker result dict from run_analysis
      - stats_long: MultiIndex rows (ticker, strategy) with AnnRet/AnnVol/Sharpe
      - params: DataFrame of OLS params by ticker
    """
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed

    os.makedirs(out_dir, exist_ok=True)

    cleaned = [clean_symbol(t) for t in tickers if isinstance(t, str) and t.strip()]

    results: Dict[str, Dict[str, object]] = {}
    errors: Dict[str, str] = {}

    def _one(tk: str) -> Tuple[str, Optional[Dict[str, object]], Optional[str]]:
        try:
            res = run_analysis(tk, ff5_path=ff5_path, mom_path=mom_path, start=start, end=end)
            # Save per-ticker artifacts
            if isinstance(res.get("stats"), pd.DataFrame):
                res["stats"].to_csv(os.path.join(out_dir, f"{tk}_stats.csv"))
            if isinstance(res.get("params"), pd.Series):
                res["params"].to_csv(os.path.join(out_dir, f"{tk}_params.csv"))
            if isinstance(res.get("contrib_table"), pd.DataFrame):
                res["contrib_table"].to_csv(os.path.join(out_dir, f"{tk}_contrib.csv"))
            if save_plots and isinstance(res.get("curves"), pd.DataFrame):
                plot_equity_curves(res["curves"], title=f"{tk} – Equity Curves", out_path=os.path.join(out_dir, f"{tk}_equity_curves.png"))
            return tk, res, None
        except Exception as e:
            return tk, None, str(e)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_one, tk): tk for tk in cleaned}
        for fut in as_completed(futs):
            tk, res, err = fut.result()
            if err is None:
                results[tk] = res
            else:
                errors[tk] = err

    # Combine stats
    stats_rows = []
    for tk, res in results.items():
        st = res.get("stats")
        if isinstance(st, pd.DataFrame) and not st.empty:
            st2 = st.copy()
            st2.insert(0, "Ticker", tk)
            stats_rows.append(st2)
    stats_long = pd.concat(stats_rows, axis=0) if stats_rows else pd.DataFrame()
    if not stats_long.empty:
        stats_long.set_index(["Ticker"], append=False, inplace=True)
        stats_long.index.name = "Ticker"
        stats_long.to_csv(os.path.join(out_dir, "_SUMMARY_stats_long.csv"))

    # Combine params
    params_rows = []
    for tk, res in results.items():
        p = res.get("params")
        if isinstance(p, pd.Series):
            sr = p.copy()
            sr.name = tk
            params_rows.append(sr)
    params_df = pd.DataFrame(params_rows)
    if not params_df.empty:
        params_df.index.name = "Ticker"
        params_df.to_csv(os.path.join(out_dir, "_SUMMARY_params.csv"))

    # Optional global plots: Sharpe by strategy per ticker
    if save_plots and not stats_long.empty:
        import matplotlib.pyplot as plt
        for metric in ["Sharpe", "AnnRet", "AnnVol"]:
            try:
                tbl = stats_long.reset_index()
                # stats_long has strategies as index rows; ensure a column exists
                if "index" in tbl.columns:
                    tbl.rename(columns={"index": "Strategy"}, inplace=True)
                if "Strategy" not in tbl.columns:
                    # the original stats rows keep strategy as DataFrame index
                    tbl = tbl.rename_axis("Strategy").reset_index()
                # One figure per strategy
                for strat, sub in tbl.groupby("Strategy"):
                    sub_sorted = sub.sort_values(metric, ascending=(metric=="AnnVol"))
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.bar(sub_sorted["Ticker"], sub_sorted[metric])
                    ax.set_title(f"{metric} – {strat}")
                    ax.set_xlabel("Ticker")
                    ax.set_ylabel(metric)
                    plt.xticks(rotation=90)
                    plt.tight_layout()
                    plt.savefig(os.path.join(out_dir, f"_SUMMARY_{metric}_{strat}.png"), dpi=150)
                    plt.close()
            except Exception:
                pass

    return {"results": results, "errors": errors, "stats_long": stats_long, "params": params_df}


# End of module
