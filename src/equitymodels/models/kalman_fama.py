
"""
Time‑Varying State‑Space Kalman Filter (+ RTS smoother)
=======================================================

Self‑contained NumPy implementation that supports:
- Time‑varying transition/observation matrices F_t, H_t and noises Q_t, R_t
- Optional intercepts c_t (state) and d_t (obs)
- Missing observations (NaNs) at any time (partial measurement update)
- Exact log‑likelihood via Cholesky
- Rauch–Tung–Striebel (RTS) fixed‑interval smoothing

Indexing convention
-------------------
We assume the **prediction** uses F_t and Q_t to go from t-1 → t:
    a_{t|t-1} = F_t a_{t-1|t-1} + c_t
    P_{t|t-1} = F_t P_{t-1|t-1} F_t' + Q_t
Measurement at time t uses H_t and R_t:
    y_t = d_t + H_t a_t + eps_t,  eps_t ~ N(0, R_t)

So F_t, Q_t, c_t refer to the transition into time t and
H_t, R_t, d_t refer to the measurement at time t.

Outputs
-------
- a_pred[t], P_pred[t] : a_{t|t-1}, P_{t|t-1}
- a_filt[t], P_filt[t] : a_{t|t},   P_{t|t}
- v[t], S[t]           : innovation and its covariance for observed dims
- K[t]                 : Kalman gain (stored as full size with NaNs in unobserved cols)
- loglik               : total Gaussian log-likelihood (ignores all‑missing timesteps)
- a_smooth, P_smooth   : RTS smoothed states/covariances

Minimal dependencies: numpy only (no SciPy).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Dict, List
import numpy as np

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _as_time_varying(M: np.ndarray, T: int, name: str) -> np.ndarray:
    """Broadcast a 2D matrix to (T, m, n) if needed, or validate 3D.
    """
    M = np.asarray(M)
    if M.ndim == 2:
        m, n = M.shape
        return np.broadcast_to(M, (T, m, n)).copy()
    if M.ndim == 3:
        if M.shape[0] != T:
            raise ValueError(f"{name} has T={M.shape[0]} ≠ {T}")
        return M.copy()
    raise ValueError(f"{name} must be 2D or 3D")


def _as_time_varying_vec(v: Optional[np.ndarray], T: int, n: int, name: str) -> np.ndarray:
    if v is None:
        return np.zeros((T, n))
    v = np.asarray(v)
    if v.ndim == 1:
        if v.shape[0] != n:
            raise ValueError(f"{name} length {v.shape[0]} ≠ {n}")
        return np.broadcast_to(v, (T, n)).copy()
    if v.ndim == 2:
        if v.shape != (T, n):
            raise ValueError(f"{name} shape {v.shape} ≠ ({T},{n})")
        return v.copy()
    raise ValueError(f"{name} must be 1D or 2D")


def _chol_solve(S: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Solve S X = B using Cholesky, with small jitter if needed."""
    try:
        L = np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        S = S + 1e-9 * np.eye(S.shape[0])
        L = np.linalg.cholesky(S)
    # Solve L Y = B, then L' X = Y
    Y = np.linalg.solve(L, B)
    X = np.linalg.solve(L.T, Y)
    return X


def _chol_logdet(S: np.ndarray) -> float:
    try:
        L = np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        S = S + 1e-9 * np.eye(S.shape[0])
        L = np.linalg.cholesky(S)
    return 2.0 * np.sum(np.log(np.diag(L)))


# -----------------------------------------------------------------------------
# Core Kalman filter
# -----------------------------------------------------------------------------

def kalman_filter_tv(
    y: np.ndarray,
    F: np.ndarray,
    H: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    a0: np.ndarray,
    P0: np.ndarray,
    c: Optional[np.ndarray] = None,
    d: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Time‑varying Kalman filter with missing‑data support and log‑likelihood.

    Parameters
    ----------
    y : (T, m) array with NaNs allowed for missing observations.
    F : (T, n, n) or (n, n)
    H : (T, m, n) or (m, n)
    Q : (T, n, n) or (n, n)
    R : (T, m, m) or (m, m)
    a0: (n,) prior mean of state at t=0|t=-1
    P0: (n,n) prior cov
    c : (T, n) or (n,) optional state intercept
    d : (T, m) or (m,) optional observation intercept

    Returns dict with arrays/lists as documented in the module docstring.
    """
    y = np.asarray(y, float)
    T, m = y.shape

    # Dimensions from a0/P0
    a0 = np.asarray(a0, float)
    P0 = np.asarray(P0, float)
    n = a0.shape[0]
    assert P0.shape == (n, n)

    # Broadcast time‑varying matrices/vectors
    F = _as_time_varying(np.asarray(F, float), T, "F")
    H = _as_time_varying(np.asarray(H, float), T, "H")
    Q = _as_time_varying(np.asarray(Q, float), T, "Q")
    R = _as_time_varying(np.asarray(R, float), T, "R")
    c = _as_time_varying_vec(None if c is None else np.asarray(c, float), T, n, "c")
    d = _as_time_varying_vec(None if d is None else np.asarray(d, float), T, m, "d")

    # Storage
    a_pred = np.zeros((T, n))
    P_pred = np.zeros((T, n, n))
    a_filt = np.zeros((T, n))
    P_filt = np.zeros((T, n, n))

    v_list: List[np.ndarray] = []
    S_list: List[np.ndarray] = []
    K_list: List[np.ndarray] = []

    loglik = 0.0

    # Initialize with prior a0, P0 then perform prediction to t=0
    a_pr = F[0] @ a0 + c[0]
    P_pr = F[0] @ P0 @ F[0].T + Q[0]

    for t in range(T):
        if t > 0:
            a_pr = F[t] @ a_fi + c[t]
            P_pr = F[t] @ P_fi @ F[t].T + Q[t]

        a_pred[t] = a_pr
        P_pred[t] = P_pr

        # Measurement update with observed slice only
        yt = y[t]
        obs_mask = np.isfinite(yt)
        if not np.any(obs_mask):
            # No observation: filtered = predicted
            a_fi, P_fi = a_pr, P_pr
            v_list.append(np.full((0,), np.nan))
            S_list.append(np.full((0, 0), np.nan))
            K_list.append(np.full((n, 0), np.nan))
            a_filt[t] = a_fi
            P_filt[t] = P_fi
            continue

        Ht = H[t][obs_mask, :]          # (k,n)
        Rt = R[t][np.ix_(obs_mask, obs_mask)]  # (k,k)
        dt = d[t][obs_mask]             # (k,)
        yt_obs = yt[obs_mask]

        # Innovation v and its covariance S
        v = yt_obs - (dt + Ht @ a_pr)
        S = Ht @ P_pr @ Ht.T + Rt

        # Gain via Cholesky solve for stability
        PHt = P_pr @ Ht.T               # (n,k)
        K = _chol_solve(S, PHt.T).T     # (n,k)

        # Update
        a_fi = a_pr + K @ v
        P_fi = P_pr - K @ S @ K.T

        # Log-likelihood contribution
        k = v.shape[0]
        quad = v.T @ _chol_solve(S, v)
        logdet = _chol_logdet(S)
        loglik += -0.5 * (k * np.log(2 * np.pi) + logdet + quad)

        # Store
        a_filt[t] = a_fi
        P_filt[t] = P_fi
        v_list.append(v)
        S_list.append(S)

        # Make a full-size K with NaNs for missing obs (n x m)
        K_full = np.full((n, m), np.nan)
        K_full[:, obs_mask] = K
        K_list.append(K_full)

    return {
        "a_pred": a_pred,
        "P_pred": P_pred,
        "a_filt": a_filt,
        "P_filt": P_filt,
        "innovations": v_list,
        "S": S_list,
        "K": K_list,
        "loglik": float(loglik),
    }


# -----------------------------------------------------------------------------
# RTS Smoother
# -----------------------------------------------------------------------------

def rts_smoother(
    a_pred: np.ndarray,
    P_pred: np.ndarray,
    a_filt: np.ndarray,
    P_filt: np.ndarray,
    F: np.ndarray,
    Q: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Rauch–Tung–Striebel smoother for time‑varying systems."""
    T, n = a_filt.shape
    F = _as_time_varying(F, T, "F")
    Q = _as_time_varying(Q, T, "Q")

    a_s = np.zeros_like(a_filt)
    P_s = np.zeros_like(P_filt)

    # Initialize at T-1
    a_s[-1] = a_filt[-1]
    P_s[-1] = P_filt[-1]

    for t in range(T - 2, -1, -1):
        # Smoother gain J_t = P_{t|t} F_{t+1}' P_{t+1|t}^{-1}
        # Note: here F[t+1] maps t -> t+1
        Ft1 = F[t + 1]
        P_pred_next = P_pred[t + 1]
        # Solve P_{t+1|t} X = Ft1 P_{t|t}
        # We'll compute J_t via: J = P_filt[t] F_{t+1}' (P_pred[t+1])^{-1}
        RHS = (P_filt[t] @ Ft1.T)  # (n,n)
        J = np.linalg.solve(P_pred_next, RHS.T).T

        a_s[t] = a_filt[t] + J @ (a_s[t + 1] - a_pred[t + 1])
        P_s[t] = P_filt[t] + J @ (P_s[t + 1] - P_pred_next) @ J.T

    return {"a_smooth": a_s, "P_smooth": P_s}


# -----------------------------------------------------------------------------
# Convenience: Build Nelson–Siegel observation matrix H given maturities
# -----------------------------------------------------------------------------

def nelson_siegel_loadings(maturities: Iterable[float], lam: float) -> np.ndarray:
    """Return H (m,n) for NS with state [beta0, beta1, beta2].

    y(τ) = beta0*1 + beta1*L1(τ) + beta2*L2(τ)
    L1(τ) = (1 - e^{-λτ})/(λτ)
    L2(τ) = L1(τ) - e^{-λτ}
    """
    taus = np.asarray(list(maturities), float)
    L1 = (1 - np.exp(-lam * taus)) / (lam * taus)
    L2 = L1 - np.exp(-lam * taus)
    H = np.column_stack([np.ones_like(taus), L1, L2])
    return H


# -----------------------------------------------------------------------------
# Time‑varying parameter (TVP) regression wrapper
# -----------------------------------------------------------------------------

def tvp_regression(
    y: np.ndarray,
    X: np.ndarray,
    include_intercept: bool = True,
    phi: Optional[float] = None,
    q_scale: float = 1e-5,
    r_scale: Optional[float] = None,
    a0: Optional[np.ndarray] = None,
    P0_scale: float = 1.0,
    names: Optional[Iterable[str]] = None,
    return_dict: bool = False,
):
    """Kalman TVP regression: y_t = x_t' beta_t + eps_t, beta_t = F beta_{t-1} + eta_t.

    Parameters
    ----------
    y : (T,) or (T,1) target series
    X : (T,k) regressors (e.g., [1, MKT_RF, SMB, ...])
    include_intercept : if False, do not auto‑prepend a column of ones
    phi : AR(1) coefficient for beta dynamics. If None -> random walk (F=I).
          If scalar -> same phi for all states. If array‑like len=k, elementwise AR(1).
    q_scale : Process noise scale. Larger -> more time variation. Try 1e‑6 .. 1e‑3.
    r_scale : Measurement noise variance. If None -> set to var(y - X*beta_OLS).
    a0 : Initial beta mean (k,). If None -> OLS beta (or zeros if singular).
    P0_scale : Initial covariance scale * I.
    names : Optional names for coefficients (len=k). Used for DataFrame columns.
    return_dict : If True, also return raw filter/smoother dicts.

    Returns
    -------
    betas_sm : (T,k) smoothed time‑varying coefficients as np.ndarray
    (optionally dict with extras when return_dict=True)
    """
    y = np.asarray(y, float).reshape(-1)
    T = y.shape[0]

    X = np.asarray(X, float)
    if include_intercept:
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X = np.column_stack([np.ones(T), X])
    else:
        if X.ndim == 1:
            X = X.reshape(-1, 1)
    T2, k = X.shape
    assert T2 == T, "X and y must have same number of rows"

    # OLS fallback for a0 and r_scale
    XtX = X.T @ X
    try:
        beta_ols = np.linalg.solve(XtX, X.T @ y)
    except np.linalg.LinAlgError:
        beta_ols = np.zeros(k)
    resid = y - X @ beta_ols
    s2 = float(np.var(resid, ddof=min(k, T) + 1)) if T > k + 1 else float(np.var(resid))

    if r_scale is None or not np.isfinite(r_scale):
        r_scale = max(s2, 1e-8)
    if a0 is None:
        a0 = beta_ols

    # Build time‑varying system
    if phi is None:
        F = np.broadcast_to(np.eye(k), (T, k, k)).copy()
    else:
        phi_vec = np.broadcast_to(np.asarray(phi, float).reshape(-1), (k,))
        F = np.broadcast_to(np.diag(phi_vec), (T, k, k)).copy()

    Q = np.broadcast_to(q_scale * np.eye(k), (T, k, k)).copy()
    H = np.broadcast_to(np.zeros((1, k)), (T, 1, k)).copy()
    for t in range(T):
        H[t, 0, :] = X[t]
    R = np.broadcast_to(np.array([[r_scale]]), (T, 1, 1)).copy()

    out = kalman_filter_tv(y.reshape(T, 1), F, H, Q, R, a0=a0, P0=P0_scale * np.eye(k))
    sm = rts_smoother(out["a_pred"], out["P_pred"], out["a_filt"], out["P_filt"], F, Q)

    betas_sm = sm["a_smooth"]  # (T,k)

    if names is not None:
        import pandas as pd
        betas_sm = pd.DataFrame(betas_sm, columns=list(names))

    if return_dict:
        out_dict = {**out, **sm, "F": F, "H": H, "Q": Q, "R": R, "X": X, "y": y}
        return betas_sm, out_dict
    return betas_sm


# -----------------------------------------------------------------------------
# Visualization & interpretation helpers
# -----------------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt

def tvp_make_components(
    data_m: pd.DataFrame,
    betas_df: pd.DataFrame,
    factor_cols=("MKT_RF","SMB","HML","RMW","CMA","MOM"),
):
    """Return fitted excess (y_hat), per-factor contributions, residuals, and 24m rolling R^2.

    Expects data_m with 'Excess_Return' and the factor columns in decimals, and
    betas_df with columns ['alpha','beta_<FACTOR>', ...] aligned on the same index.
    """
    betas_df = betas_df.copy()
    betas_df.index = data_m.index

    X = data_m[list(factor_cols)].values
    Xc = pd.DataFrame(
        np.column_stack([np.ones(len(data_m)), X]),
        index=data_m.index,
        columns=["alpha"] + [f for f in factor_cols],
    )
    # Map beta column names to Xc columns
    # betas_df cols expected: ["alpha","beta_MKT", ...]; rename to match Xc
    rename_map = {"beta_MKT":"MKT_RF","beta_SMB":"SMB","beta_HML":"HML","beta_RMW":"RMW","beta_CMA":"CMA","beta_MOM":"MOM"}
    β = betas_df.rename(columns=rename_map)

    # Align columns order
    β = β[Xc.columns]

    contribs = Xc * β
    y_hat = contribs.sum(axis=1).rename("Fitted_Excess_TVP")
    resid = (data_m["Excess_Return"] - y_hat).rename("Residual")

    r2 = (y_hat.rolling(24).var() / data_m["Excess_Return"].rolling(24).var()).rename("R2_24m")
    return y_hat, contribs, resid, r2


def tvp_plot_betas(betas_df: pd.DataFrame, cols=None, ci_stds: pd.DataFrame=None, title: str=None):
    """Plot selected time-varying betas; optionally add ±1.96*std CI bands if ci_stds provided."""
    betas_df = betas_df.copy()
    if cols is None:
        cols = [c for c in betas_df.columns if c.startswith("beta_")]
    fig, ax = plt.subplots(figsize=(10,4))
    for c in cols:
        ax.plot(betas_df.index, betas_df[c], label=c)
        if ci_stds is not None and c in ci_stds.columns:
            lo = betas_df[c] - 1.96*ci_stds[c]
            hi = betas_df[c] + 1.96*ci_stds[c]
            ax.fill_between(betas_df.index, lo, hi, alpha=0.15, linewidth=0)
    ax.set_title(title or "Time-varying betas")
    ax.legend()
    ax.set_xlabel("Date"); ax.set_ylabel("Beta")
    plt.tight_layout()


def tvp_plot_contributions(contribs: pd.DataFrame, stack=True, title: str=None):
    """Plot per-factor contributions each month (alpha, MKT, SMB, HML, RMW, CMA, MOM)."""
    # Ensure consistent order
    order = [c for c in ["alpha","MKT_RF","SMB","HML","RMW","CMA","MOM"] if c in contribs.columns]
    C = contribs[order]
    fig, ax = plt.subplots(figsize=(10,4))
    if stack:
        ax.stackplot(C.index, [C[c].values for c in C.columns], labels=C.columns)
    else:
        for c in C.columns:
            ax.plot(C.index, C[c], label=c)
    ax.set_title(title or "Per-factor contributions to excess return")
    ax.legend(loc="best")
    ax.set_xlabel("Date"); ax.set_ylabel("Contribution")
    plt.tight_layout()


def tvp_plot_fit_vs_actual(y: pd.Series, y_hat: pd.Series, title: str=None):
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(y.index, y, label="Actual Excess")
    ax.plot(y_hat.index, y_hat, label="Fitted (TVP)")
    ax.set_title(title or "Actual vs Fitted (TVP)")
    ax.set_xlabel("Date"); ax.set_ylabel("Return")
    ax.legend()
    plt.tight_layout()


def tvp_plot_r2(r2: pd.Series, title: str=None):
    fig, ax = plt.subplots(figsize=(10,3))
    ax.plot(r2.index, r2)
    ax.set_title(title or "Rolling R^2 (24m)")
    ax.set_xlabel("Date"); ax.set_ylabel("R^2")
    plt.tight_layout()


def tvp_label_regimes(betas_df: pd.DataFrame) -> pd.Series:
    """Simple example: label regimes from beta paths. Customize thresholds as needed."""
    s = pd.Series("Neutral", index=betas_df.index)
    if "beta_MKT" in betas_df.columns:
        s[betas_df["beta_MKT"] > 1.5] = "High-beta"
        s[betas_df["beta_MKT"] < 0.7] = s[betas_df["beta_MKT"] < 0.7].where(s[betas_df["beta_MKT"] < 0.7] != "High-beta", other="High-beta")
    if "beta_HML" in betas_df.columns:
        s[betas_df["beta_HML"] < -0.3] = "Growth"
        s[betas_df["beta_HML"] > 0.3] = "Value"
    # combine: if both conditions true, prefer combined label
    both = (betas_df.get("beta_MKT",0) > 1.5) & (betas_df.get("beta_HML",0) < -0.3)
    s[both] = "High-beta Growth"
    return s.rename("Regime")


def tvp_plot_regimes(series: pd.Series, regimes: pd.Series, title: str=None):
    """Plot a single series with background spans indicating regimes (non-"Neutral")."""
    series = series.copy(); regimes = regimes.reindex(series.index).fillna("Neutral")
    fig, ax = plt.subplots(figsize=(10,3))
    ax.plot(series.index, series.values, label=series.name or "Series")
    # Shade regime segments
    current = None; start = None
    for t, lab in zip(series.index, regimes):
        if lab != current:
            if current not in (None, "Neutral") and start is not None:
                ax.axvspan(start, t, alpha=0.12)
            current, start = lab, t
    # Close last span
    if current not in (None, "Neutral") and start is not None:
        ax.axvspan(start, series.index[-1], alpha=0.12)
    ax.set_title(title or f"{series.name} with regimes")
    ax.set_xlabel("Date"); ax.set_ylabel(series.name or "Value")
    ax.legend(loc="best")
    plt.tight_layout()

# -----------------------------------------------------------------------------
# Tiny example (simulation) — remove or adapt for your project
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)

    T = 200
    n = 3   # state dim (e.g., NS betas)
    m = 5   # number of observed yields

    # State dynamics: AR(1) on each beta
    phi = np.array([0.98, 0.95, 0.90])
    F = np.stack([np.diag(phi)] * T)
    Q = np.stack([0.0001 * np.eye(n)] * T)
    c = np.zeros((T, n))

    # Observation: Nelson–Siegel loads
    maturities = np.array([0.25, 0.5, 1, 3, 5])
    lam = 0.73
    H0 = nelson_siegel_loadings(maturities, lam)  # (m,n)
    H = np.broadcast_to(H0, (T, m, n)).copy()
    d = np.zeros((T, m))
    R = np.stack([0.0004 * np.eye(m)] * T)

    # Simulate
    a = np.zeros((T, n))
    y = np.zeros((T, m))

    a_prev = np.zeros(n)
    for t in range(T):
        a_t = F[t] @ a_prev + rng.multivariate_normal(np.zeros(n), Q[t])
        y_t = d[t] + H[t] @ a_t + rng.multivariate_normal(np.zeros(m), R[t])
        a[t] = a_t
        y[t] = y_t
        a_prev = a_t

    # Introduce some missing obs
    y[50:60, 2] = np.nan

    # Prior
    a0 = np.zeros(n)
    P0 = 1.0 * np.eye(n)

    out = kalman_filter_tv(y, F, H, Q, R, a0, P0, c=c, d=d)
    sm = rts_smoother(out["a_pred"], out["P_pred"], out["a_filt"], out["P_filt"], F, Q)

    print("LogLik:", out["loglik"])  # just to show it runs
    print("Smoothed last state:", sm["a_smooth"][-1])
