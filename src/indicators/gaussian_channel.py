"""
Gaussian Channel indicator (converted from Pine Script) - DonovanWall

This module implements a pandas-based version of the TradingView Pine script
`Gaussian Channel [DW]`. It approximates the same filter logic and outputs
filter, upper/lower bands, and simple color tags for visualization.

Usage:
    import pandas as pd
    from src.indicators.gaussian_channel import gaussian_channel

    df = pd.DataFrame({
        'high': ..., 'low': ..., 'close': ...
    })

    out = gaussian_channel(df)
    # out is a dict with keys: 'filt','hband','lband','fcolor','barcolor'

Notes:
- This is a best-effort conversion from Pine Script (series-based IIR filters)
  to a pandas implementation using explicit recursion over rows.
- The `lag` used for reduced-lag mode is rounded to the nearest integer.

"""

from typing import Dict, Optional
import math
import numpy as np
import pandas as pd


def _tr(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True Range series like Pine's `tr(true)`."""
    prev_close = close.shift(1)
    a = high - low
    b = (high - prev_close).abs()
    c = (low - prev_close).abs()
    tr = pd.concat([a, b, c], axis=1).max(axis=1)
    tr.iloc[0] = (high.iloc[0] - low.iloc[0])
    return tr


def _nz(x, default=0.0):
    return default if pd.isna(x) else x


def f_filt9x(alpha: float, s: pd.Series, i: int) -> pd.Series:
    """Compute the i-pole Gaussian-like filter over series `s` using alpha.

    This implements the recursive formula from the original Pine script in a
    row-by-row loop (IIR). Returns a pandas Series of same length as `s`.
    """
    n = len(s)
    f = np.zeros(n, dtype=float)
    x = 1.0 - alpha

    # Precompute integer weights m2..m9 based on pole count i
    def m2(i):
        return 36 if i == 9 else 28 if i == 8 else 21 if i == 7 else 15 if i == 6 else 10 if i == 5 else 6 if i == 4 else 3 if i == 3 else 1 if i == 2 else 0
    def m3(i):
        return 84 if i == 9 else 56 if i == 8 else 35 if i == 7 else 20 if i == 6 else 10 if i == 5 else 4 if i == 4 else 1 if i == 3 else 0
    def m4(i):
        return 126 if i == 9 else 70 if i == 8 else 35 if i == 7 else 15 if i == 6 else 5 if i == 5 else 1 if i == 4 else 0
    def m5(i):
        return 126 if i == 9 else 56 if i == 8 else 21 if i == 7 else 6 if i == 6 else 1 if i == 5 else 0
    def m6(i):
        return 84 if i == 9 else 28 if i == 8 else 7 if i == 7 else 1 if i == 6 else 0
    def m7(i):
        return 36 if i == 9 else 8 if i == 8 else 1 if i == 7 else 0
    def m8(i):
        return 9 if i == 9 else 1 if i == 8 else 0
    def m9(i):
        return 1 if i == 9 else 0

    m2v = m2(i)
    m3v = m3(i)
    m4v = m4(i)
    m5v = m5(i)
    m6v = m6(i)
    m7v = m7(i)
    m8v = m8(i)
    m9v = m9(i)

    for t in range(n):
        s_t = _nz(s.iat[t], 0.0)
        # helper to get prior f with nz
        def f_prev(k):
            if t - k < 0:
                return 0.0
            return f[t - k]

        # compute the recursive polynomial-like formula matching Pine code
        term = (math.pow(alpha, i) * s_t)
        term += i * x * f_prev(1)
        if i >= 2:
            term -= m2v * math.pow(x, 2) * f_prev(2)
        if i >= 3:
            term += m3v * math.pow(x, 3) * f_prev(3)
        if i >= 4:
            term -= m4v * math.pow(x, 4) * f_prev(4)
        if i >= 5:
            term += m5v * math.pow(x, 5) * f_prev(5)
        if i >= 6:
            term -= m6v * math.pow(x, 6) * f_prev(6)
        if i >= 7:
            term += m7v * math.pow(x, 7) * f_prev(7)
        if i >= 8:
            term -= m8v * math.pow(x, 8) * f_prev(8)
        if i == 9:
            term += m9v * math.pow(x, 9) * f_prev(9)

        f[t] = term

    return pd.Series(f, index=s.index)


def f_pole(alpha: float, s: pd.Series, N: int):
    """Return (filtn, filt1) where filtn is the N-pole filter and filt1 the 1-pole."""
    filtn = f_filt9x(alpha, s, N)
    filt1 = f_filt9x(alpha, s, 1)
    return filtn, filt1


def gaussian_channel(
    df: Optional[pd.DataFrame] = None,
    source: Optional[pd.Series] = None,
    per: int = 144,
    N: int = 4,
    mult: float = 1.414,
    mode_lag: bool = False,
    mode_fast: bool = False,
) -> Dict[str, pd.Series]:
    """Compute Gaussian Channel outputs.

    Parameters:
        df: DataFrame containing `high`, `low`, `close` (optional if `source` provided)
        source: Series used as input (overrides df if provided). If not provided,
                `hlc3 = (high+low+close)/3` from `df` will be used.
        per: Sampling period (default 144)
        N: number of poles (1..9)
        mult: filtered true range multiplier
        mode_lag: reduced lag mode
        mode_fast: fast response mode

    Returns dictionary with keys: `filt`, `hband`, `lband`, `fcolor`, `barcolor`,
    and also `filtntr`/`filttr` for convenience.
    """
    if source is None:
        if df is None:
            raise ValueError("Either df or source must be provided")
        if not all(col in df.columns for col in ("high", "low", "close")):
            raise ValueError("df must contain 'high','low','close' columns when source is not provided")
        source = (df["high"] + df["low"] + df["close"]) / 3.0

    # create working copies
    src = source.astype(float)

    # components for beta/alpha
    beta = (1 - math.cos(4 * math.asin(1) / per)) / (math.pow(1.414, 2.0 / N) - 1)
    alpha = -beta + math.sqrt(beta * beta + 2 * beta)

    # lag as integer (rounded)
    lag = int(round((per - 1) / (2 * N)))

    # srcdata and trdata
    if df is not None:
        tr_series = _tr(df["high"], df["low"], df["close"])
    else:
        # Without high/low/close we cannot compute TR, use zeros
        tr_series = pd.Series(0.0, index=src.index)

    if mode_lag:
        # use shifted values by lag
        src_shift = src.shift(lag).fillna(method=None).fillna(0.0)
        srcdata = src + (src - src_shift)

        tr_shift = tr_series.shift(lag).fillna(0.0)
        trdata = tr_series + (tr_series - tr_shift)
    else:
        srcdata = src
        trdata = tr_series

    # Filtered values
    filtn, filt1 = f_pole(alpha, srcdata, N)
    filtntr, filt1tr = f_pole(alpha, trdata, N)

    # Lag reduction / fast mode
    if mode_fast:
        filt = (filtn + filt1) / 2.0
        filttr = (filtntr + filt1tr) / 2.0
    else:
        filt = filtn
        filttr = filtntr

    # Bands
    hband = filt + filttr * mult
    lband = filt - filttr * mult

    # Colors (hex strings) matching original script
    COLOR_UP = "#0aff68"
    COLOR_UP_DARK = "#00752d"
    COLOR_DOWN = "#ff0a5a"
    COLOR_DOWN_DARK = "#990032"
    COLOR_NEUTRAL = "#cccccc"
    COLOR_HIGHLIGHT = "#0aff1b"
    COLOR_DOWN_LOW = "#ff0a11"

    prev_filt = filt.shift(1)

    fcolor = pd.Series(COLOR_NEUTRAL, index=filt.index)
    fcolor[filt > prev_filt] = COLOR_UP
    fcolor[filt < prev_filt] = COLOR_DOWN

    # barcolor logic
    src_prev = src.shift(1)
    barcolor = pd.Series(COLOR_NEUTRAL, index=src.index)

    cond1 = (src > src_prev) & (src > filt) & (src < hband)
    cond2 = (src > src_prev) & (src >= hband)
    cond3 = (src <= src_prev) & (src > filt)
    cond4 = (src < src_prev) & (src < filt) & (src > lband)
    cond5 = (src < src_prev) & (src <= lband)
    cond6 = (src >= src_prev) & (src < filt)

    barcolor[cond1] = COLOR_UP
    barcolor[cond2] = COLOR_HIGHLIGHT
    barcolor[cond3] = COLOR_UP_DARK
    barcolor[cond4] = COLOR_DOWN
    barcolor[cond5] = COLOR_DOWN_LOW
    barcolor[cond6] = COLOR_DOWN_DARK

    return {
        "gaussian_channel_filt": filt,
        "gaussian_channel_hband": hband,
        "gaussian_channel_lband": lband,
        "gaussian_channel_fcolor": fcolor,
        "gaussian_channel_barcolor": barcolor,
        "gaussian_channel_filtntr": filtntr,
        "gaussian_channel_filttr": filttr,
    }


if __name__ == "__main__":
    # quick smoke test when run as script (requires pandas)
    import sys

    print("Gaussian Channel module loaded. Run gaussian_channel(df) with a DataFrame containing high/low/close.")
    if len(sys.argv) > 1:
        print("No command-line runner provided; this module is intended for import.")