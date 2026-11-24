"""
fda_utils.py — the BUS 493A course-standard toolkit.

WHY THIS FILE EXISTS
--------------------
In 2024-25 yfinance changed two defaults that break almost every older finance
tutorial you will find online (including ones an AI will happily write for you):

  1. `auto_adjust` now defaults to True.  That means the `Close` column is
     ALREADY adjusted for splits and dividends, and there is NO `Adj Close`
     column any more.  Any code that says df["Adj Close"] raises KeyError.

  2. `yf.download(...)` returns a two-level (MultiIndex) column index, even for
     a single ticker: ('Close', 'AAPL'), ('High', 'AAPL'), ...
     So df["Close"] gives you back a DataFrame, not a Series, and arithmetic
     silently does something different from what you meant.

The `multi_level_index=False` argument flattens the columns, BUT ONLY WHEN YOU
REQUEST A SINGLE TICKER.  Ask for a list of tickers and the columns stay
MultiIndex.  That asymmetry is the trap: code that works on one ticker breaks
on three.

This module hides all of that behind ONE contract:

    get_prices(...)  and  get_returns(...)  ALWAYS return a DataFrame whose
    columns are the tickers you asked for, in the order you asked for them —
    whether you asked for 1 ticker or 20.

Import this in every problem set.  Do not re-invent the download call.

USAGE
-----
    from fda_utils import get_prices, get_returns, assign_tickers

    px  = get_prices("AAPL", "2020-01-01", "2026-09-01")     # DataFrame, 1 col
    px  = get_prices(["AAPL", "XOM"], "2020-01-01", "2026-09-01")  # 2 cols
    ret = get_returns(["AAPL", "XOM"], "2020-01-01", "2026-09-01", kind="log")

Tested on: Python 3.11, yfinance 0.2.65, pandas 2.1.4, numpy 1.26.4.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

__all__ = [
    "get_prices",
    "get_returns",
    "monthly_returns",
    "to_monthly",
    "assign_tickers",
    "FDA_UNIVERSE",
    "SECTORS",
    "load_ff_factors",
    "annualize_return",
    "annualize_vol",
    "sharpe_ratio",
    "max_drawdown",
    "summarize_performance",
    "cache_dir",
    "clear_cache",
    "TRADING_DAYS",
]

TRADING_DAYS = 252

# --------------------------------------------------------------------------
# Local cache
# --------------------------------------------------------------------------
# Twenty students hitting Yahoo at the same moment in a 9:30am lab will get
# rate-limited.  We cache each (ticker, start, end, price) pull as a CSV next
# to this file so a re-run is instant and works offline.  Delete the folder
# (or call clear_cache()) whenever you want a genuinely fresh pull.


def cache_dir() -> Path:
    """Folder used for the on-disk price cache. Created on first use."""
    d = Path(__file__).resolve().parent / "_cache"
    d.mkdir(exist_ok=True)
    return d


def clear_cache() -> int:
    """Delete every cached CSV. Returns how many files were removed."""
    n = 0
    for f in cache_dir().glob("*.csv"):
        f.unlink()
        n += 1
    return n


def _cache_path(tickers: list[str], start: str, end: str, price: str) -> Path:
    key = f"{'-'.join(tickers)}_{start}_{end}_{price}".replace("^", "IDX")
    if len(key) > 120:  # long ticker lists -> stable short name
        key = f"{key[:80]}_{abs(hash(key)) % 10**8}"
    return cache_dir() / f"{key}.csv"


# --------------------------------------------------------------------------
# The two functions you actually call
# --------------------------------------------------------------------------


def _as_list(tickers) -> list[str]:
    if isinstance(tickers, str):
        return [tickers.strip().upper()]
    return [str(t).strip().upper() for t in tickers]


def get_prices(tickers, start, end, price="Close", use_cache=True) -> pd.DataFrame:
    """Adjusted daily prices as a DataFrame with one column per ticker.

    Parameters
    ----------
    tickers : str or sequence of str
        "AAPL" or ["AAPL", "XOM", "JPM"]. Case-insensitive.
    start, end : str
        "YYYY-MM-DD". `end` is EXCLUSIVE, the same as yfinance itself.
    price : {"Close", "Open", "High", "Low", "Volume"}
        "Close" is already split- and dividend-adjusted because we pass
        auto_adjust=True. There is no "Adj Close" any more — that is the point.
    use_cache : bool
        Read/write a local CSV cache. Set False to force a fresh download.

    Returns
    -------
    pd.DataFrame
        Index = DatetimeIndex of trading days. Columns = the tickers you asked
        for, in that order. ALWAYS a DataFrame, never a Series.

    Raises
    ------
    ValueError
        If Yahoo returned nothing at all (bad ticker, dead ticker, no network,
        or a date range with no trading days). Failing loudly here is
        deliberate: a silent empty frame produces nonsense 200 lines later.
    """
    tickers = _as_list(tickers)
    path = _cache_path(tickers, str(start), str(end), price)

    if use_cache and path.exists():
        out = pd.read_csv(path, index_col=0, parse_dates=True)
        # A cache written from a single-column pull can come back with the
        # column named differently; trust the requested order instead.
        out.columns = [str(c).upper() for c in out.columns]
        missing = [t for t in tickers if t not in out.columns]
        if not missing:
            return out[tickers]
        # cache is stale/partial -> fall through and re-download

    with warnings.catch_warnings():
        # yfinance emits a FutureWarning about the auto_adjust default even
        # when you pass it explicitly on some versions. We pass it explicitly,
        # so the warning is noise.
        warnings.simplefilter("ignore", FutureWarning)
        raw = yf.download(
            tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            multi_level_index=False,
            group_by="column",
        )

    if raw is None or len(raw) == 0:
        raise ValueError(
            f"yfinance returned no data for {tickers} between {start} and {end}.\n"
            "Check: (a) is the ticker still listed? (b) is the date range valid "
            "and does it contain trading days? (c) are you online?"
        )

    # --- normalise to one column per ticker, whatever shape came back -------
    if isinstance(raw.columns, pd.MultiIndex):
        # Multi-ticker pull: level 0 = field, level 1 = ticker.
        if price not in raw.columns.get_level_values(0):
            raise ValueError(
                f"Field {price!r} not present. Available: "
                f"{sorted(set(raw.columns.get_level_values(0)))}"
            )
        out = raw[price].copy()
    else:
        # Single-ticker pull flattened by multi_level_index=False:
        # columns are Close/High/Low/Open/Volume with no ticker anywhere.
        if price not in raw.columns:
            raise ValueError(
                f"Field {price!r} not present. Available: {list(raw.columns)}"
            )
        out = raw[[price]].copy()
        out.columns = tickers[: out.shape[1]]

    out = out.dropna(how="all")
    if out.empty:
        raise ValueError(
            f"All rows were NaN for {tickers} between {start} and {end}."
        )

    # Guarantee the promised column set and order.
    out.columns = [str(c).upper() for c in out.columns]
    absent = [t for t in tickers if t not in out.columns]
    if absent:
        raise ValueError(
            f"Yahoo returned no {price} series for: {absent}. "
            "Most often this means the ticker is delisted or misspelled."
        )
    out = out[tickers]
    out.index.name = "Date"

    if use_cache:
        try:
            out.to_csv(path)
        except OSError:
            pass  # a read-only folder must not break a student's homework

    return out


def get_returns(tickers, start, end, kind="log", price="Close",
                use_cache=True) -> pd.DataFrame:
    """Daily returns as a DataFrame with one column per ticker.

    kind="log"    -> r_t = ln(P_t / P_{t-1})     (additive across time)
    kind="simple" -> r_t = P_t / P_{t-1} - 1     (additive across assets)

    Which one you want is a judgement call you will be asked to defend:
    log returns aggregate cleanly over TIME (sum of daily log returns is the
    log return of the period), simple returns aggregate cleanly across ASSETS
    (a portfolio's simple return is the weighted mean of its holdings').
    """
    if kind not in {"log", "simple"}:
        raise ValueError(f"kind must be 'log' or 'simple', got {kind!r}")

    px = get_prices(tickers, start, end, price=price, use_cache=use_cache)
    if kind == "log":
        rets = np.log(px / px.shift(1))
    else:
        rets = px.pct_change(fill_method=None)
    return rets.dropna(how="all")


# --------------------------------------------------------------------------
# Personalised data slices (every student works on different tickers)
# --------------------------------------------------------------------------

FDA_UNIVERSE: dict[str, list[str]] = {
    "Communication Services": ["GOOGL", "META", "NFLX", "DIS", "CMCSA",
                               "T", "VZ", "TMUS", "EA", "WBD"],
    "Consumer Discretionary": ["AMZN", "TSLA", "HD", "MCD", "NKE",
                               "SBUX", "LOW", "TJX", "BKNG", "ORLY"],
    "Consumer Staples":       ["PG", "KO", "PEP", "COST", "WMT",
                               "MDLZ", "CL", "KMB", "GIS", "KHC"],
    "Energy":                 ["XOM", "CVX", "COP", "SLB", "EOG",
                               "PSX", "MPC", "VLO", "OXY", "WMB"],
    "Financials":             ["JPM", "BAC", "WFC", "GS", "MS",
                               "BLK", "SCHW", "AXP", "C", "USB"],
    "Health Care":            ["JNJ", "UNH", "LLY", "PFE", "ABBV",
                               "MRK", "TMO", "ABT", "AMGN", "GILD"],
    "Industrials":            ["CAT", "HON", "UPS", "BA", "GE",
                               "DE", "LMT", "RTX", "UNP", "MMM"],
    "Information Technology": ["AAPL", "MSFT", "NVDA", "AVGO", "CRM",
                               "ADBE", "ORCL", "AMD", "INTC", "CSCO"],
    "Materials":              ["LIN", "APD", "SHW", "ECL", "NEM",
                               "FCX", "DOW", "DD", "PPG", "NUE"],
    "Utilities":              ["NEE", "DUK", "SO", "D", "AEP",
                               "SRE", "EXC", "XEL", "ED", "WEC"],
}

SECTORS: list[str] = sorted(FDA_UNIVERSE)  # fixed, alphabetical, 10 sectors


def assign_tickers(student_id, n=3, as_frame=False):
    """Your personal tickers for this course, derived from your student ID.

    THE RULE (you can and should check this by hand):
      Let d  = the LAST TWO DIGITS of your UPS student ID, as a number 00-99.
      Let tens = d // 10  and  ones = d % 10.
      Let step = max(1, 10 // n).
      For j = 0, 1, ..., n-1:
          sector j  = SECTORS[(ones + j*step) % 10]        (alphabetical list)
          ticker j  = FDA_UNIVERSE[sector j][(tens + j) % 10]

    Because `step` is chosen so that j*step never repeats mod 10 for the values
    of n we use, you always get n DIFFERENT sectors. That matters: a
    single-sector basket is one bet wearing n hats, and several questions in
    this course only make sense across sectors.

    Parameters
    ----------
    student_id : str or int
        Your full student ID. Only the last two digits are used, so
        "1234567" and 67 give the same answer.
    n : int
        How many tickers the problem set asks for (1, 3, or 5 in this course).
    as_frame : bool
        True -> a DataFrame with sector labels, handy for pasting into a report.

    Returns
    -------
    list[str] (default) or pd.DataFrame
    """
    if not 1 <= n <= 10:
        raise ValueError("n must be between 1 and 10")

    digits = "".join(ch for ch in str(student_id) if ch.isdigit())
    if len(digits) < 2:
        raise ValueError(
            f"Could not read two digits from student_id={student_id!r}. "
            "Pass your full student ID, e.g. assign_tickers('1234567', n=3)."
        )
    d = int(digits[-2:])
    tens, ones = d // 10, d % 10
    step = max(1, 10 // n)

    rows = []
    for j in range(n):
        sector = SECTORS[(ones + j * step) % 10]
        ticker = FDA_UNIVERSE[sector][(tens + j) % 10]
        rows.append((ticker, sector))

    if as_frame:
        return pd.DataFrame(rows, columns=["ticker", "sector"])
    return [t for t, _ in rows]


# --------------------------------------------------------------------------
# Fama-French factors from the local CSV (no pandas_datareader, no !wget)
# --------------------------------------------------------------------------


def to_monthly(df, how="last") -> pd.DataFrame:
    """Resample a daily series/frame to month-END observations.

    Use this instead of `.resample("M")` or `.resample("ME")`.

    Why: pandas renamed the month-end alias from "M" to "ME". Version 2.1
    accepts only "M"; version 3.0 accepts only "ME". Code written with either
    string breaks on the other. Passing the OFFSET OBJECT `pd.offsets.MonthEnd()`
    works on every version, so that is what this helper does. You will meet this
    again — pandas frequency aliases are a common source of AI-written code that
    ran fine last year and errors today.
    """
    rule = pd.offsets.MonthEnd()
    grouped = df.resample(rule)
    if how == "last":
        return grouped.last()
    if how == "sum":
        return grouped.sum()
    if how == "mean":
        return grouped.mean()
    raise ValueError("how must be 'last', 'sum' or 'mean'")


def monthly_returns(tickers, start, end, kind="simple", use_cache=True):
    """Month-end simple (or log) returns, aligned to Fama-French month-ends.

    Fama-French factors are published monthly. Regressing DAILY stock returns
    on MONTHLY factors — by merging and forward-filling — is a real and common
    mistake: it repeats each factor value ~21 times and destroys the standard
    errors. Convert the stock side to monthly instead, with this.
    """
    px = get_prices(tickers, start, end, use_cache=use_cache)
    m = to_monthly(px, how="last")
    rets = np.log(m / m.shift(1)) if kind == "log" else m.pct_change(fill_method=None)
    return rets.dropna(how="all")


_FF_URL = {
    "monthly": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
               "F-F_Research_Data_Factors_CSV.zip",
    "daily": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
             "F-F_Research_Data_Factors_daily_CSV.zip",
}


def _parse_ff_text(text: str, freq: str) -> pd.DataFrame:
    """Parse Ken French's CSV layout: preamble, header, data block(s)."""
    width = 6 if freq == "monthly" else 8      # YYYYMM vs YYYYMMDD
    fmt = "%Y%m" if freq == "monthly" else "%Y%m%d"

    rows = []
    started = False
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if not started:
            # The header row is the one naming the factors.
            if "Mkt-RF" in line:
                started = True
            continue
        if len(parts) < 5 or not parts[0].isdigit():
            if rows:
                break        # blank line after the block -> annual block next
            continue
        if len(parts[0]) != width:
            break            # 4-digit years = the annual block; stop
        rows.append(parts[:5])

    if not rows:
        raise ValueError("Could not find a Fama-French data block in this file.")

    ff = pd.DataFrame(rows, columns=["date", "Mkt-RF", "SMB", "HML", "RF"])
    ff["date"] = pd.to_datetime(ff["date"], format=fmt)
    if freq == "monthly":
        ff["date"] += pd.offsets.MonthEnd(0)
    # French publishes these in PERCENT. Convert to decimals so they can be
    # compared directly with returns from get_returns().
    ff = ff.set_index("date").astype(float) / 100.0
    ff.index.name = "Date"
    return ff


def load_ff_factors(freq="monthly", refresh=False, path=None) -> pd.DataFrame:
    """Fama-French 3 factors, fetched fresh from Ken French's data library.

    Parameters
    ----------
    freq : {"monthly", "daily"}
    refresh : bool
        True re-downloads even if a cached copy exists.
    path : str or Path, optional
        Parse this local CSV instead of downloading. Use for offline work.

    Returns
    -------
    pd.DataFrame indexed by date with columns ['Mkt-RF','SMB','HML','RF'],
    expressed as DECIMALS (0.0123 = 1.23%), matching get_returns().

    Notes
    -----
    The copy of `F-F_Research_Data_Factors.CSV` that shipped in the Lectures
    folder stops in **November 2019**. It is kept only as an offline fallback;
    do not build a 2026 analysis on it. If you see the stale-data warning,
    you are offline and your sample is truncated.
    """
    if freq not in _FF_URL:
        raise ValueError("freq must be 'monthly' or 'daily'")

    if path is not None:
        return _parse_ff_text(Path(path).read_text(errors="ignore"), freq)

    cached = cache_dir() / f"ff3_{freq}.csv"
    if cached.exists() and not refresh:
        ff = pd.read_csv(cached, index_col=0, parse_dates=True)
        ff.index.name = "Date"
        return ff

    import io
    import urllib.request
    import zipfile

    try:
        req = urllib.request.Request(
            _FF_URL[freq], headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=45) as resp:
            blob = resp.read()
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            text = z.read(z.namelist()[0]).decode("latin-1")
        ff = _parse_ff_text(text, freq)
        try:
            ff.to_csv(cached)
        except OSError:
            pass
        return ff
    except Exception as exc:                      # offline / site down / blocked
        here = Path(__file__).resolve().parent
        fallback = next(
            (c for c in [
                here / "F-F_Research_Data_Factors.CSV",
                here.parent / "Lectures"
                / "4. Asset Pricing Models and Portfolio Allocation"
                / "F-F_Research_Data_Factors.CSV",
            ] if c.exists()),
            None,
        )
        if fallback is None or freq != "monthly":
            raise RuntimeError(
                f"Could not download Fama-French {freq} factors ({exc}) and no "
                "local fallback file was found."
            ) from exc
        warnings.warn(
            f"Could not reach Ken French's data library ({exc}). Falling back "
            f"to the local CSV, WHICH ENDS IN NOVEMBER 2019. Any result you "
            f"report from this is on a stale sample — say so, or get online "
            f"and re-run.",
            stacklevel=2,
        )
        return _parse_ff_text(fallback.read_text(errors="ignore"), "monthly")


# --------------------------------------------------------------------------
# Performance statistics (shared by PS4, PS7 and the project)
# --------------------------------------------------------------------------


def annualize_return(returns, kind="log", periods=TRADING_DAYS) -> float:
    """Annualised (geometric) return from a daily return series."""
    r = pd.Series(returns).dropna()
    if r.empty:
        return np.nan
    if kind == "log":
        return float(np.expm1(r.mean() * periods))
    return float((1.0 + r).prod() ** (periods / len(r)) - 1.0)


def annualize_vol(returns, periods=TRADING_DAYS) -> float:
    """Annualised volatility = daily sd * sqrt(252).

    The sqrt(252) scaling assumes returns are serially uncorrelated. They are
    not, exactly — which is why annualised vol is an estimate, not a fact.
    """
    r = pd.Series(returns).dropna()
    return float(r.std(ddof=1) * np.sqrt(periods)) if len(r) > 1 else np.nan


def sharpe_ratio(returns, rf_annual=0.0, kind="log",
                 periods=TRADING_DAYS) -> float:
    """Annualised Sharpe ratio, net of a constant annual risk-free rate."""
    r = pd.Series(returns).dropna()
    if len(r) < 2:
        return np.nan
    vol = annualize_vol(r, periods)
    if vol == 0 or np.isnan(vol):
        return np.nan
    return float((annualize_return(r, kind=kind, periods=periods) - rf_annual) / vol)


def max_drawdown(returns, kind="log") -> float:
    """Worst peak-to-trough decline of the cumulative wealth curve (negative)."""
    r = pd.Series(returns).dropna()
    if r.empty:
        return np.nan
    wealth = np.exp(r.cumsum()) if kind == "log" else (1.0 + r).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


def summarize_performance(returns, rf_annual=0.0, kind="log") -> pd.Series:
    """One-line performance summary used across PS4, PS7 and the project."""
    r = pd.Series(returns).dropna()
    return pd.Series(
        {
            "ann_return": annualize_return(r, kind=kind),
            "ann_vol": annualize_vol(r),
            "sharpe": sharpe_ratio(r, rf_annual=rf_annual, kind=kind),
            "max_drawdown": max_drawdown(r, kind=kind),
            "n_obs": int(len(r)),
        }
    )


if __name__ == "__main__":
    # Smoke test: run `python fda_utils.py` to check your install.
    print("fda_utils smoke test")
    print("-" * 60)
    tk = assign_tickers("1234567", n=3, as_frame=True)
    print("assign_tickers('1234567', n=3):")
    print(tk.to_string(index=False))
    px = get_prices(list(tk["ticker"]), "2023-01-01", "2024-01-01")
    print(f"\nprices  shape={px.shape}  cols={list(px.columns)}")
    rt = get_returns(list(tk["ticker"]), "2023-01-01", "2024-01-01")
    print(f"returns shape={rt.shape}")
    print("\nper-ticker summary:")
    print(rt.apply(summarize_performance).round(4).to_string())
    print("\nOK — your environment is working.")
