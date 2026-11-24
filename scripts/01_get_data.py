"""Prepare the frozen Yahoo Finance price snapshot for the BUS 493 project.

Run from any directory. Offline by default; --refresh explicitly downloads a
new snapshot with the supplied course helper. The main cutoff stays fixed.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STOCKS = {
    "MSFT": "Information Technology",
    "JPM": "Financials",
    "XOM": "Energy",
    "COST": "Consumer Staples",
    "JNJ": "Health Care",
}
START = "2016-01-01"
END = "2026-11-01"  # Exclusive, matching the assignment's main cutoff.
SOURCE_START = "2015-12-01"  # Prior closing prices are needed for first returns.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    data_dir = ROOT / "data"
    source_dir = data_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    print(pd.Series(STOCKS, name="Sector").rename_axis("Ticker").to_string())
    series = []
    hashes = {}
    for ticker in STOCKS:
        path = source_dir / f"{ticker}_{SOURCE_START}_{END}_Close.csv"
        if args.refresh:
            import yfinance as yf
            from fda_utils import get_prices
            yf.set_tz_cache_location(str(data_dir / "yf_cache"))
            fresh = get_prices(ticker, SOURCE_START, END, use_cache=False)
            if fresh.empty or fresh.isna().any().any():
                raise ValueError(f"Invalid downloaded prices for {ticker}")
            fresh.to_csv(path, index_label="Date")
        prices = pd.read_csv(path, index_col="Date", parse_dates=True)
        if list(prices.columns) != [ticker]:
            raise ValueError(f"Unexpected columns in {path.name}")
        if not prices.index.is_unique or not prices.index.is_monotonic_increasing:
            raise ValueError(f"Dates must be unique and sorted for {ticker}")
        if prices.empty or not np.isfinite(prices.values).all() or (prices <= 0).any().any():
            raise ValueError(f"Missing, nonfinite, or nonpositive prices for {ticker}")
        series.append(prices[ticker])
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    all_prices = pd.concat(series, axis=1)
    if all_prices.isna().any().any():
        raise ValueError("Stock trading dates differ. Investigate before calculating returns.")
    all_prices = all_prices.loc[(all_prices.index >= SOURCE_START) & (all_prices.index < END)]
    prices = all_prices.loc[all_prices.index >= START]
    # Calculate changes BEFORE trimming, so the first trading day uses the prior close.
    simple = all_prices.pct_change(fill_method=None).loc[prices.index]
    log_pct = (100 * np.log(all_prices / all_prices.shift(1))).loc[prices.index]
    monthly = all_prices.resample("ME").last().pct_change(fill_method=None)
    monthly = monthly.loc[(monthly.index >= START) & (monthly.index < END)]
    if simple.isna().any().any() or log_pct.isna().any().any():
        raise ValueError("A prior close is missing at the start of the sample.")
    prices.to_csv(data_dir / "daily_adjusted_prices.csv", index_label="Date")
    simple.to_csv(data_dir / "daily_simple_returns.csv", index_label="Date")
    log_pct.to_csv(data_dir / "daily_log_returns_pct.csv", index_label="Date")
    monthly.to_csv(data_dir / "monthly_simple_returns.csv", index_label="Date")
    audit = {
        "source": "Yahoo Finance adjusted Close, downloaded via course fda_utils.py / yfinance",
        "snapshot_origin": "Proposal work on 2026-11-11; rerun with --refresh to replace source files",
        "source_start": SOURCE_START,
        "analysis_start": START,
        "cutoff_exclusive": END,
        "first_observation": str(prices.index.min().date()),
        "last_observation": str(prices.index.max().date()),
        "daily_observations": len(prices),
        "monthly_observations": len(monthly),
        "missing_values": int(prices.isna().sum().sum()),
        "trading_dates": "Identical across five stocks; exchange-holiday completeness not independently audited",
        "source_sha256": hashes,
        "sectors": STOCKS,
        "return_convention": "Part 1 uses 100 * log(P[t]/P[t-1]); monthly simple returns saved for Parts 2/3",
    }
    if args.refresh:
        audit["snapshot_origin"] = f"Refreshed {pd.Timestamp.now(tz='UTC').isoformat()}"
    (data_dir / "provenance.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"Saved {len(prices)} daily rows, {len(monthly)} monthly rows; last date {prices.index[-1].date()}")


if __name__ == "__main__":
    main()
