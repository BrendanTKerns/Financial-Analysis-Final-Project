"""Regenerate Part 1 figures from saved inputs and estimates.

Run 01_get_data.py, then 02_P1_diagnostics_garch.py before this file.
Each figure is saved as PNG for the report and SVG for scalable reuse.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
BLUE = "#18527b"
ORANGE = "#b94f1c"


def save(figure, name):
    figure.savefig(FIGURES / f"{name}.png", dpi=165, bbox_inches="tight")
    figure.savefig(FIGURES / f"{name}.svg", bbox_inches="tight")
    plt.close(figure)


def main():
    FIGURES.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": .18, "figure.facecolor": "white"})
    prices = pd.read_csv(ROOT / "data/daily_adjusted_prices.csv", index_col="Date", parse_dates=True)
    returns = pd.read_csv(ROOT / "data/daily_log_returns_pct.csv", index_col="Date", parse_dates=True)
    acfs = pd.read_csv(ROOT / "results/acf.csv")
    summary = pd.read_csv(ROOT / "results/risk_summary.csv").set_index("ticker")
    for ticker in prices:
        fig, axes = plt.subplots(3, 1, figsize=(9, 6.1), sharex=True, layout="constrained")
        axes[0].plot(prices.index, prices[ticker], color=BLUE, lw=.85)
        axes[0].set_ylabel("Adjusted price ($)")
        axes[0].set_title(f"{ticker} | Price, log price, and daily log returns", loc="left", weight="bold")
        axes[1].plot(prices.index, np.log(prices[ticker]), color=BLUE, lw=.85)
        axes[1].set_ylabel("Log price")
        axes[2].plot(returns.index, returns[ticker], color=BLUE, lw=.65)
        axes[2].axhline(0, color="black", lw=.5)
        axes[2].set_ylabel("Log return (%)")
        axes[2].set_xlabel("Trading date | January 2016 - October 2026")
        save(fig, f"P1_{ticker}_transformations")
        fig, axes = plt.subplots(1, 2, figsize=(9, 3), sharey=True, layout="constrained")
        for axis, label in zip(axes, ["returns", "squared_returns"]):
            rows = acfs[(acfs.ticker == ticker) & (acfs.series == label)]
            axis.bar(rows.lag, rows.acf, color=BLUE if label == "returns" else ORANGE, width=.6)
            band = rows.reference_band.iloc[0]
            axis.axhspan(-band, band, color="gray", alpha=.15)
            axis.axhline(0, color="black", lw=.6)
            axis.set_xticks(range(1, 11))
            axis.set_xlabel("Lag in trading days")
            axis.set_title(label.replace("_", " ").capitalize())
        axes[0].set_ylabel("Autocorrelation")
        fig.suptitle(f"{ticker} | ACF at lags 1-10", weight="bold")
        save(fig, f"P1_{ticker}_acf")
        fit = pd.read_csv(ROOT / f"results/{ticker}_fitted_series.csv", index_col="Date", parse_dates=True)
        fig, axis = plt.subplots(figsize=(9, 3.1), layout="constrained")
        axis.plot(fit.index, fit.conditional_vol_pct * np.sqrt(252), color=BLUE, lw=.9, label="Conditional volatility")
        long_run = summary.loc[ticker, "long_run_annual_vol_pct"]
        if np.isfinite(long_run):
            axis.axhline(long_run, color=ORANGE, lw=1, ls="--", label="Model long-run volatility")
        axis.set_title(f"{ticker} | GARCH volatility through time", loc="left", weight="bold")
        axis.set_ylabel("Annualized volatility (%)")
        axis.legend(frameon=False, fontsize=9)
        save(fig, f"P1_{ticker}_volatility")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    x = np.arange(len(summary))
    axes[0].bar(x-.18, summary.sample_annual_vol_pct, width=.36, color=BLUE, label="Sample")
    axes[0].bar(x+.18, summary.long_run_annual_vol_pct, width=.36, color=ORANGE, label="GARCH long-run")
    axes[0].set_ylabel("Annualized volatility (%)")
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].bar(x, summary.half_life_days, color=BLUE)
    axes[1].set_ylabel("Variance-shock half-life (trading days)")
    axes[2].bar(x, summary.mean_squared_acf_1_10, color=ORANGE)
    axes[2].set_ylabel("Mean squared-return ACF, lags 1-10")
    for axis in axes:
        axis.set_xticks(x, summary.index)
    fig.suptitle("Five stocks | Volatility level, shock duration, and clustering", weight="bold")
    save(fig, "P1_risk_comparison")
    print("Saved 16 Part 1 figures in PNG and SVG formats.")


if __name__ == "__main__":
    main()
