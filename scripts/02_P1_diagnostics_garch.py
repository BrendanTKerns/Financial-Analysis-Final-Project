"""Part 1: stationarity, autocorrelation, and constant-mean GARCH(1,1).

Inputs and outputs are relative to the repository, regardless of working
directory. Units: log returns in percent, daily variance in percent squared.
"""
from pathlib import Path
import importlib.metadata
import json
import warnings

import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.tsa.stattools import adfuller, kpss, acf
from statsmodels.stats.diagnostic import acorr_ljungbox

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TRADING_DAYS = 252


def stationarity_row(ticker, label, values, regression="c"):
    """Opposite nulls: ADF has a unit root; KPSS is stationary.

    Preserve KPSS boundary warnings because 0.01/0.10 are table limits,
    not exact p-values. ADF uses AIC lag selection, KPSS automatic bandwidth.
    """
    values = values.dropna()
    adf = adfuller(values, regression=regression, autolag="AIC", result_object=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        kp = kpss(values, regression=regression, nlags="auto", result_object=False)
    bound = "="
    if kp[1] <= 0.01:
        bound = "<="
    elif kp[1] >= 0.10:
        bound = ">="
    stationary = adf[1] < .05 and kp[1] >= .05
    nonstationary = adf[1] >= .05 and kp[1] < .05
    decision = "supports stationarity" if stationary else "supports nonstationarity" if nonstationary else "mixed evidence"
    row = dict(ticker=ticker, series=label, regression=regression, n=len(values),
               adf_stat=adf[0], adf_p=adf[1], adf_lags=adf[2], adf_nobs=adf[3],
               kpss_stat=kp[0], kpss_p_bound=bound, kpss_p=kp[1], kpss_lags=kp[2],
               decision_5pct=decision, warnings=" | ".join(str(w.message) for w in caught))
    for name, value in adf[4].items():
        row["adf_critical_" + name] = value
    for name, value in kp[3].items():
        row["kpss_critical_" + name] = value
    return row


def fit_garch(values):
    """Gaussian quasi-MLE with robust standard errors; keep the required mean.

    rescale=False keeps mu in percent and omega in percent squared. Percent
    returns already have a suitable numerical scale. Reject failed fits.
    """
    model = arch_model(values, mean="Constant", vol="GARCH", p=1, q=1,
                       dist="normal", rescale=False)
    fit = model.fit(disp="off", cov_type="robust", options={"maxiter": 2000})
    if fit.convergence_flag != 0 or not np.isfinite(fit.params).all():
        raise RuntimeError(f"GARCH did not converge: {fit.optimization_result.message}")
    return fit


def risk_metrics(fit, values):
    omega = fit.params["omega"]
    persistence = fit.params["alpha[1]"] + fit.params["beta[1]"]
    finite_variance = 0 < persistence < 1
    variance = omega / (1 - persistence) if finite_variance else np.nan
    return {
        "persistence": persistence,
        "half_life_days": np.log(.5) / np.log(persistence) if finite_variance else np.nan,
        "long_run_variance_pct2": variance,
        "long_run_annual_vol_pct": np.sqrt(variance * TRADING_DAYS),
        "sample_annual_vol_pct": values.std(ddof=1) * np.sqrt(TRADING_DAYS),
    }


def main():
    RESULTS.mkdir(exist_ok=True)
    prices = pd.read_csv(ROOT / "data/daily_adjusted_prices.csv", index_col="Date", parse_dates=True)
    returns = pd.read_csv(ROOT / "data/daily_log_returns_pct.csv", index_col="Date", parse_dates=True)
    tests, correlations, parameters, summaries, sensitivities, diagnostics = [], [], [], [], [], []
    for ticker in prices:
        values = returns[ticker]
        transformations = {"price": prices[ticker], "log_price": np.log(prices[ticker]), "log_return_pct": values}
        for label, transformed in transformations.items():
            tests.append(stationarity_row(ticker, label, transformed))
        # A trend specification checks whether rejecting level stationarity
        # simply reflects a deterministic trend in the price/log-price series.
        for label in ["price", "log_price"]:
            tests.append(stationarity_row(ticker, label, transformations[label], regression="ct"))
        for label, transformed in {"returns": values, "squared_returns": values ** 2}.items():
            coefficients = acf(transformed, nlags=10, fft=True)
            for lag in range(1, 11):
                correlations.append(dict(ticker=ticker, series=label, lag=lag,
                                         acf=coefficients[lag], reference_band=1.96 / np.sqrt(len(values))))
        fit = fit_garch(values)
        for param in fit.params.index:
            parameters.append(dict(ticker=ticker, parameter=param, estimate=fit.params[param],
                                   robust_se=fit.std_err[param], p_value=fit.pvalues[param],
                                   ci_low=fit.conf_int().loc[param, "lower"],
                                   ci_high=fit.conf_int().loc[param, "upper"]))
        metrics = risk_metrics(fit, values)
        covariance = fit.param_cov.loc[["alpha[1]", "beta[1]"], ["alpha[1]", "beta[1]"]]
        metrics["persistence_se"] = np.sqrt(covariance.to_numpy().sum())
        metrics["persistence_ci_high"] = metrics["persistence"] + 1.96 * metrics["persistence_se"]
        metrics["persistence_ci_low"] = metrics["persistence"] - 1.96 * metrics["persistence_se"]
        metrics.update(ticker=ticker, n=len(values), convergence_flag=int(fit.convergence_flag),
                       loglikelihood=fit.loglikelihood, max_abs_return_pct=values.abs().max(),
                       extreme_date=str(values.abs().idxmax().date()))
        # These Ljung-Box p-values are descriptive. Standard white-noise
        # inference for raw returns is imperfect under heteroskedasticity.
        for label, transformed in {"returns": values, "squared_returns": values ** 2,
                                   "standardized_residuals": fit.std_resid,
                                   "squared_standardized_residuals": fit.std_resid ** 2}.items():
            lb = acorr_ljungbox(transformed, lags=[10], return_df=True).iloc[0]
            diagnostics.append(dict(ticker=ticker, series=label, lag=10,
                                    lb_stat=lb["lb_stat"], lb_pvalue=lb["lb_pvalue"]))
        metrics["mean_squared_acf_1_10"] = np.mean(acf(values ** 2, nlags=10, fft=True)[1:])
        # Only a sensitivity exercise: cap the lowest/highest 1% and refit.
        # The main estimate always uses the complete unmodified return series.
        low, high = values.quantile([.01, .99])
        capped = values.clip(lower=low, upper=high)
        capfit = fit_garch(capped)
        sensitivities.append(dict(ticker=ticker, capped_days=int((values != capped).sum()),
                                  lower_cap_pct=low, upper_cap_pct=high,
                                  **risk_metrics(capfit, capped)))
        summaries.append(metrics)
        pd.DataFrame({"return_pct": values, "conditional_vol_pct": fit.conditional_volatility,
                      "standardized_residual": fit.std_resid}).to_csv(RESULTS / f"{ticker}_fitted_series.csv", index_label="Date")
        (RESULTS / f"{ticker}_model.txt").write_text(str(fit.summary()), encoding="utf-8")
        print(ticker, {k: round(metrics[k], 4) for k in ["persistence", "half_life_days", "sample_annual_vol_pct", "long_run_annual_vol_pct"]})
    for name, rows in [("stationarity", tests), ("acf", correlations), ("garch_parameters", parameters),
                       ("risk_summary", summaries), ("winsorized_sensitivity", sensitivities), ("ljung_box", diagnostics)]:
        pd.DataFrame(rows).to_csv(RESULTS / (name + ".csv"), index=False)
    versions = {package: importlib.metadata.version(package) for package in ["numpy", "pandas", "scipy", "statsmodels", "arch", "matplotlib", "yfinance", "reportlab"]}
    (RESULTS / "package_versions.json").write_text(json.dumps(versions, indent=2), encoding="utf-8")
    print("Saved complete test tables, fitted series, robust parameter inference, and sensitivity results.")


if __name__ == "__main__":
    main()
