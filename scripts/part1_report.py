"""Build the explanatory Part 1 PDF from the saved analysis tables and figures."""
from pathlib import Path
from xml.sax.saxutils import escape
import json

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image

ROOT = Path(__file__).resolve().parents[1]
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="Body", fontName="Helvetica", fontSize=10, leading=14, spaceAfter=9))
styles.add(ParagraphStyle(name="SmallNote", fontName="Helvetica", fontSize=8.5, leading=11, spaceAfter=7))
styles.add(ParagraphStyle(name="ReportTitle", fontName="Helvetica-Bold", fontSize=23, leading=27, spaceAfter=13))
styles.add(ParagraphStyle(name="Section", fontName="Helvetica-Bold", fontSize=16, leading=20, spaceAfter=12))
styles.add(ParagraphStyle(name="Sub", fontName="Helvetica-Bold", fontSize=11, leading=14, spaceAfter=6))
story = []


def paragraph(text, style="Body"):
    story.append(Paragraph(escape(text), styles[style]))


def table(rows, widths=None):
    wrapped = [[Paragraph(escape(str(value)), styles["SmallNote"]) for value in row] for row in rows]
    obj = Table(wrapped, colWidths=widths, repeatRows=1, hAlign="LEFT")
    obj.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e4edf3")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6f7")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.extend([obj, Spacer(1, 10)])


def figure(name, width=500):
    from PIL import Image as PILImage
    path = ROOT / "figures" / f"{name}.png"
    with PILImage.open(path) as img:
        height = width * img.height / img.width
    story.append(Image(str(path), width=width, height=height, hAlign="LEFT"))
    story.append(Spacer(1, 7))


def pvalue(value):
    return "<0.0001" if value < .0001 else f"{value:.4f}"


def page(title):
    if story:
        story.append(PageBreak())
    paragraph(title, "Section")


def footer(canvas, doc):
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#56616b"))
    canvas.drawString(48, 25, "BUS 493 | Part 1 | Main cutoff November 1, 2026")
    canvas.drawRightString(564, 25, str(doc.page))


def main():
    risk = pd.read_csv(ROOT / "results/risk_summary.csv").set_index("ticker")
    tests = pd.read_csv(ROOT / "results/stationarity.csv")
    params = pd.read_csv(ROOT / "results/garch_parameters.csv")
    diag = pd.read_csv(ROOT / "results/ljung_box.csv")
    sensitivity = pd.read_csv(ROOT / "results/winsorized_sensitivity.csv").set_index("ticker")
    acfs = pd.read_csv(ROOT / "results/acf.csv")
    provenance = json.loads((ROOT / "data/provenance.json").read_text())
    sectors = provenance["sectors"]
    quiet = risk.sample_annual_vol_pct.idxmin()
    noisy = risk.sample_annual_vol_pct.idxmax()
    slow = risk.half_life_days.idxmax()
    cluster = risk.mean_squared_acf_1_10.idxmax()

    paragraph("Understanding risk and volatility", "ReportTitle")
    paragraph("Part 1 code, results, and walkthrough for Brendan Kerns and Chi Innis")
    table([["Ticker", "Sector"]] + [[t, sectors[t]] for t in risk.index], [70, 440])
    paragraph(f"The analysis uses {provenance['daily_observations']:,} daily observations per stock from {provenance['first_observation']} to {provenance['last_observation']}. The assignment's November 1 cutoff is a Sunday. The input files are the Yahoo adjusted-price snapshot saved during the November 11 proposal work. Adjusted prices account for splits and dividends. This run uses that fixed snapshot so the results can be reproduced offline.")
    paragraph(f"{quiet} has the lowest observed annualized volatility ({risk.loc[quiet, 'sample_annual_vol_pct']:.1f}%), while {noisy} has the highest ({risk.loc[noisy, 'sample_annual_vol_pct']:.1f}%). {cluster} has the strongest average squared-return autocorrelation over lags 1-10. {slow} has the slowest estimated decay of variance shocks. These describe different aspects of risk, so the rankings need not match.")
    figure("P1_risk_comparison", 510)
    paragraph("Read this guide alongside Part1_Walkthrough.ipynb. The notebook shows intermediate calculations and the tables. All estimates come from scripts/02_P1_diagnostics_garch.py, and scripts/05_figures.py regenerates the figures.", "SmallNote")

    page("1  From prices to usable returns")
    paragraph("A stationary series has statistical properties that remain stable over time. A rising stock-price level often fails this condition. Taking logs changes the scale but may leave the trend intact. Differencing log prices produces log returns, which measure proportional daily changes.")
    paragraph("The transformation is r[t] = 100 x (log(P[t]) - log(P[t-1])). A move from 100 to 101 gives about 0.995% as a log return and 1% as a simple return. We compute changes before trimming the dates so the first January return uses the previous December close. GARCH uses percentage log returns throughout.")
    paragraph("ADF starts with a unit-root null. A small p-value rejects it. KPSS starts with a stationarity null. A small p-value rejects stationarity. At the 5% level, ADF below 0.05 together with KPSS at least 0.05 supports using a stationary series. Neither test proves that the distribution is identical in every period.")
    rows = [["Stock", "Series", "ADF stat", "ADF p", "KPSS stat", "KPSS p"]]
    for _, row in tests[tests.regression == "c"].iterrows():
        rows.append([row.ticker, row.series.replace("log_return_pct", "log return"), f"{row.adf_stat:.3f}", pvalue(row.adf_p), f"{row.kpss_stat:.3f}", f"{row.kpss_p_bound}{row.kpss_p:.2f}"])
    table(rows, [43, 90, 80, 78, 80, 80])
    paragraph("The table uses an intercept, ADF lag selection by AIC, and automatic KPSS bandwidth. KPSS p-values at 0.01 and 0.10 are bounds from its reference table. Full lag counts, effective sample sizes, critical values, warnings, and supplementary trend tests are in results/stationarity.csv.", "SmallNote")
    paragraph("All five price and log-price series fail the joint level-stationarity check. Log returns satisfy both tests for every stock, so we stop at that transformation. In the supplementary trend tests, MSFT's price ADF is borderline (p = 0.0489) while KPSS still rejects; that result is mixed, and the log-price tests still support differencing.", "SmallNote")
    return_checks = tests[(tests.series == "log_return_pct") & (tests.regression == "c")]
    paragraph("Return-series conclusions at 5%: " + "; ".join(f"{r.ticker} {r.decision_5pct}" for _, r in return_checks.iterrows()) + ".")

    page("2  What the autocorrelations tell us")
    paragraph("An autocorrelation compares a series with an earlier copy of itself. Lag 1 compares today's return with the previous trading day's return. Lag 10 compares it with the return ten trading days earlier. We calculate the first ten lags separately for returns and squared returns.")
    paragraph("Squaring removes the sign and emphasizes large moves. Persistent squared returns mean that large moves tend to be near other large moves, whether those moves are gains or losses. This is volatility clustering. It motivates a model with changing conditional variance, even when return direction is difficult to predict.")
    rows = [["Stock", "Return ACF lag 1", "Squared ACF lag 1", "Mean squared ACF 1-10", "Return LB p", "Squared LB p"]]
    for ticker in risk.index:
        def ac(label):
            return acfs[(acfs.ticker == ticker) & (acfs.series == label) & (acfs.lag == 1)].acf.iloc[0]
        def lb(label):
            return diag[(diag.ticker == ticker) & (diag.series == label)].lb_pvalue.iloc[0]
        rows.append([ticker, f"{ac('returns'):.3f}", f"{ac('squared_returns'):.3f}", f"{risk.loc[ticker, 'mean_squared_acf_1_10']:.3f}", pvalue(lb('returns')), pvalue(lb('squared_returns'))])
    table(rows, [45, 88, 88, 105, 90, 90])
    paragraph("LB is the Ljung-Box test using lags 1-10 together. Its null is zero autocorrelation at these lags. The usual p-values for raw returns can be distorted by changing variance, so use them as descriptive diagnostics. A small return ACF can still be statistically detectable in a large sample. It does not establish a profitable forecasting rule.")
    paragraph("The shaded ACF bands are the approximate white-noise reference +/-1.96/sqrt(n), not a joint confidence region for all ten lags. They are a visual reference, especially for squared returns. The mean squared-return ACF is a transparent descriptive ranking of short-horizon clustering, not a formal test of differences between stocks.")
    paragraph("Here, the return Ljung-Box tests reject zero autocorrelation for every stock, so calling the raw returns entirely uncorrelated would overstate the result. Squared-return dependence is much stronger, especially for JPM. The GARCH residual diagnostics later in the guide show how much dependence remains after adjusting for changing volatility.")
    paragraph("What to do with clustering", "Sub")
    paragraph("Allow the risk estimate to change with recent shocks. GARCH does this through the conditional variance. In portfolio decisions, a fixed full-sample standard deviation can miss a temporary risk spike. Conditional volatility helps monitor that change, while Parts 2 and 3 will address factor exposure and portfolio weights.")

    page("3  How the GARCH model works")
    paragraph("We fit the required constant-mean GARCH(1,1) to each stock. The model is r[t] = mu + epsilon[t], with epsilon[t] = sigma[t] x z[t]. Its variance equation is h[t] = omega + alpha x epsilon[t-1]^2 + beta x h[t-1], where h[t] = sigma[t]^2.")
    table([
        ["Quantity", "Meaning"],
        ["mu", "Average daily percentage log return in the constant-mean equation."],
        ["omega", "Positive baseline term in daily variance, measured in percent squared."],
        ["alpha", "How strongly the next variance responds to the previous squared surprise."],
        ["beta", "How much of the previous conditional variance carries forward."],
        ["alpha + beta", "Persistence of expected variance after a shock. Near one means slow decay."],
        ["ln(0.5) / ln(alpha + beta)", "Trading days for a deviation of expected variance to halve, assuming persistence lies between zero and one."],
        ["omega / (1 - alpha - beta)", "Model-implied long-run DAILY VARIANCE, valid only when persistence is below one."],
        ["sqrt(long-run variance x 252)", "Model long-run annualized volatility in percent."],
        ["sample SD(return) x sqrt(252)", "Observed annualized volatility over the full sample, in percent."],
    ], [180, 330])
    paragraph("The likelihood uses normal innovations and robust standard errors. Robust inference reduces reliance on normality for coefficient uncertainty, but the normal model can still understate extreme-tail behavior. A parameter's standard error describes estimation uncertainty, and its p-value tests a zero coefficient using an asymptotic approximation.")
    paragraph("We compare annualized volatility with annualized volatility. Variance and standard deviation have different units, so comparing the raw long-run variance directly with annual sample volatility would be misleading. The full daily variance is also reported to satisfy the assignment.")
    paragraph("GARCH permits conditional volatility to move through time even when the unconditional return process is stationary. The symmetric model responds equally to equally sized positive and negative surprises. It does not capture leverage asymmetry or all structural changes.")

    page("4  Compare the five stocks")
    rows = [["Stock", "Persistence", "Half-life days", "Long-run variance %^2", "GARCH annual vol %", "Sample annual vol %"]]
    for ticker, row in risk.iterrows():
        rows.append([ticker, f"{row.persistence:.4f}", f"{row.half_life_days:.1f}", f"{row.long_run_variance_pct2:.3f}", f"{row.long_run_annual_vol_pct:.2f}", f"{row.sample_annual_vol_pct:.2f}"])
    table(rows, [43, 85, 85, 100, 97, 97])
    paragraph(f"Within this five-stock sample, {quiet} represents the calmer holding by observed volatility. That is evidence about {quiet}, rather than an estimate for every company in {sectors[quiet]}. {noisy} has the largest observed volatility. A large allocation to it would therefore need a stronger diversification or expected-return justification in Part 3.")
    paragraph(f"{cluster} ranks highest on the average squared-return ACF, meaning its large moves show the strongest short-horizon clustering by this measure. {slow} has the largest GARCH half-life. Its fitted variance shocks are expected to fade more slowly, which matters for how long a temporary risk increase may persist.")
    paragraph("Part 1 risk posture", "Sub")
    paragraph(f"Keep a diversified five-name allocation as the starting point and avoid concentrating solely because one stock recently performed well. Consider a smaller exposure to {noisy} if Parts 2 and 3 do not show sufficient compensation or diversification benefit. Treat {quiet} as a candidate for the lower-volatility portion of the portfolio. Monitor stocks with persistent volatility especially closely after large shocks. Exact weights require the cross-stock covariances and optimization in Part 3.")
    paragraph("How sensitive are the estimates to extreme returns?", "Sub")
    rows = [["Stock", "Capped days", "Original persistence", "Capped persistence", "Original half-life", "Capped half-life"]]
    for ticker, row in sensitivity.iterrows():
        rows.append([ticker, int(row.capped_days), f"{risk.loc[ticker, 'persistence']:.4f}", f"{row.persistence:.4f}", f"{risk.loc[ticker, 'half_life_days']:.1f}", f"{row.half_life_days:.1f}"])
    table(rows, [45, 65, 100, 100, 100, 100])
    paragraph("The sensitivity fit caps returns at each stock's sample 1st and 99th percentiles. It deliberately changes the data and usually reduces volatility. It tests sensitivity to tails, not whether extreme observations should be discarded. Large shifts in persistence or half-life weaken a precise ranking.", "SmallNote")

    for ticker, row in risk.iterrows():
        page(f"{ticker}  |  {sectors[ticker]}")
        subset = params[params.ticker == ticker]
        table([["Parameter", "Estimate", "Robust SE", "p-value"]] +
              [[r.parameter, f"{r.estimate:.5f}", f"{r.robust_se:.5f}", pvalue(r.p_value)] for _, r in subset.iterrows()], [100, 130, 130, 150])
        paragraph(f"Persistence is {row.persistence:.4f}; variance-shock half-life is {row.half_life_days:.1f} trading days. Model long-run annualized volatility is {row.long_run_annual_vol_pct:.1f}%, compared with {row.sample_annual_vol_pct:.1f}% observed over the sample. The largest absolute daily log return is {row.max_abs_return_pct:.2f}% on {row.extreme_date}.", "SmallNote")
        figure(f"P1_{ticker}_transformations", 315)
        figure(f"P1_{ticker}_acf", 380)
        figure(f"P1_{ticker}_volatility", 380)

    page("5  What the model leaves unexplained")
    rows = [["Stock", "Std. residual LB p", "Squared std. residual LB p", "Persistence 95% interval"]]
    for ticker, row in risk.iterrows():
        d = diag[diag.ticker == ticker].set_index("series")
        rows.append([ticker, pvalue(d.loc["standardized_residuals", "lb_pvalue"]), pvalue(d.loc["squared_standardized_residuals", "lb_pvalue"]), f"{row.persistence_ci_low:.3f} to {row.persistence_ci_high:.3f}"])
    table(rows, [50, 135, 165, 160])
    paragraph("Standardized residuals are surprises divided by the fitted daily volatility. Autocorrelation left in these residuals suggests the constant-mean equation misses some dependence. Autocorrelation left in their squares suggests the variance equation misses some volatility structure. The post-fit Ljung-Box tests are approximate diagnostics because parameters were estimated on the same observations.")
    paragraph("A persistence confidence interval reaching one means that finite long-run variance is uncertain even if the point estimate is below one. Half-life and long-run variance are very sensitive near that boundary. These intervals use the robust covariance of alpha and beta and a normal approximation, which is imperfect near parameter constraints.")
    paragraph("XOM is the clearest concern in this run. Its squared standardized residual test has p = 0.023, suggesting remaining volatility dependence at the 5% level. Its estimated half-life is about 243 trading days, falling to about 177 after capping extreme returns. Its persistence interval includes one. MSFT and JPM also have intervals reaching one. Treat these half-lives as uncertain point estimates, with particular caution for XOM.")
    paragraph("The ADF/KPSS evidence addresses unit roots and level or trend stationarity. It does not establish stable tails across crises or rule out structural breaks. Gaussian GARCH has symmetric shocks, a constant mean, and one fixed parameter set over nearly eleven years. A converged optimizer establishes numerical completion, not model adequacy.")
    paragraph("This is a retrospective fit, with parameters estimated using the full sample. Conditional-volatility plots are in-sample estimates. They are not an out-of-sample trading test. Price dates are identical across all five files, but a separate exchange-calendar audit was not performed. One company per sector cannot support a general sector-wide ranking.")

    page("6  Follow the code and reproduce the results")
    table([
        ["File", "What happens and why"],
        ["01_get_data.py", "Read the fixed snapshots, reject duplicate or invalid prices, align trading dates, and compute returns before trimming the sample. Save derived data and provenance."],
        ["02_P1_diagnostics_garch.py", "Loop over all five tickers. Run the stationarity sequence, calculate the first ten ACFs, fit GARCH, derive risk measures, and save coefficient uncertainty and diagnostics."],
        ["05_figures.py", "Read the saved tables and fitted series. Generate every Part 1 figure in PNG and SVG formats."],
        ["part1_report.py", "Read the calculated tables and figures, then build this explanation PDF."],
        ["Part1_Walkthrough.ipynb", "Run code in small sections, inspect the results, and work through a return calculation and GARCH formulas."],
    ], [165, 345])
    paragraph("Checks to do yourself", "Sub")
    paragraph("Choose a ticker and two consecutive prices. Calculate 100 x log(new price / old price) and compare it with the return file. For the same ticker, add alpha and beta from garch_parameters.csv. Substitute that sum into the half-life formula and compare with risk_summary.csv. Then explain why its ACF plots support or weaken the case for GARCH.")
    paragraph("Sources and implementation references", "Sub")
    for text in [
        "Course assignment, page 2, Part 1 Risk and volatility; page 5, repository structure. Course fda_utils.py is included for explicit refreshes.",
        "Yahoo Finance via yfinance, source prices saved during proposal work on November 11, 2026. Individual file checksums are in data/provenance.json.",
        "ADF API and null hypothesis: https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html",
        "KPSS API and bounded p-values: https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.kpss.html",
        "GARCH specification and percentage returns: https://arch.readthedocs.io/en/stable/univariate/univariate_volatility_modeling.html",
        "Robust covariance: https://arch.readthedocs.io/en/latest/univariate/generated/generated/arch.univariate.ConstantMean.fit.html",
    ]:
        paragraph(text, "SmallNote")
    paragraph("AI assistance is documented in AI_USE_LOG.txt. Review the reasoning and record your own verification before submitting any of this material.", "SmallNote")
    SimpleDocTemplate(str(ROOT / "Part1_Explanation.pdf"), pagesize=(612, 792),
                      rightMargin=48, leftMargin=48, topMargin=42, bottomMargin=44,
                      title="BUS 493 Part 1 Risk and Volatility Explained", author="Brendan Kerns and Chi Innis").build(story, onFirstPage=footer, onLaterPages=footer)
    print("Created Part1_Explanation.pdf")


if __name__ == "__main__":
    main()
