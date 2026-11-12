BUS 493 | Five-stock portfolio project

Proposal-stage repository. The numbered files reserve the analysis steps. Script 01 currently prints the stock universe. Data downloads, models and figures will be implemented during the analysis.

Setup and planned run order
From the repository root, use a Python virtual environment and install packages with:
  python -m pip install -r requirements.txt

  python scripts/01_get_data.py
    Download adjusted prices and factor data, then calculate monthly returns.
  python scripts/02_P1_diagnostics_garch.py
    Run stationarity, autocorrelation, and GARCH diagnostics for each stock.
  python scripts/03_P2_factors.py
    Fit monthly CAPM and Fama-French three-factor models for each stock.
  python scripts/04_P3_portfolio.py
    Build the efficient frontier and check the stability of portfolio weights.
  python scripts/05_figures.py
    Generate the report figures from the analysis outputs.

Inputs: Yahoo Finance adjusted prices for MSFT, JPM, XOM, COST and JNJ, and monthly US factors from Kenneth French's Data Library. Save downloaded and derived inputs in data/. Record download dates and package versions for the submitted analysis. Place required course helpers in scripts/ when used. Generate each report figure in figures/. The completed report will be report.pdf at the repository root.

Brendan owns Part 1 and writes the Part 3 code. Chi owns Part 2. Both partners interpret Part 3 and agree on the final allocation.
