"""Download adjusted prices and factor data, then calculate monthly returns.

Deliverable 1 starter. Analysis implementation is pending.
"""

STOCKS = {
    "MSFT": "Information Technology",
    "JPM": "Financials",
    "XOM": "Energy",
    "COST": "Consumer Staples",
    "JNJ": "Health Care",
}

if __name__ == "__main__":
    print("Ticker\tSector")
    for ticker, sector in STOCKS.items():
        print(f"{ticker}\t{sector}")
