import datetime as dt
import pathlib
import sys

import pandas as pd
import yfinance as yf

TICKERS = ["^SPX", "SPY"]            
OUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "chains"


def snapshot(ticker: str) -> pd.DataFrame:
    tk = yf.Ticker(ticker)
    spot = tk.history(period="5d")["Close"].iloc[-1]          
    frames = []

    for expiry in tk.options:                                 # every listed expiry date
        chain = tk.option_chain(expiry)
        for kind, df in (("call", chain.calls), ("put", chain.puts)):
            df = df.copy()
            df["type"], df["expiry"] = kind, expiry
            frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["underlying"], out["spot"] = ticker, spot
    out["snapshot_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    return out


def main() -> int:
    today = dt.date.today().isoformat()
    failures = 0
    for ticker in TICKERS:
        folder = OUT_DIR / ticker.replace("^", "")
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{today}.csv.gz"
        if path.exists():                                     
            print(f"{ticker}: already saved for {today}")
            continue
        try:
            df = snapshot(ticker)
            df.to_csv(path, index=False, compression="gzip")
            print(f"{ticker}: {len(df):,} contracts, {df['expiry'].nunique()} expiries -> {path}")
        except Exception as e:                                
            failures += 1
            print(f"{ticker}: FAILED ({e})", file=sys.stderr)
    return 1 if failures == len(TICKERS) else 0


if __name__ == "__main__":
    sys.exit(main())