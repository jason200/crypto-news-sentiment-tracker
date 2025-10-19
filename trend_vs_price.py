import os
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
import yfinance as yf

DATA_PATH = "data/crypto-news.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

START = "2021-10-12"
END   = "2023-12-19"

KEYWORDS = {
    "bitcoin": {"ticker": "BTC-USD", "outfile": f"{OUTPUT_DIR}/bitcoin_trend_vs_price.png"},
    "ethereum": {"ticker": "ETH-USD", "outfile": f"{OUTPUT_DIR}/ethereum_trend_vs_price.png"},
}

# --------------------------
# Sentiment helper
# --------------------------
def analyze_sentiment(text: str) -> float:
    return TextBlob(text).sentiment.polarity


# --------------------------
# Dataset loader
# --------------------------
def load_headlines(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Normalize column names
    if "title" not in df.columns and "headline" in df.columns:
        df = df.rename(columns={"headline": "title"})
    if "date" not in df.columns:
        for cand in ["published", "time", "pubDate", "datetime"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "date"})
                break
    df = df.dropna(subset=["date"])
    if "title" not in df.columns:
        for cand in ["text", "summary", "content"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "title"})
                break

    df = df.dropna(subset=["title"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[(df["date"] >= START) & (df["date"] <= END)]
    return df


# --------------------------
# Sentiment trend (monthly)
# --------------------------
def monthly_sentiment(df: pd.DataFrame, keyword: str) -> pd.DataFrame:
    sub = df[df["title"].str.contains(keyword, case=False, na=False)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["month", "sentiment"])
    sub["sentiment"] = sub["title"].apply(analyze_sentiment)
    sub["month"] = sub["date"].dt.to_period("M").astype(str)
    trend = sub.groupby("month", as_index=False)["sentiment"].mean()
    return trend


# --------------------------
# Price trend (monthly)
# --------------------------
def monthly_price(ticker: str) -> pd.DataFrame:
    data = yf.download(ticker, start=START, end=END, progress=False, auto_adjust=False)
    if data.empty:
        return pd.DataFrame(columns=["month", "close"])

    data = data.rename(columns={"Close": "close"})
    data.index = pd.to_datetime(data.index)

    data = data.resample("ME").last()[["close"]].copy()
    data = data.reset_index()
    data.columns = ["Date", "close"]
    data["month"] = data["Date"].dt.to_period("M").astype(str)
    data = data[["month", "close"]]
    return data


# --------------------------
# Normalization helper
# --------------------------
def minmax01(series: pd.Series) -> pd.Series:
    if series.empty or series.nunique() == 1:
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - series.min()) / (series.max() - series.min())


# --------------------------
# Plot overlay
# --------------------------
def plot_overlay(keyword: str, s_trend: pd.DataFrame, p_trend: pd.DataFrame, outfile: str):
    s_trend = s_trend.reset_index(drop=True)
    p_trend = p_trend.reset_index(drop=True)
    s_trend["month"] = s_trend["month"].astype(str)
    p_trend["month"] = p_trend["month"].astype(str)

    merged = pd.merge(s_trend, p_trend, on="month", how="outer").sort_values("month")

    # ✅ Fixed: no deprecated fillna(method="ffill")
    merged["sentiment_norm"] = minmax01(merged["sentiment"].astype(float).ffill())
    merged["price_norm"] = minmax01(merged["close"].astype(float).ffill())

    plt.figure(figsize=(9, 5))
    plt.plot(merged["month"], merged["sentiment_norm"], marker="o", label="Monthly Sentiment (normalized)")
    plt.plot(merged["month"], merged["price_norm"], marker="s", label="Monthly Close Price (normalized)")
    plt.title(f"{keyword.capitalize()}: Sentiment vs. Price ({START} → {END})")
    plt.xlabel("Month")
    plt.ylabel("Normalized Scale (0–1)")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()
    print(f"✅ Saved overlay chart → {outfile}")


# --------------------------
# Main
# --------------------------
def main():
    print(f"✅ Loading headlines from {DATA_PATH}")
    df = load_headlines(DATA_PATH)
    if df.empty:
        print("✅ No rows in the requested date range.")
        return

    for kw, cfg in KEYWORDS.items():
        print(f"\n✅ Building trends for: {kw} ({cfg['ticker']})")
        s_trend = monthly_sentiment(df, kw)
        p_trend = monthly_price(cfg["ticker"])

        if s_trend.empty:
            print(f"✅ No headlines matched keyword: {kw}")
        if p_trend.empty:
            print(f"✅ No price data for: {cfg['ticker']}")

        if not s_trend.empty and not p_trend.empty:
            plot_overlay(kw, s_trend, p_trend, cfg["outfile"])


if __name__ == "__main__":
    main()