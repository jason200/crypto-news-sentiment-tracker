from pyserini.search.lucene import LuceneSearcher
from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

# ==============================
# Configuration
# ==============================
INDEX_DIR = "indexes/crypto"
OUTPUT_DIR = "outputs"
DATA_PATH = "data/crypto-news.csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

QUERIES = [
    "bitcoin price prediction",
    "ethereum upgrade",
    "Bitcoin and crypto rally", 
    "crypto regulation",
    "blockchain security",
    "NFT Sales",
    "altcoin market",
    "US Fed rate cut could prove perfect catalyst for Bitcoin and crypto rally", 
    "btc price",
    "bitcoin price"  # different result wih btc price
]

# ==============================
# Utility Functions
# ==============================

def analyze_sentiment(text):
    """Return polarity score (-1 to +1)."""
    blob = TextBlob(text)
    return blob.sentiment.polarity


def run_query(searcher, query, top_k=20):
    """Retrieve top results and compute sentiment scores."""
    hits = searcher.search(query, k=top_k)
    results = []

    for hit in hits:
        try:
            raw_doc = searcher.doc(hit.docid).raw()
            if not raw_doc:
                continue

            doc = json.loads(raw_doc)
            content = doc.get("contents", "")
            polarity = analyze_sentiment(content)
            results.append({
                "docid": hit.docid,
                "score": hit.score,
                "polarity": polarity,
                "text": content
            })
        except Exception as e:
            print(f"Skipped doc {hit.docid}: {e}")
            continue

    return pd.DataFrame(results)


def summarize_sentiment(df, query):
    """Aggregate and visualize sentiment scores."""
    if df.empty:
        print(f"No results found for query '{query}'.")
        return

    avg_sentiment = df["polarity"].mean()
    pos = (df["polarity"] > 0).sum()
    neg = (df["polarity"] < 0).sum()
    neu = (df["polarity"] == 0).sum()

    print(f"Avg Sentiment: {avg_sentiment:.3f} | +:{pos}  -:{neg}  0:{neu}")

    # Histogram chart
    plt.figure(figsize=(6, 4))
    plt.hist(df["polarity"], bins=10, color="skyblue", edgecolor="black")
    plt.title(f"Sentiment Distribution: {query}")
    plt.xlabel("Polarity (-1=Negative, +1=Positive)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{query.replace(' ', '_')}_hist.png")
    plt.close()


# ==============================
# Trend Tracker Extension
# ==============================

def compute_sentiment_trend(keyword):
    """Compute and visualize monthly sentiment trend for a given keyword."""
    try:
        df = pd.read_csv(DATA_PATH)
        df = df.dropna(subset=["date", "title"])
    except Exception as e:
        print(f"Could not read dataset for trend tracking: {e}")
        return

    df = df[df["title"].str.contains(keyword, case=False, na=False)]
    if df.empty:
        print(f"No articles found for keyword '{keyword}' in dataset.")
        return

    df["sentiment"] = df["title"].apply(analyze_sentiment)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["month"] = df["date"].dt.to_period("M")
    trend = df.groupby("month")["sentiment"].mean().reset_index()

    plt.figure(figsize=(7, 4))
    plt.plot(trend["month"].astype(str), trend["sentiment"], marker="o", color="blue")
    plt.title(f"Sentiment Trend Over Time: {keyword}")
    plt.xlabel("Month")
    plt.ylabel("Average Sentiment (-1=Neg, +1=Pos)")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{keyword.replace(' ', '_')}_trend.png")
    plt.close()
    print(f"Trend chart saved for '{keyword}'.")


# ==============================
# Main Execution
# ==============================

if __name__ == "__main__":
    print("=" * 70)
    print("✅ Running Crypto News Sentiment Demo + Trend Tracker")
    print("=" * 70, "\n")

    searcher = LuceneSearcher(INDEX_DIR)

    for q in QUERIES:
        print(f"✅ Query: {q}")
        try:
            df = run_query(searcher, q)
            summarize_sentiment(df, q)
            compute_sentiment_trend(q.split()[0])  # trend per main keyword
        except Exception as e:
            print(f"   ❌ Error processing query '{q}': {e}")
        print()

    print("✅ All queries processed and trend charts generated in /outputs/")