# crypto_news_sentiment.py
# Author: Seokhyun Lee (sl251)
# Description: Runs BM25 retrieval on crypto news headlines and performs sentiment analysis using TextBlob.

from pyserini.search.lucene import LuceneSearcher
from textblob import TextBlob
import matplotlib.pyplot as plt
import numpy as np
import os

INDEX_DIR = "indexes/crypto_news"

if not os.path.exists(INDEX_DIR):
    print("✅ Index not found. Please build the index first using Pyserini.")
    print("Example:")
    print("python -m pyserini.index.lucene --collection JsonCollection --input processed_corpus --index indexes/crypto_news --generator DefaultLuceneDocumentGenerator --threads 2 --storePositions --storeDocvectors --storeRaw")
    exit()

searcher = LuceneSearcher(INDEX_DIR)
searcher.set_bm25(k1=0.9, b=0.4)

print("\nWelcome to the Crypto News Sentiment Search Tool 🔎")
query = input("Enter a crypto-related keyword (e.g., 'bitcoin regulation'): ").strip()

hits = searcher.search(query, k=10)
if not hits:
    print("No results found.")
    exit()

sentiments = []
print(f"\nTop 10 results for query: '{query}'\n")
for i, hit in enumerate(hits):
    doc = hit.raw
    polarity = TextBlob(doc).sentiment.polarity
    sentiments.append(polarity)
    print(f"{i+1:02d}. {doc[:120]}... | Sentiment: {polarity:+.2f}")

# Summary stats
avg_sent = np.mean(sentiments)
print(f"\n✅ Average Sentiment Score: {avg_sent:+.2f} (-1=negative, +1=positive)")

# Visualization
plt.figure(figsize=(7,4))
plt.hist(sentiments, bins=8, color='lightblue', edgecolor='black')
plt.title(f"Sentiment Distribution for '{query}'")
plt.xlabel("Sentiment Polarity (-1 to 1)")
plt.ylabel("Frequency")
plt.axvline(avg_sent, color='red', linestyle='--', label=f'Average: {avg_sent:+.2f}')
plt.legend()
plt.tight_layout()
plt.show()