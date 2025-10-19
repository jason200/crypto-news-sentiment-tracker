# preprocess.py
# CS410 Project: Crypto News Sentiment Search and Trend Tracker
# Author: Seokhyun Lee (sl251)
# Description: Cleans crypto news dataset and converts it into JSONL format for Pyserini indexing.

import pandas as pd
import json
import os

input_file = "data/crypto_news.csv"
output_dir = "processed_corpus"
output_file = os.path.join(output_dir, "crypto_news.jsonl")

os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(input_file)
df = df.dropna(subset=["title"])
df = df.drop_duplicates(subset=["title"])

with open(output_file, "w", encoding="utf-8") as f:
    for i, row in df.iterrows():
        doc = {
            "id": str(i),
            "contents": f"{row['title']} ({row.get('date', 'unknown date')})"
        }
        json.dump(doc, f)
        f.write("\n")

print(f"✅ Preprocessing complete. Saved {len(df)} documents to {output_file}")