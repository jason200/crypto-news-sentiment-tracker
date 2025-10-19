# build_index.py
import os, json, pandas as pd, subprocess
from tqdm import tqdm

DATA_PATH = "data/crypto-news.csv"
PROCESSED_DIR = "processed_corpus/crypto"
INDEX_DIR = "indexes/crypto"

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs("indexes", exist_ok=True)

print("🔧 Step 1: Preprocessing corpus...")
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["title"]) 

for i, row in tqdm(df.iterrows(), total=len(df)):
    doc = {"id": str(i), "contents": str(row['title'])}
    with open(f"{PROCESSED_DIR}/doc{i}.json", "w") as f:
        json.dump(doc, f)

print("✅ Finished preprocessing!")

print("✅ Step 2: Building Lucene index with Pyserini...")
subprocess.run([
    "python", "-m", "pyserini.index.lucene",
    "--collection", "JsonCollection",
    "--input", PROCESSED_DIR,
    "--index", INDEX_DIR,
    "--generator", "DefaultLuceneDocumentGenerator",
    "--threads", "1",
    "--storePositions", "--storeDocvectors", "--storeRaw"
], check=True)

print(f"✅ Index built successfully at: {INDEX_DIR}")