# crypto-news-sentiment-Search and Trend-tracker

UIUC CS410 Final Project — Fall 2025  
Crypto News Sentiment Search and Trend Tracker  
A Python-based system for BM25 retrieval, sentiment analysis, and financial trend visualization.

## Author

Seokhyun Lee (sl251)  
Course: CS410 – Text Information Systems  
Instructor: Prof. ChengXiang Zhai  
Institution: University of Illinois Urbana-Champaign  

---

## Project Overview

This project implements a lightweight information retrieval pipeline to search cryptocurrency news headlines and analyze sentiment trends. It supports BM25-based headline search, sentiment scoring, histogram plotting, and visualization of sentiment trends alongside actual BTC/ETH market prices.

---

## Features

- BM25 retrieval using Pyserini (Lucene index)
- Sentiment analysis using TextBlob (–1 to +1)
- Sentiment histograms for selected queries
- Monthly sentiment aggregation
- Overlay of sentiment vs BTC/ETH historical prices (via Yahoo Finance)
- Fully CPU-compatible; no GPU needed

---

## Repository Structure

crypto-news-sentiment-tracker/  
├── data/                         # Raw CSV data (crypto-news.csv)  
├── processed_corpus/            # Preprocessed JSONL documents  
├── indexes/                     # BM25 index (Lucene format)  
├── outputs/                     # PNG output files  
│   ├── sentiment_histograms/    # Histograms for sentiment distribution  
│   └── trend_plots/             # Monthly sentiment vs price trend plots  
├── main.py                      # Retrieval, indexing, and querying  
├── demo_queries.py              # Sample queries with histogram generation  
├── trend_vs_price.py            # Sentiment trend overlay with BTC/ETH price  
├── requirements.txt             # Python dependencies  
└── README.md                    # Project documentation  

---

## Installation

### 1. Create environment

```bash
conda create -n crypto410 python=3.10 -y
conda activate crypto410
```

### 2. Install dependencies

#### Option A — using requirements.txt (recommended)
```bash
pip install -r requirements.txt
python -m textblob.download_corpora
```

#### Option B — manual install (if requirements.txt is not available)
```bash
pip install pyserini==0.24.0 pandas textblob matplotlib yfinance tqdm numpy
python -m textblob.download_corpora
```

> Note: Ensure Java 11 is installed and `java -version` works. Pyserini requires Java.

---

## Data Preparation

To clean and convert the dataset to JSONL format:

```bash
python main.py 
```

This reads `data/crypto-news.csv`, processes and stores output in `processed_corpus/crypto/`.

---

## Indexing

To build the BM25 index using Pyserini:

```bash
python main.py --build-index
```

Lucene index is saved to `indexes/crypto/`.

---

## Searching

Run a BM25 search query and view ranked results with sentiment scores:

```bash
python main.py --query "bitcoin regulation"
```

Outputs top-20 headlines with their BM25 score and sentiment polarity.

---

## Sentiment Histogram Visualization

To run example queries and plot histograms:

```bash
python demo_queries.py
```

Output files are saved in `outputs/sentiment_histograms/`.

---

## Sentiment Trend vs Market Price

To generate trend overlays between monthly sentiment and BTC/ETH prices:

```bash
python trend_vs_price.py
```

Output charts are saved in `outputs/trend_plots/`.

---

## Notes

- Headlines are short and sentiment is computed using TextBlob (rule-based).
- Sentiment is aggregated monthly.
- Prices are fetched from Yahoo Finance using the `yfinance` API.
- No full-article text is used; analysis is headline-based.

---

## Limitations

- No synonym normalization (e.g., “BTC” ≠ “Bitcoin”).
- TextBlob may misclassify neutral financial headlines.
- Results depend on headline phrasing and may miss subtle sentiment.

---

## Future Work

- Replace TextBlob with FinBERT for finance-specific sentiment scoring.
- Add synonym/entity normalization (BTC ↔ Bitcoin).
- Integrate live news stream (e.g., Twitter, RSS).
- Develop web-based frontend with search interface.
- Add query expansion using pseudo-relevance feedback.

---

## Contact

Seokhyun Lee  
Email: sl251@illinois.edu
