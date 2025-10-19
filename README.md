# crypto-news-sentiment-tracker
UIUC CS410 Final Project — Crypto News Sentiment Search and Trend Tracker A Python-based system integrating BM25 retrieval, TextBlob sentiment analysis, and financial trend visualization.

# Crypto News Sentiment Search and Trend Tracker

**Author:** Seokhyun Lee (sl251)  
**Course:** CS410 – Text Information Systems  
**Semester:** Fall 2025  
**Instructor:** ChengXiang Zhai  
**Institution:** University of Illinois Urbana-Champaign  

---

## 🧭 Overview

This project implements a lightweight **information retrieval and sentiment analysis pipeline** designed to study public sentiment trends across cryptocurrency-related news headlines and correlate them with actual market behavior.

The system combines:
- **Pyserini BM25** — for text-based retrieval from a corpus of crypto news headlines  
- **TextBlob** — for sentiment polarity scoring (–1 = negative, +1 = positive)  
- **Matplotlib** — for visualization of sentiment histograms and time-based trend overlays  
- **Yahoo Finance API (yfinance)** — for fetching historical BTC/USD and ETH/USD prices  

All computations run efficiently on CPU hardware, making this system fully reproducible in standard academic environments.
