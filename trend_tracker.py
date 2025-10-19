# trend_tracker.py
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

def compute_sentiment(text):
    """Compute sentiment polarity using TextBlob (-1 = negative, +1 = positive)."""
    return TextBlob(str(text)).sentiment.polarity

def plot_trend(keyword, csv_path='data/crypto-news.csv'):
    """Plot sentiment trend over time for a given keyword."""
    # Load dataset
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['date', 'title'])
    
    # Filter by keyword
    df = df[df['title'].str.contains(keyword, case=False, na=False)].copy()
    if df.empty:
        print(f"No articles found for keyword: {keyword}")
        return

    # Compute sentiment scores
    df['sentiment'] = df['title'].apply(compute_sentiment)

    # Convert to datetime and group by month
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['month'] = df['date'].dt.to_period('M')

    monthly_sentiment = df.groupby('month')['sentiment'].mean().reset_index()

    # Plot sentiment trend
    plt.figure(figsize=(8, 4))
    plt.plot(monthly_sentiment['month'].astype(str),
             monthly_sentiment['sentiment'],
             marker='o', linestyle='-', color='blue')
    plt.title(f"Sentiment Trend Over Time: {keyword}")
    plt.xlabel("Month")
    plt.ylabel("Average Sentiment (-1=Negative, +1=Positive)")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"outputs/{keyword.replace(' ', '_')}_trend.png")
    plt.show()

    print(f"✅ Trend chart saved: outputs/{keyword.replace(' ', '_')}_trend.png")

if __name__ == "__main__":
    # Example test
    plot_trend("bitcoin")
    plot_trend("ethereum")