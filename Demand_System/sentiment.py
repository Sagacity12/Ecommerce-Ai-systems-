import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import pipeline

# LOADING TOKEN FROM HUGGINGFACE
from dotenv import load_dotenv
import os

load_dotenv()
hf_token = os.getenv('HF_TOKEN')

from huggingface_hub import login
login(token=hf_token)

# creating directories and reading the dataset
CSV_FILE_PATH = 'Dataset/data_features.csv'
OUTPUT_FILE_PATH = 'Dataset/data_sentiment.csv'
os.makedirs('feature_engineering_output', exist_ok=True)


df = pd.read_csv(CSV_FILE_PATH, encoding_errors='replace')
print("____loaded features data:", df.shape)

# Loading HuggingFace Sentiment Model
print("\n── Loading HuggingFace Sentiment Model...")

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model = "distilbert-base-uncased-finetuned-sst-2-english",
    truncation = True,
    max_length = 512
)
print("Model loaded")

# Preparing product names for sentiment
print("\n── Prepare product names")
unique_products = df[['product_code', 'product_name']].drop_duplicates()
unique_products['product_name'] = unique_products['product_name'].fillna('unknown product')
unique_products['product_name'] = unique_products['product_name'].astype(str).str.strip()

print(f"Unique products to score: {len(unique_products)}")
print(unique_products.head(10))

# Run sentiment on unique products
print("\n── Running sentiment analysis...")

def get_sentiment_score(text):
    try:
        result = sentiment_pipeline(text[:512])[0]
        score = result['score']
        # POSITIVE -> positive score, NEGATIVE -> negative score
        return  score if result['label'] == 'POSITIVE' else -score
    except Exception as e:
        return 0.0

# Run in the batches of 64 for speed
batch_size = 64
scores = []
texts = unique_products['product_name'].values.tolist()

for i in range(0, len(texts), batch_size):
    batch = texts[i:i + batch_size]
    results = sentiment_pipeline(batch, truncation=True, max_length=512)
    for r in results:
        score = r['score'] if r['label'] == 'POSITIVE' else -r['score']
        scores.append(score)

    if i % 500 == 0:
        print(f" Processed {i}/{len(texts)} products...")

unique_products['sentiment_score'] = scores
print("\nSentiment scoring complete")
print(unique_products[['product_code', 'product_name', 'sentiment_score']].head(15))

# Merge sentiment Back to main Df
print("\n── Merge sentiment scores")

df = df.merge(
    unique_products[['product_code',  'sentiment_score']],
    on='product_code',
    how='left'
)

df['sentiment_score'] = df['sentiment_score'].fillna(0.0)

print("After merge:", df.shape)
print(df[['product_code', 'product_name', 'sentiment_score']].head(10))

# Sentiment Distribution
print("\n── Sentiment Distribution")
print(df['sentiment_score'].describe())

plt.figure(figsize=(8, 5))
sns.histplot(df['sentiment_score'], bins=50, kde=True, color='steelblue')
plt.title('Product Name Sentiment Score Distribution')
plt.xlabel('Sentiment Score (-1 = Negative, +1 = Positive)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('feature_engineering_output/sentiment_distribution.png')
plt.show()

# Sentiment vs demand
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='sentiment_score', y='order_quantity', alpha=0.2, color='coral')
plt.title('Sentiment Score vs Order Quantity')
plt.xlabel('Sentiment Score')
plt.ylabel('Order Quantity')
plt.tight_layout()
plt.savefig('feature_engineering_output/sentiment_vs_demand.png')
plt.show()

df.to_csv(OUTPUT_FILE_PATH, index=False)

print("\n── SENTIMENT COMPLETE")
print(f"Final shape : {df.shape}")
print(f"Columns     : {df.columns.tolist()}")
print(f"Saved to    : {OUTPUT_FILE_PATH}")