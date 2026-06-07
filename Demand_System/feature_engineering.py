import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import *

OUTPUT_PATH = 'feature_engineering_output'
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Load the clean Data
df = pd.read_csv('Dataset/data_cleaned.csv', encoding_errors='replace')
df['order_date'] = pd.to_datetime(df['order_date'])
df.sort_values('order_date', inplace=True)
df.reset_index(drop=True, inplace=True)

print("── Loaded clean data:", df.shape)

# Time Features
print("\n--- Time Feature")

df['order_quarter'] = df['order_date'].dt.quarter
df['order_week'] = df['order_date'].dt.isocalendar().week.astype(int)
df['day_of_week_num'] = df['order_date'].dt.dayofweek
df['is_weekend'] = (df['day_of_week_num'] >= 5).astype(int)

df['order_date'] = pd.to_datetime(df['order_date']).dt.date
df['order_date'] = pd.to_datetime(df['order_date'])

print(df[['order_date', 'order_quarter', 'order_week',
          'is_weekend', 'day_of_week_num']].head(10))


# Daily Demand Aggregate (needed for lag + rolling)
print("\n──  Daily Demand Aggregation")

daily = (df.groupby('order_date')['order_quantity']
         .sum()
         .reset_index()
         .rename(columns={'order_quantity': 'daily_demand'}))
daily.sort_values('order_date', inplace=True)
daily = daily.set_index('order_date').asfreq('D', fill_value=0).reset_index()
# daily.reset_index(drop=True, inplace=True)
print(daily.head(10))

# Lad Features
print("\n── Lag Features")
daily['lag_7'] = daily['daily_demand'].shift(7)
daily['lag_14'] = daily['daily_demand'].shift(14)
daily['lag_30'] = daily['daily_demand'].shift(30)
print(daily[['order_date', 'daily_demand', 'lag_7', 'lag_14', 'lag_30']].head(35))

# Rolling Averages
print("\n── Rolling Averages")

daily['rolling_7_avg']  = daily['daily_demand'].rolling(window=7,  min_periods=1).mean()
daily['rolling_30_avg'] = daily['daily_demand'].rolling(window=30, min_periods=1).mean()
daily['rolling_7_std']  = daily['daily_demand'].rolling(window=7,  min_periods=1).std()

print(daily[['order_date', 'daily_demand',
             'rolling_7_avg', 'rolling_30_avg', 'rolling_7_std']].head(35))

# Merge Daily Features Back Into Main DF
print("\n----- Merge lag + rolling back to main df")

df = df.merge(daily[['order_date', 'daily_demand', 'lag_7', 'lag_14',
                      'lag_30', 'rolling_7_avg', 'rolling_30_avg',
                      'rolling_7_std']],
              on='order_date', how='left')

print("After merge:", df.shape)

# Price * Promotion Interaction
print("\n── Price × Promotion Interaction")

# Promotion flag — orders where price is 20% below product average
avg_price = df.groupby('product_code')['price_per_unit'].transform('mean')
df['is_promoted']         = (df['price_per_unit'] < avg_price * 0.80).astype(int)
df['price_x_promotion']   = df['price_per_unit'] * df['is_promoted']
df['discount_depth']      = ((avg_price - df['price_per_unit']) / avg_price).clip(lower=0)

print(f"Promoted rows    : {df['is_promoted'].sum()}")
print(f"Non-promoted rows: {(df['is_promoted'] == 0).sum()}")
# print(df[['product_code', 'price_per_unit', 'is_promoted',
#           'price_x_promotion', 'discount_depth']].head(10))

# Wholesaler Features
print("\n── Wholesaler Features")

# Total demand per product — what wholesaler sees
df['product_total_demand'] = df.groupby('product_code')['order_quantity'].transform('sum')

# Unique retailers ordering this product — buyer reach
df['product_unique_buyers'] = df.groupby('product_code')['retailer_id'].transform('nunique')

# How frequently this product is reordered
df['product_order_freq'] = df.groupby('product_code')['order_id'].transform('count')

# Revenue this product generates for wholesaler
df['product_revenue'] = df.groupby('product_code')['order_value_ghs'].transform('sum')

# Average order value per region — demand strength by region
df['region_avg_order_value'] = df.groupby(
    'buyer_region_encoded')['order_value_ghs'].transform('mean')

# Days since product was last ordered — freshness signal
last_order_per_product    = df.groupby('product_code')['order_date'].transform('max')
df['days_since_last_order'] = (df['order_date'].max() - last_order_per_product).dt.days

# Slow mover flag — SRS FR-CAT-11 (not reordered in 60+ days)
df['is_slow_mover'] = (df['days_since_last_order'] >= 60).astype(int)

# Retailer order count — how active is this buyer
df['retailer_order_count'] = df.groupby('retailer_id')['order_id'].transform('count')

# Retailer avg quantity — buying pattern
df['retailer_avg_quantity'] = df.groupby('retailer_id')['order_quantity'].transform('mean')

# Retailer total spend — value of this retailer to wholesaler
df['retailer_total_spend'] = df.groupby('retailer_id')['order_value_ghs'].transform('sum')

print(df[['product_code', 'product_total_demand', 'product_unique_buyers',
          'product_order_freq', 'is_slow_mover',
          'retailer_order_count', 'retailer_avg_quantity']].head(10))

# Drop rows with NaN from Lag (first 30 days)
print("\n── Drop NaN rows from lag features")
before = len(df)
df.dropna(subset=['lag_7', 'lag_14', 'lag_30'], inplace=True)
print(f"Dropped {before - len(df)} rows | Remaining: {df.shape}")

# Visualize Features
print("\n── Visualizations")

# Lag vs demand
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, lag in zip(axes, ['lag_7', 'lag_14', 'lag_30']):
    ax.scatter(df[lag], df['order_quantity'], alpha=0.1, color='steelblue', s=5)
    ax.set_title(f'{lag} vs order_quantity')
    ax.set_xlabel(lag)
    ax.set_ylabel('order_quantity')
plt.suptitle('Lag Features vs Demand')
plt.tight_layout()
plt.savefig(f'{OUTPUT_PATH}/lag_vs_demand.png')
plt.show()

# Rolling averages over time
plt.figure(figsize=(14, 5))
plt.plot(daily['order_date'], daily['daily_demand'],   alpha=0.3, label='Daily Demand')
plt.plot(daily['order_date'], daily['rolling_7_avg'],  label='7-day Rolling Avg')
plt.plot(daily['order_date'], daily['rolling_30_avg'], label='30-day Rolling Avg')
plt.title('Demand with Rolling Averages')
plt.xlabel('Date')
plt.ylabel('Demand')
plt.legend()
plt.tight_layout()
plt.savefig(f'{OUTPUT_PATH}/rolling_averages.png')
plt.show()

# Promoted vs non-promoted demand
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x='is_promoted', y='order_quantity', palette='coolwarm')
plt.title('Demand: Promoted vs Non-Promoted')
plt.xticks([0, 1], ['Not Promoted', 'Promoted'])
plt.tight_layout()
plt.savefig(f'{OUTPUT_PATH}/promoted_vs_demand.png')
plt.show()

df.to_csv('Dataset/data_features.csv', index=False)
