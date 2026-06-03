import warnings
warnings.filterwarnings('ignore')

import os

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import load_and_prepare


OUTPUT_PATH = 'eda_output'
os.makedirs(OUTPUT_PATH, exist_ok=True)

df = load_and_prepare()
print(df)
# print(df.info())
# print(df.describe())
# print(df.head())
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")

# print(df.columns.tolist())

df.rename(columns={
    'InvoiceNo'   : 'order_id',
    'StockCode'   : 'product_code',
    'Description' : 'product_name',
    'Quantity'    : 'order_quantity',
    'InvoiceDate' : 'order_date',
    'UnitPrice'   : 'price_per_unit',
    'CustomerID'  : 'retailer_id',
    'Country'     : 'buyer_region'
}, inplace=True)

df['buyer_region_original'] = df['buyer_region']
region_mapping ={
    'United Kingdom'      : 'Greater Accra',
    'USA'     : 'Central Region',
    'France'  : 'Ashanti Region',
    'Germany' : 'Western Region',
    'Australia' : 'Northern Region',
    'EIRE' : 'Oti Region',
    'Poland' : 'Savannah Region',
    'Belgium' : 'Upper East Region',
    'Iceland' : 'Upper West Region' ,
    'Cyprus' : 'Western North Region',
    'Israel' : 'North East Region',
    'Brazil' : 'Eastern Region',
    'Greece' : 'Volta Region',
    'Lebanon' : 'Ahafo Region',
    'Malta' : 'Bono Region',
    'Spain' : 'Bono East Region'

}

# Apply remapping
df['buyer_region'] = df['buyer_region'].replace(region_mapping)

print(df['buyer_region'].unique())
# print(df[region_mapping].value_counts(16))
print(df.columns.tolist())
print(df.head())

# fixing datatypes for order using datetime
df['order_date'] = pd.to_datetime(df['order_date'])
df['retailer_id'] = df['retailer_id'].astype(str)

# Deriving new columns
df['order_value_ghs'] = df['order_quantity'] * df['price_per_unit']
df['order_month'] = df['order_date'].dt.month
df['order_day'] = df['order_date'].dt.day_name()
df['order_year'] = df['order_date'].dt.year

print("\nUpdated Columns:\n", df.columns.tolist())
print(df.head())

# Statistical summary
print("\nStatistical Summary:")
print(df[['order_quantity', 'price_per_unit', 'order_value_ghs']].describe())

# Univariate - Target Variable
plt.figure(figsize=(8, 5))
sns.histplot(df['order_quantity'], bins=50, kde=True, color='steelblue')
plt.title('Distribution of Order Quantity (Demand)')
plt.xlabel('Order Quantity')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(f'{OUTPUT_PATH}/demand_distribution.png')
plt.show()

# Demand Trend Over Time
daily_demand = df.groupby('order_date')['order_quantity'].sum().reset_index()
plt.figure(figsize=(14, 5))
sns.lineplot(data=daily_demand, x='order_date', y='order_quantity', color='steelblue')
plt.title('Daily Demand Trend Over Time')
plt.xlabel('Order Date')
plt.ylabel('Total Quantity Ordered')
plt.tight_layout()
plt.savefig(f'{OUTPUT_PATH}/demand_trend.png')
plt.show()

# Demand by region
region_demand = (df.groupby('buyer_region')['order_quantity']
                 .sum()
                 .sort_values(ascending=False)
                 .head(10)
                 .reset_index())
plt.figure(figsize=(10, 6))
sns.barplot(data=region_demand, x='order_quantity', y='buyer_region', palette='Oranges_r')
plt.title('Top 10 Regions by Demand')
plt.xlabel('Total Quantity Ordered')
plt.ylabel('Region')
plt.tight_layout()
plt.savefig(f'{OUTPUT_PATH}/demand_by_region.png')
plt.show()

#  Monthly Demand Pattern
monthly_demand = df.groupby('order_month')['order_quantity'].sum().reset_index()
plt.figure(figsize=(10, 5))
sns.barplot(data=monthly_demand, x='order_month', y='order_quantity', palette='coolwarm')
plt.title('Monthly Demand Pattern (Seasonality)')
plt.xlabel('Month')
plt.ylabel('Total Quantity Ordered')
plt.tight_layout()
plt.savefig(f'{OUTPUT_PATH}/monthly_seasonality.png')
plt.show()

# yearly demand pattern
yearly_demand = df.groupby('order_year')['order_quantity'].sum().reset_index()
plt.figure(figsize=(10, 5))
sns.barplot(data=yearly_demand, x='order_year', y='order_quantity', palette='coolwarm')
plt.title('Yearly Demand Pattern (Seasonality)')
plt.xlabel('Year')
plt.ylabel('Total Quantity Ordered')
plt.tight_layout()
plt.savefig(f'{OUTPUT_PATH}/yearly_seasonality.png')
plt.show()
# print(yearly_demand)

# Price and Demand
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='price_per_unit', y='order_quantity', alpha=0.3, color='coral')
plt.title('Price per Unit vs Order Quantity (Demand)')
plt.xlabel('Price per Unit')
plt.ylabel('Order Quantity')
plt.tight_layout()
plt.savefig(f'{OUTPUT_PATH}/price_vs_demand.png')
plt.show()


# Correlation Heatmap
cols = ['order_quantity', 'price_per_unit', 'order_value_ghs', 'order_month', 'order_day']

# handling both object and StringDtype
corr_df = df[cols].apply(
    lambda x: pd.factorize(x)[0] if x.dtype == object or pd.api.types.is_string_dtype(x) else x
)

plt.figure(figsize=(8, 6))
sns.heatmap(
    corr_df.corr(),
    annot=True,
    cmap='YlOrRd',
    fmt='.2f'
)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig(f'{OUTPUT_PATH}/correlation_heatmap.png')
plt.show()