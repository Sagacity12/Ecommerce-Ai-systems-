import warnings
warnings.filterwarnings('ignore')

import os
import seaborn as sns
import matplotlib.pyplot as plt
from utils import load_and_prepare
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


OUTPUT_PATH = 'preprocessing_output'
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs('Dataset', exist_ok=True)

df = load_and_prepare()
# Missing values
print("Missing Values")
print(df.isnull().sum())

df.dropna(subset=['retailer_id'], inplace=True)
df['product_name'] = df['product_name'].fillna('Unknown')
print(df.shape)

# Removing Bad records
print("Bad Records")

cancelled = df['order_id'].astype(str).str.startswith('C').sum()
df = df[~df['order_id'].astype(str).str.startswith('C')]
print(f"Cancelled Records: {cancelled}")

neg_qty = (df['order_quantity'] <= 0).sum()
df = df[df['order_quantity'] > 0]
print(f"Negative Qty: {neg_qty}")

neg_price = (df['price_per_unit'] <= 0).sum()
df = df[df['price_per_unit'] > 0]
print(f"Negative Price: {neg_price}")

print("After Bad Records:",  df.shape)

# Dropin duplicate records
print("Duplicates")
dupes = df.duplicated().sum()
df.drop_duplicates(inplace=True)
print(f"Duplicates: {dupes} | Shape: {df.shape}")

# Removing Outlier
print("\n----- Outlier Removal (IQR) -----")

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    before = len(df)
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    print(f"{column}: removed {before - len(df)} | range [{lower_bound:.2f} -> {upper_bound:.2f}]")
    return df

df = remove_outliers_iqr(df, 'order_quantity')
df = remove_outliers_iqr(df, 'price_per_unit')
df = remove_outliers_iqr(df, 'order_value_ghs')
print("After outlier Removal:", df.shape)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, ['order_quantity', 'price_per_unit', 'order_value_ghs']):
    sns.boxplot(y=df[col], ax=ax, color='steelblue')
    ax.set_title(f'{col} (cleaned)')
plt.suptitle('Boxplots After Outlier Removal')
plt.tight_layout()
plt.savefig(f'{OUTPUT_PATH}/boxplots_cleaned.png')
plt.show()

# Encoding the Categorical columns
print("\n── Encode Categorical Columns")
le_region = LabelEncoder()
le_day = LabelEncoder()
df['buyer_region_encoded'] = le_region.fit_transform(df['buyer_region'])
df['order_day_encoded']    = le_day.fit_transform(df['order_day'])
print(df[['buyer_region', 'buyer_region_encoded']].drop_duplicates().sort_values('buyer_region_encoded'))

# Time - Aware Train/Test Split
print("\n---- Time-Aware Train/Test Split")

df.sort_values('order_date', inplace=True)
df.reset_index(drop=True, inplace=True)

split_index = int(len(df) * 0.8)
train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

print(f"Train: {train_df.shape} | {train_df['order_date'].min().date()} → {train_df['order_date'].max().date()}")
print(f"Test : {test_df.shape}  | {test_df['order_date'].min().date()} → {test_df['order_date'].max().date()}")


# Scaling Numerical Features
print("\n── Scale Numerical Features")
scaler = MinMaxScaler()
train_df[['price_per_unit_scaled', 'order_value_scaled']] = scaler.fit_transform(
    train_df[['price_per_unit', 'order_value_ghs']]
)
test_df[['price_per_unit_scaled', 'order_value_scaled']] = scaler.transform(
    test_df[['price_per_unit', 'order_value_ghs']]
)
print("Scaled: price_per_unit, order_value_ghs")

df.to_csv('Dataset/data_cleaned.csv', index=False)
train_df.to_csv('Dataset/train.csv', index=False)
test_df.to_csv('Dataset/test.csv', index=False)

print("\n── PREPROCESSING COMPLETE")
print(f"Final shape : {df.shape}")
print(f"Columns     : {df.columns.tolist()}")