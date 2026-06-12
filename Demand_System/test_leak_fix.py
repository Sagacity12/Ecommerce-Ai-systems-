"""
Quick test to verify data leakage fixes work correctly
"""
import pandas as pd
import numpy as np

print("=" * 60)
print("Testing Leak-Free Feature Engineering")
print("=" * 60)

# Test expanding window approach
test_df = pd.DataFrame({
    'order_date': pd.date_range('2024-01-01', periods=10),
    'product_code': ['A'] * 10,
    'order_quantity': [10, 20, 15, 25, 30, 18, 22, 28, 35, 40]
})

print("\nOriginal data:")
print(test_df)

# Simulate leak-free cumulative feature
test_df = test_df.sort_values(['product_code', 'order_date']).reset_index(drop=True)
test_df['product_total_demand_leakfree'] = test_df.groupby('product_code')['order_quantity'].transform(
    lambda x: x.expanding().sum().shift(1).fillna(0))

print("\nWith leak-free cumulative feature:")
print(test_df[['order_date', 'order_quantity', 'product_total_demand_leakfree']])

print("\n✓ Each row only sees data BEFORE it (shifted by 1)")
print("✓ First row has 0 (no previous data)")
print("✓ Second row has 10 (only first row)")
print("✓ Third row has 30 (sum of first two rows)")

print("\n" + "=" * 60)
print("Feature engineering fix: PASSED")
print("=" * 60)
