import warnings

from Demand_System.adaptive import AdaptiveDemandModel, UserProfileStore

warnings.filterwarnings("ignore")

import os
import json
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


os.makedirs('outputs/models',   exist_ok=True)
os.makedirs('outputs/plots',    exist_ok=True)



# Loading the Data
print("\n--- Load Data")

df = pd.read_csv('Dataset/data_sentiment.csv', encoding_errors='replace')
df['order_date'] = pd.to_datetime(df['order_date'])
df = df.drop_duplicates()
df.sort_values('order_date', inplace=True)
df.reset_index(drop=True, inplace=True)
print("Shape: ", df.shape)

# Feature And Target
print("\n--: Featured & Target")

FEATURES = [
# RETAILER
    'price_per_unit',
    'order_month',
    'order_quarter',
    'order_week',
    'day_of_week_num',
    'is_weekend',
    'buyer_region_encoded',
    'order_day_encoded',
    'lag_7',
    'lag_14',
    'lag_30',
    'rolling_7_avg',
    'rolling_30_avg',
    'rolling_7_std',
    'is_promoted',
    'price_x_promotion',
    'discount_depth',
    'sentiment_score',
    'retailer_order_count',
    'retailer_avg_quantity',
    'retailer_total_spend',

    # WHOLESALER
    'product_total_demand',
    'product_unique_buyers',
    'product_order_freq',
    'product_revenue',
    'region_avg_order_value',
    'days_since_last_order',
    'is_slow_mover',
]

TARGET = 'order_quantity'

df = df.dropna(subset=FEATURES + [TARGET])
print(f"Shape after dropna : {df.shape}")
print(f"Features           : {len(FEATURES)}")
print(f"Retailer features  : 18")
print(f"Wholesaler features: 7")

# Time Aware Train/Test split
print("\n---: Time-Aware Train/Test Split")

split_index = int(len(df) * 0.8)
train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

X_train = train_df[FEATURES]
y_train = train_df[TARGET]
X_test = test_df[FEATURES]
y_test = test_df[TARGET]

print(f"Train : {X_train.shape} | {train_df['order_date'].min().date()} → {train_df['order_date'].max().date()}")
print(f"Test  : {X_test.shape}  | {test_df['order_date'].min().date()} → {test_df['order_date'].max().date()}")

# Evaluation
def evaluate(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n{'=' * 45}")
    print(f"  {name}")
    print(f"{'=' * 45}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  R²   : {r2:.4f}")
    return {'Model': name, 'RMSE': round(rmse, 4),
            'MAE': round(mae, 4), 'R2': round(r2, 4)}
results = []

# multiple linear regression as model instantiated or training
print("\n---: Multiple Linear Regression")

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)
results.append(evaluate("Multiple Linear Regression", y_test, lr_preds))
joblib.dump(lr_model, 'outputs/models/linear_regression.pkl')

plt.figure(figsize=(12, 4))
plt.plot(y_test.values[:300], label='Actual',    color='steelblue', alpha=0.7)
plt.plot(lr_preds[:300],      label='Predicted', color='coral',     alpha=0.7)
plt.title('Multiple Linear Regression — Actual vs Predicted')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/plots/lr_actual_vs_predicted.png')
plt.show()

# Using XGBOOST Regression
# To ensemble modeling technique that builds a series  of weak learner, each aimed at correcting the errors of the previous one.
print("\n---: XGBoost Regressor")

#  XGBoost algorithm — an ensemble of 200 decision trees that each learn from the previous one's mistakes.
xgb_model = xgb.XGBRegressor(
    objective= 'reg:squarederror',
    learning_rate= 0.1,
    max_depth= 6,
    n_estimators= 200,
    subsample= 0.8,
    random_state= 42,
    verbose= 0,
)
xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)],verbose=50)
xgb_preds = xgb_model.predict(X_test)
results.append(evaluate("XGBoost Regressor", y_test, xgb_preds))
joblib.dump(xgb_model, 'outputs/models/xgb_regressor.pkl')

feat_imp = pd.Series( xgb_model.feature_importances_, index = FEATURES ).sort_values(ascending = False)

plt.figure(figsize=(12, 8))
sns.barplot(x=feat_imp.values, y=feat_imp.index, palette='Blues_r')
plt.title('XGBoost — Feature Importance (Retailer + Wholesaler)')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('outputs/plots/xgb_feature_importance.png')
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(y_test.values[:300], label='Actual',    color='steelblue', alpha=0.7)
plt.plot(xgb_preds[:300],     label='Predicted', color='coral',     alpha=0.7)
plt.title('XGBoost — Actual vs Predicted')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/plots/xgb_actual_vs_predicted.png')
plt.show()

# Hyperparameter tuning using GridSearch
print("\n---: GridSearchCV")

param_grid = {
'max_depth'        : [3, 6, 9],
    'learning_rate'    : [0.01, 0.05, 0.1],
    'n_estimators'     : [100, 200, 500],
    'colsample_bytree' : [0.7, 1.0],
}

grid_search = GridSearchCV(
    estimator= xgb.XGBRegressor(objective= 'reg:squarederror', random_state= 42, verbose= 0),
    param_grid= param_grid,
    scoring= 'neg_mean_squared_error',
    cv= 3,
    verbose= 2,
    n_jobs= -1,
)
grid_search.fit(X_train, y_train)
print("\nBest Param:", grid_search.best_params_)

tuned_preds = grid_search.predict(X_test)
results.append(evaluate("XGBoost Tuned GridSearch", y_test, tuned_preds))
joblib.dump(grid_search.best_estimator_, 'outputs/models/xgboost_tuned.pkl')

# Long-Short Term Memory  LSTM
print("\n---: LSTM")

scaler_lstm = MinMaxScaler()
X_train_scaled = scaler_lstm.fit_transform(X_train)
X_test_scaled = scaler_lstm.transform(X_test)

X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

lstm_model = Sequential([
    LSTM(128, return_sequences=True,
         input_shape=(1, len(FEATURES))),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
lstm_model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = lstm_model.fit(
    X_train_lstm, y_train, validation_data=(X_test_lstm, y_test),
    epochs= 50,
    batch_size= 64,
    callbacks=[early_stop],
    verbose= 1
)

lstm_preds = lstm_model.predict(X_test_lstm).flatten()
results.append(evaluate("LSTM", y_test, lstm_preds))
lstm_model.save('outputs/models/lstm_model.keras')
joblib.dump(scaler_lstm, 'outputs/models/lstm_scaler.keras')

plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'],     label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('LSTM — Training vs Validation Loss')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/plots/lstm_loss.png')
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(y_test.values[:300], label='Actual',    color='steelblue', alpha=0.7)
plt.plot(lstm_preds[:300],    label='Predicted', color='green',     alpha=0.7)
plt.title('LSTM — Actual vs Predicted')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/plots/lstm_actual_vs_predicted.png')
plt.show()

#  Adaptive System
print("\n  Initialising User Profile Store...")
store = UserProfileStore()

print("\n  Building profiles (Retailer + Wholesaler)...")
sample_orders = train_df.sample(500, random_state=42)
for _, row in sample_orders.iterrows():
    store.update_retailer(row['retailer_id'], row)
    store.update_wholesaler(row['product_code'], row)

store.summary()

# Retailer sample
sample_rid = sample_orders['retailer_id'].iloc[0]
rp = store.get_retailer(sample_rid)
print(f"\n  Retailer {sample_rid}:")
print(f"    Orders       : {rp['order_count']}")
print(f"    Avg quantity : {rp['avg_quantity']:.2f}")
print(f"    Avg value    : GHS {rp['avg_order_value']:.2f}")
print(f"    Top product  : {max(rp['favourite_products'], key=rp['favourite_products'].get)}")
if rp['reorder_pattern']:
    print(f"    Avg reorder gap: {np.mean(rp['reorder_pattern']):.1f} days")

# Wholesaler sample
sample_pid = sample_orders['product_code'].iloc[0]
wp = store.get_wholesaler(sample_pid)
print(f"\n  Wholesaler product {sample_pid}:")
print(f"    Total orders     : {wp['total_orders']}")
print(f"    Total qty sold   : {wp['total_quantity_sold']:.0f}")
print(f"    Total revenue    : GHS {wp['total_revenue']:.2f}")
print(f"    Unique buyers    : {len(wp['unique_buyers'])}")
print(f"    Is slow mover    : {wp['is_slow_mover']}")

# Slow movers
slow = store.get_slow_movers()
print(f"\n  Slow movers (SRS FR-CAT-11): {len(slow)}")

# Top retailers
print("\n  Top 5 Retailers:")
for rid, rdata in store.get_top_retailers(5):
    print(f"    {rid} → {rdata['order_count']} orders | "
          f"avg qty: {rdata['avg_quantity']:.1f} | "
          f"avg value: GHS {rdata['avg_order_value']:.2f}")

# Online learning simulation
print("\n  Simulating new incoming orders...")
adaptive_model = AdaptiveDemandModel(FEATURES)
adaptive_model.initial_fit(X_train, y_train)

new_orders = test_df.sample(20, random_state=99)
for _, row in new_orders.iterrows():
    X_new = pd.DataFrame([row[FEATURES]])
    y_new = pd.Series([row[TARGET]])
    adaptive_model.update(X_new, y_new)
    store.update_retailer(row['retailer_id'], row)
    store.update_wholesaler(row['product_code'], row)

print("\n  Adaptive update log:")
adaptive_model.show_log()

adaptive_preds = adaptive_model.predict(X_test)
results.append(evaluate("Adaptive SGD (Online Learning)", y_test, adaptive_preds))

pd.DataFrame(adaptive_model.log).to_csv(
    'outputs/adaptive/adaptive_update_log.csv', index=False)


# MODEL COMPARISON
print("\n── Model Comparison")

results_df = pd.DataFrame(results)
print("\n", results_df.to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, metric, color in zip(axes,
                              ['RMSE', 'MAE', 'R2'],
                              ['steelblue', 'coral', 'seagreen']):
    sns.barplot(data=results_df, x='Model', y=metric, ax=ax, color=color)
    ax.set_title(metric)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25,
                       ha='right', fontsize=7)
    ax.set_xlabel('')
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=7)

plt.suptitle('Model Comparison — Retailer + Wholesaler Features',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/plots/model_comparison.png')
plt.show()

results_df.to_csv('outputs/model_results.csv', index=False)

print("\n── MODELING COMPLETE")
print(results_df.to_string(index=False))
print("\nSaved:")
print("  outputs/models/          — all model files")
print("  outputs/adaptive/        — retailer + wholesaler profiles")
print("  outputs/plots/           — all visualizations")
print("  outputs/model_results.csv")