import json
import os
from datetime import datetime

import joblib
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import MinMaxScaler

# Adaptive learning System
# WHY ADAPTIVE LEARNING IS NECESSARY FOR THIS PLATFORM
# The models trained in this pipeline were built on the UCI Online Retail/wholesale dataset
# — a UK-based wholesale gift company from 2010 to 2011. This platform serves a
# Ghanaian ecommerce market in 2024 and beyond. That gap is significant.
# The UCI data reflects UK wholesale buying behaviour, Christmas gift demand peaks,
# European seasonal patterns, and GBP-denominated pricing. None of that maps
# directly to Ghanaian retail consumers, local festive cycles like Homowo or
# Chale Wote, West African seasonal demand shifts, or GHS-based pricing dynamics.
# The model learned patterns from a market that is fundamentally different from
# the one this platform operates in.
# Beyond the data mismatch, the model will drift over time. Even if predictions
# are reasonable at launch, the model has no awareness of new products added to
# the catalogue, price changes, shifting buyer behaviour, or supplier restock
# cycles unique to the Ghanaian supply chain. Without retraining, the model
# grows stale and its forecasts become increasingly unreliable.
# Most importantly, every order placed on this platform is ground truth the
# original model was never trained on. A buyer in Greater Accra ordering Ankara
# fabric at the end of the month after a MoMo promotion is a signal no UCI
# record could ever capture. Adaptive learning is what allows the system to
# ingest these transactions continuously and retrain on the patterns that are
# actually emerging in this market — payday demand cycles, regional spikes
# during local festivals, and wholesaler behaviour specific to this supply chain.
#
# A static model trained once on borrowed data is a starting point, not a
# finished product. Left unchanged it will overstock the wrong products, miss
# stockout risks, and lose the trust of the retailers and wholesalers depending
# on its forecasts. Adaptive learning is what closes the gap between the dataset
# this system was bootstrapped on and the real platform it is meant to serve.

print("\n---: Adaptive Learning System")
# User Profile Store (Retailer + Wholesaler)
class UserProfileStore:
    def __init__(self, save_path='outputs/adaptive/user_profiles.json'):
        self.save_path = save_path
        self.profiles = self._load()

    def _load(self):
        if os.path.exists(self.save_path):
            with open(self.save_path, 'r') as f:
                return json.load(f)
        return {'retailers': {}, 'wholesalers': {}}

    def _save(self):
        with open(self.save_path, 'w') as f:
            json.dump(self.profiles, f, indent=2, default=str)

    def update_retailer(self, retailer_id, order):
        rid = str(retailer_id)
        if rid not in self.profiles['retailers']:
            self.profiles['retailers'][rid] = {
                'order_count': 0,
                'avg_quantity': 0.0,
                'avg_order_value': 0.0,
                'favourite_products': {},
                'preferred_regions': {},
                'last_order_date': None,
                'demand_history': [],
                'reorder_pattern': []
            }
        p = self.profiles['retailers'][rid]
        qty = float(order['order_quantity'])
        val = float(order.get('order_value_ghs', 0))

        p['order_count'] += 1
        p['avg_quantity'] = ( (p['avg_quantity'] * (p['order_count'] - 1) + qty) / p['order_count'])
        p['avg_order_value'] = ( (p['avg_order_value'] * (p['order_count'] - 1) + val) / p['order_count'])

        if p['last_order_date']:
            last = pd.to_datetime(p['last_order_date'])
            curr = pd.to_datetime(str(order.get('order_date', '')))
            gap = (curr - last).days
            if gap > 0:
                p['reorder_pattern'].append(gap)
                if len(p['reorder_pattern']) > 10:
                    p['reorder_pattern'].pop(0)

        p['last_order_date'] = str(order.get('order_date', ''))
        prod = str(order.get('product_code', 'unknown'))
        p['favourite_products'][prod] = \
            p['favourite_products'].get(prod, 0) + 1

        region = str(order.get('buyer_region', 'unknown'))
        p['preferred_regions'][region] = \
            p['preferred_regions'].get(region, 0) + 1

        p['demand_history'].append(qty)
        if len(p['demand_history']) > 30:
            p['demand_history'].pop(0)

        self.profiles['retailers'][rid] = p
        self._save()
        return p

    def update_wholesaler(self, product_code, order):
        pid = str(product_code)
        if pid not in self.profiles['wholesalers']:
            self.profiles['wholesalers'][pid] = {
                'total_orders': 0,
                'total_quantity_sold': 0.0,
                'total_revenue': 0.0,
                'unique_buyers': [],
                'avg_order_quantity': 0.0,
                'demand_history': [],
                'buyer_regions': {},
                'days_since_last_sale': 0,
                'last_sale_date': None,
                'is_slow_mover': False
            }
        p = self.profiles['wholesalers'][pid]
        qty = float(order['order_quantity'])
        val = float(order.get('order_value_ghs', 0))
        rid = str(order.get('retailer_id', 'unknown'))

        p['total_orders'] += 1
        p['total_quantity_sold'] += qty
        p['total_revenue'] += val
        p['avg_order_quantity'] = p['total_quantity_sold'] / p['total_orders']

        if rid not in p['unique_buyers']:
            p['unique_buyers'].append(rid)

        region = str(order.get('buyer_region', 'unknown'))
        p['buyer_regions'][region] = \
            p['buyer_regions'].get(region, 0) + 1

        if p['last_sale_date']:
            last = pd.to_datetime(p['last_sale_date'])
            curr = pd.to_datetime(str(order.get('order_date', '')))
            p['days_since_last_sale'] = (curr - last).days
            p['is_slow_mover'] = p['days_since_last_sale'] >= 60

        p['last_sale_date'] = str(order.get('order_date', ''))

        p['demand_history'].append(qty)
        if len(p['demand_history']) > 30:
            p['demand_history'].pop(0)

        self.profiles['wholesalers'][pid] = p
        self._save()
        return p

    def get_retailer(self, retailer_id):
        return self.profiles['retailers'].get(str(retailer_id), None)

    def get_wholesaler(self, product_code):
        return self.profiles['wholesalers'].get(str(product_code), None)

    def get_slow_movers(self):
        return {pid: p for pid, p in self.profiles['wholesalers'].items()
                if p.get('is_slow_mover', False)}

    def get_top_retailers(self, n=5):
        return sorted(
            self.profiles['retailers'].items(),
            key = lambda x: x[1]['order_count'],
            reverse=True
        )[:n]

    def summary(self):
        print(f"  Retailer profiles  : {len(self.profiles['retailers'])}")
        print(f"  Wholesaler products: {len(self.profiles['wholesalers'])}")
        print(f"  Slow movers flagged: {len(self.get_slow_movers())}")

# Using SGDRegressor for the Adaptive Model
class AdaptiveDemandModel:
    def __init__(self, features,
                 save_path='outputs/models/adaptive_model.pkl'):
        self.features = features
        self.save_path = save_path
        self.scaler = MinMaxScaler()
        self._fitted = False
        self.log = []
        self.model = self._load()

    def _load(self):
        if os.path.exists(self.save_path):
            print("Loaded existing adaptive model")
            return joblib.load(self.save_path)

        return SGDRegressor(
            loss = 'squared_error',
            learning_rate = 'adaptive',
            eta0 = 0.01,
            max_iter = 1,
            tol = None,
            warm_start = True,
            random_state = 42,
        )
    def initial_fit(self, X, y):
        print("Initial fit on historical data...")
        X_scaled = self.scaler.fit_transform(X)
        batch_size = 1000
        for i in range(0, len(X_scaled), batch_size):
            self.model.partial_fit(X_scaled[i : i + batch_size], y.iloc[i : i + batch_size])
        self._fitted = True
        self._save()
        print(f" Fit complete - {len(X)} samples")

    def update(self, X_new, y_new):
        if not self._fitted:
            print(" Run initial_fit first")
            return
        X_scaled = self.scaler.transform(X_new)
        y_pred = self.model.predict(X_scaled)
        self.log.append({
            'timestamp' : str(datetime.now()),
            'actual' : float(y_new.values[0]),
            'predicted' : float(y_pred[0]),
            'error' : float(abs(y_new.values[0] - y_pred[0]))
        })
        self.model.partial_fit(X_scaled, y_new)
        self._save()

    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))

    def _save(self):
        joblib.dump(self.model, self.save_path)

    def show_log(self, last_n = 10):
        if not self.log:
            print("No updates log")
            return
        print(pd.DataFrame(self.log).tail(last_n).to_string(index=False))




