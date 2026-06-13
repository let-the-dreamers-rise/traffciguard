"""
Gridlock Hackathon 2.0 - Round 1
Traffic demand forecasting (normalized demand regression).

Approach (fully reproducible, no test-label leakage):
  - Decode geohash -> approximate lat/lon (continuous spatial signal)
  - Parse timestamp -> minutes of day + cyclic sin/cos encoding
  - Target ENCODE the dominant spatio-temporal interaction (geohash x timestamp)
    using OUT-OF-FOLD means for train (honest CV) and full-train means for test
  - Fallback target encodings at coarser keys (geohash, prefix x timestamp, timestamp)
  - Impute Temperature via hierarchical group medians
  - LightGBM gradient boosting
Score metric: max(0, 100 * R2)
"""
import warnings
warnings.filterwarnings("ignore", message="Mean of empty slice")
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import lightgbm as lgb

SEED = 42
np.random.seed(SEED)

# ----------------------------------------------------------------------
# Geohash decoder (base32) -> center lat/lon
# ----------------------------------------------------------------------
_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"
_DECODE = {c: i for i, c in enumerate(_BASE32)}

def geohash_decode(gh):
    lat_lo, lat_hi = -90.0, 90.0
    lon_lo, lon_hi = -180.0, 180.0
    even = True
    for ch in gh:
        cd = _DECODE[ch]
        for mask in (16, 8, 4, 2, 1):
            if even:
                mid = (lon_lo + lon_hi) / 2
                if cd & mask:
                    lon_lo = mid
                else:
                    lon_hi = mid
            else:
                mid = (lat_lo + lat_hi) / 2
                if cd & mask:
                    lat_lo = mid
                else:
                    lat_hi = mid
            even = not even
    return (lat_lo + lat_hi) / 2, (lon_lo + lon_hi) / 2

def add_latlon(df):
    cache = {gh: geohash_decode(gh) for gh in df['geohash'].unique()}
    df['lat'] = df['geohash'].map(lambda g: cache[g][0])
    df['lon'] = df['geohash'].map(lambda g: cache[g][1])
    return df

# ----------------------------------------------------------------------
# Time parsing
# ----------------------------------------------------------------------
def parse_time(df):
    parts = df['timestamp'].str.split(':', expand=True).astype(int)
    df['hour'] = parts[0]
    df['minute'] = parts[1]
    df['mins_of_day'] = df['hour'] * 60 + df['minute']
    df['slot'] = df['mins_of_day'] // 15
    ang = 2 * np.pi * df['mins_of_day'] / (24 * 60)
    df['t_sin'] = np.sin(ang)
    df['t_cos'] = np.cos(ang)
    return df

# ----------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------
train = pd.read_csv("dataset/train.csv")
test = pd.read_csv("dataset/test.csv")
TARGET = 'demand'
y = train[TARGET].values

for df in (train, test):
    add_latlon(df)
    parse_time(df)
    df['gh5'] = df['geohash'].str[:5]
    df['gh4'] = df['geohash'].str[:4]
    df['gh3'] = df['geohash'].str[:3]

# ----------------------------------------------------------------------
# Categorical encodings
# ----------------------------------------------------------------------
train['LargeVehicles'] = (train['LargeVehicles'] == 'Allowed').astype(int)
test['LargeVehicles'] = (test['LargeVehicles'] == 'Allowed').astype(int)
train['Landmarks'] = (train['Landmarks'] == 'Yes').astype(int)
test['Landmarks'] = (test['Landmarks'] == 'Yes').astype(int)

for col in ['RoadType', 'Weather']:
    cats = pd.Categorical(pd.concat([train[col], test[col]]))
    train[col] = pd.Categorical(train[col], categories=cats.categories).codes
    test[col] = pd.Categorical(test[col], categories=cats.categories).codes  # -1 = missing

# ----------------------------------------------------------------------
# Temperature imputation: hierarchical group medians
# ----------------------------------------------------------------------
full = pd.concat([train, test], ignore_index=True)
med_gh_slot = full.groupby(['geohash', 'slot'])['Temperature'].transform('median')
med_gh = full.groupby('geohash')['Temperature'].transform('median')
med_slot = full.groupby('slot')['Temperature'].transform('median')
glob = full['Temperature'].median()
full['Temperature'] = (full['Temperature']
                       .fillna(med_gh_slot).fillna(med_gh)
                       .fillna(med_slot).fillna(glob))
train['Temperature'] = full['Temperature'].iloc[:len(train)].values
test['Temperature'] = full['Temperature'].iloc[len(train):].values

# ----------------------------------------------------------------------
# Target encoding with smoothing
# ----------------------------------------------------------------------
GLOBAL_MEAN = y.mean()

def smooth_map(df, keys, target, m):
    g = df.groupby(keys)[target].agg(['mean', 'count'])
    g['enc'] = (g['mean'] * g['count'] + GLOBAL_MEAN * m) / (g['count'] + m)
    return g['enc']

TE_KEYS = {
    'te_gh_ts':  (['geohash', 'timestamp'], 1.0),
    'te_gh5_ts': (['gh5', 'timestamp'], 2.0),
    'te_gh4_ts': (['gh4', 'timestamp'], 3.0),
    'te_gh3_ts': (['gh3', 'timestamp'], 4.0),
    'te_gh':     (['geohash'], 3.0),
    'te_ts':     (['timestamp'], 5.0),
    'te_gh5':    (['gh5'], 3.0),
    'te_road_ts':(['RoadType', 'slot'], 5.0),
}

# Temporal-neighbor encodings: demand at the SAME geohash, adjacent 15-min slots.
# Captures local rush-hour smoothness. Built from a (geohash, slot) mean table so
# we can look up slot-1 / slot+1 / 3-slot rolling mean for every row.
def neighbor_feats(stats_df, lookup_df, m):
    g = stats_df.groupby(['geohash', 'slot'])[TARGET].agg(['mean', 'count'])
    g['enc'] = (g['mean'] * g['count'] + GLOBAL_MEAN * m) / (g['count'] + m)
    enc = g['enc']
    out = {}
    for off, nm in [(-1, 'prev'), (1, 'next'), (-2, 'prev2'), (2, 'next2')]:
        key = list(zip(lookup_df['geohash'], (lookup_df['slot'] + off) % 96))
        out[f'te_ghslot_{nm}'] = pd.Index(key).map(enc).astype(float)
    cur = pd.Index(list(zip(lookup_df['geohash'], lookup_df['slot']))).map(enc).astype(float)
    stack = np.vstack([out['te_ghslot_prev'].values, cur.values, out['te_ghslot_next'].values])
    with np.errstate(invalid='ignore'):
        roll = np.where(np.isnan(stack).all(axis=0), np.nan, np.nanmean(stack, axis=0))
    out['te_ghslot_roll3'] = roll
    return out

NB_COLS = ['te_ghslot_prev', 'te_ghslot_next', 'te_ghslot_prev2',
           'te_ghslot_next2', 'te_ghslot_roll3']

# Out-of-fold encoding for train (honest), full-train encoding for test
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
for name in list(TE_KEYS) + NB_COLS:
    train[name] = np.nan
for tr_idx, va_idx in kf.split(train):
    fold_tr = train.iloc[tr_idx]
    for name, (keys, m) in TE_KEYS.items():
        enc = smooth_map(fold_tr, keys, TARGET, m)
        vals = train.iloc[va_idx].set_index(keys).index.map(enc)
        train.iloc[va_idx, train.columns.get_loc(name)] = vals.astype(float)
    nb = neighbor_feats(fold_tr, train.iloc[va_idx], 2.0)
    for nm, vals in nb.items():
        train.iloc[va_idx, train.columns.get_loc(nm)] = vals.values if hasattr(vals, 'values') else vals
for name in list(TE_KEYS) + NB_COLS:
    train[name] = train[name].fillna(GLOBAL_MEAN)

for name, (keys, m) in TE_KEYS.items():
    enc = smooth_map(train, keys, TARGET, m)
    test[name] = test.set_index(keys).index.map(enc).astype(float)
    test[name] = test[name].fillna(GLOBAL_MEAN)
nb = neighbor_feats(train, test, 2.0)
for nm, vals in nb.items():
    test[nm] = (vals.values if hasattr(vals, 'values') else vals)
    test[nm] = test[nm].fillna(GLOBAL_MEAN)

# ----------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------
FEATURES = [
    'lat', 'lon', 'mins_of_day', 'slot', 't_sin', 't_cos', 'hour',
    'RoadType', 'NumberofLanes', 'LargeVehicles', 'Landmarks',
    'Temperature', 'Weather',
] + list(TE_KEYS.keys()) + NB_COLS

Xtr = train[FEATURES]
Xte = test[FEATURES]

params = dict(
    objective='regression',
    metric='l2',
    n_estimators=4000,
    learning_rate=0.02,
    num_leaves=127,
    max_depth=-1,
    min_child_samples=30,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.7,
    reg_alpha=0.2,
    reg_lambda=0.2,
    random_state=SEED,
    n_jobs=-1,
    verbose=-1,
)

# ----------------------------------------------------------------------
# Cross-validated honest estimate of leaderboard score
# ----------------------------------------------------------------------
oof = np.zeros(len(train))
test_pred = np.zeros(len(test))
for fold, (tr_idx, va_idx) in enumerate(kf.split(Xtr)):
    model = lgb.LGBMRegressor(**params)
    model.fit(
        Xtr.iloc[tr_idx], y[tr_idx],
        eval_set=[(Xtr.iloc[va_idx], y[va_idx])],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    oof[va_idx] = model.predict(Xtr.iloc[va_idx])
    test_pred += model.predict(Xte) / kf.n_splits
    print(f"fold {fold} R2 = {r2_score(y[va_idx], oof[va_idx]):.6f}")

cv_r2 = r2_score(y, oof)
print(f"\nOOF CV R2 = {cv_r2:.6f}  ->  score = {max(0, 100*cv_r2):.4f}")

# ----------------------------------------------------------------------
# Final submission
# ----------------------------------------------------------------------
test_pred = np.clip(test_pred, 0, 1)
sub = pd.DataFrame({'Index': test['Index'].values, 'demand': test_pred})
sub.to_csv("submission.csv", index=False)
print("submission shape:", sub.shape)
print(sub.head())
