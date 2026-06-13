"""Decisive diagnostic: how strong is the raw geohash x timestamp signal?
Builds a PURE lookup submission (no model) and measures cross-day signal where possible."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.metrics import r2_score

tr=pd.read_csv("dataset/train.csv"); te=pd.read_csv("dataset/test.csv")
GM=tr.demand.mean()

# --- Build hierarchical (geohash,timestamp) lookup from ALL train ---
def lut(df,keys): return df.groupby(keys)['demand'].mean()
L_ghts = lut(tr,['geohash','timestamp'])
tr['gh5']=tr.geohash.str[:5]; te['gh5']=te.geohash.str[:5]
tr['gh4']=tr.geohash.str[:4]; te['gh4']=te.geohash.str[:4]
L_g5ts = lut(tr,['gh5','timestamp'])
L_g4ts = lut(tr,['gh4','timestamp'])
L_ts   = lut(tr,['timestamp'])

def predict(df):
    p = np.asarray(df.set_index(['geohash','timestamp']).index.map(L_ghts), dtype=float)
    for keys,L in [(['gh5','timestamp'],L_g5ts),(['gh4','timestamp'],L_g4ts),(['timestamp'],L_ts)]:
        m = np.isnan(p)
        if m.any():
            fill = np.asarray(df.set_index(keys).index.map(L), dtype=float)
            p[m] = fill[m]
    p[np.isnan(p)] = GM
    return p

pred_te = np.clip(predict(te),0,1)
sub=pd.DataFrame({'Index':te.Index.values,'demand':pred_te})
sub.to_csv("submission_pure_lookup.csv",index=False)
print("wrote submission_pure_lookup.csv", sub.shape)
print("coverage exact gh+ts:", te.set_index(['geohash','timestamp']).index.map(L_ghts).notna().mean())

# --- Honest cross-day check on the ONLY overlap we have (night slots) ---
d48=tr[tr.day==48]; d49=tr[tr.day==49]
L48=d48.groupby(['geohash','timestamp'])['demand'].mean()
m=d49.copy(); m['p']=m.set_index(['geohash','timestamp']).index.map(L48).astype(float)
m=m.dropna(subset=['p'])
print("\nCROSS-DAY (night-only overlap) day48->day49 pure lookup R2:", round(r2_score(m.demand,m.p),4))
print("n overlap:", len(m))

# Within-day48 sanity: does gh+ts nearly determine demand WITHIN a day?
# (multiple rows same gh+ts within day48?)
cnt=d48.groupby(['geohash','timestamp']).size()
print("\nday48 (gh,ts) groups with >1 row:", (cnt>1).sum(), "of", len(cnt))
dup=d48.groupby(['geohash','timestamp']).filter(lambda g: len(g)>1)
if len(dup):
    gm=dup.groupby(['geohash','timestamp'])['demand'].transform('mean')
    print("within-day48 R2 of group-mean vs actual (duplicates only):", round(r2_score(dup.demand,gm),4))
