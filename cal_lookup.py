"""Explicit day-49 calibrated lookup: predict day49 daytime as
   day48_value(gh,ts) * day49/day48 ratio (per-geohash, night-derived).
Produces several calibration variants to test on the leaderboard."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.metrics import r2_score

train=pd.read_csv("dataset/train.csv"); test=pd.read_csv("dataset/test.csv")
GM=train.demand.mean()
for df in (train,test):
    df['gh5']=df.geohash.str[:5]; df['gh4']=df.geohash.str[:4]
d48=train[train.day==48]; d49=train[train.day==49]
night_ts=d49.timestamp.unique()
d48n=d48[d48.timestamp.isin(night_ts)]

print("GLOBAL day48-night mean:",round(d48n.demand.mean(),4),
      "| day49-night mean:",round(d49.demand.mean(),4),
      "| global ratio:",round(d49.demand.mean()/d48n.demand.mean(),3))

# base day48 lookup (hierarchical)
L_ghts=d48.groupby(['geohash','timestamp'])['demand'].mean()  # day48 only (cleaner ref)
L_g5ts=d48.groupby(['gh5','timestamp'])['demand'].mean()
L_g4ts=d48.groupby(['gh4','timestamp'])['demand'].mean()
L_ts=d48.groupby(['timestamp'])['demand'].mean()
def base(df):
    p=np.asarray(df.set_index(['geohash','timestamp']).index.map(L_ghts),dtype=float)
    for keys,L in [(['gh5','timestamp'],L_g5ts),(['gh4','timestamp'],L_g4ts),(['timestamp'],L_ts)]:
        m=np.isnan(p)
        if m.any(): f=np.asarray(df.set_index(keys).index.map(L),dtype=float); p[m]=f[m]
    p[np.isnan(p)]=GM; return p

# calibration ratios (per-geohash, gh5 fallback, global)
lvl48=d48n.groupby('geohash')['demand'].mean(); lvl49=d49.groupby('geohash')['demand'].mean()
ratio_g=(lvl49/lvl48).replace([np.inf,-np.inf],np.nan)
lvl48_5=d48n.groupby('gh5')['demand'].mean(); lvl49_5=d49.groupby('gh5')['demand'].mean()
ratio_5=(lvl49_5/lvl48_5).replace([np.inf,-np.inf],np.nan)
gr=float(np.nanmedian(ratio_g.values))

def cal(df,clip=None):
    r=df.geohash.map(ratio_g).fillna(df.gh5.map(ratio_5)).fillna(gr).values
    if clip: r=np.clip(r,clip[0],clip[1])
    return r

te_base=base(test)
for tag,rr in [('global',np.full(len(test),d49.demand.mean()/d48n.demand.mean())),
               ('pergh',cal(test)),('pergh_clip',cal(test,(0.5,3.0)))]:
    pred=np.clip(te_base*rr,0,1)
    pd.DataFrame({'Index':test.Index.values,'demand':pred}).to_csv(f"submission_cal_{tag}.csv",index=False)
    print(f"wrote submission_cal_{tag}.csv  mean pred={pred.mean():.4f} (base mean={te_base.mean():.4f})")
