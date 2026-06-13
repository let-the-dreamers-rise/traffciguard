"""Generalization-focused model (the lever that's actually working):
  - Vectorized spatial KNN smoothing: mean demand of nearby geohashes at same timestamp
  - Structural interactions: {RoadType,Lanes,Weather,Landmarks,LargeVeh} x slot (target-encoded)
  - Robust smoothed geohash x timestamp encoding (generalize, don't memorize)
  - Multi-seed LightGBM ensemble
Reports OOF and the day48->day49 night proxy. All from train; no test labels."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.neighbors import NearestNeighbors
import lightgbm as lgb

SEED=42
_B32="0123456789bcdefghjkmnpqrstuvwxyz"; _D={c:i for i,c in enumerate(_B32)}
def gdec(gh):
    a,b,c,d=-90.,90.,-180.,180.; ev=True
    for ch in gh:
        cd=_D[ch]
        for mask in (16,8,4,2,1):
            if ev:
                m=(c+d)/2; c,d=(m,d) if cd&mask else (c,m)
            else:
                m=(a+b)/2; a,b=(m,b) if cd&mask else (a,m)
            ev=not ev
    return (a+b)/2,(c+d)/2

train=pd.read_csv("dataset/train.csv"); test=pd.read_csv("dataset/test.csv")
TARGET='demand'; y=train[TARGET].values; GM=y.mean()
for df in (train,test):
    cache={g:gdec(g) for g in df.geohash.unique()}
    df['lat']=df.geohash.map(lambda g:cache[g][0]); df['lon']=df.geohash.map(lambda g:cache[g][1])
    p=df.timestamp.str.split(':',expand=True).astype(int)
    df['mins']=p[0]*60+p[1]; df['slot']=df['mins']//15; df['hour']=p[0]
    df['t_sin']=np.sin(2*np.pi*df['mins']/1440); df['t_cos']=np.cos(2*np.pi*df['mins']/1440)
    df['gh5']=df.geohash.str[:5]; df['gh4']=df.geohash.str[:4]; df['gh3']=df.geohash.str[:3]
    df['LV']=(df.LargeVehicles=='Allowed').astype(int); df['LM']=(df.Landmarks=='Yes').astype(int)
for c in ['RoadType','Weather']:
    cats=pd.Categorical(pd.concat([train[c],test[c]]))
    train[c]=pd.Categorical(train[c],categories=cats.categories).codes
    test[c]=pd.Categorical(test[c],categories=cats.categories).codes
full=pd.concat([train,test],ignore_index=True)
mt=full.groupby('geohash')['Temperature'].transform('median')
full['Temperature']=full['Temperature'].fillna(mt).fillna(full['Temperature'].median())
train['Temperature']=full['Temperature'].iloc[:len(train)].values
test['Temperature']=full['Temperature'].iloc[len(train):].values

# ---------- vectorized spatial KNN smoothing (same timestamp) ----------
ghxy=pd.concat([train[['geohash','lat','lon']],test[['geohash','lat','lon']]]).drop_duplicates('geohash').set_index('geohash')
K=9
nn=NearestNeighbors(n_neighbors=min(K,len(ghxy))).fit(ghxy[['lat','lon']].values)
_,idx=nn.kneighbors(ghxy[['lat','lon']].values)
ghs=ghxy.index.values
neigh=pd.DataFrame(ghs[idx],index=ghs)  # col0=self ... colK-1

def spatial_knn(stats, app):
    base=stats.groupby(['geohash','timestamp'])[TARGET].mean()  # (gh,ts)->mean
    base_d=base.to_dict()
    out=np.zeros(len(app)); cnt=np.zeros(len(app))
    ts=app['timestamp'].values
    gh=app['geohash'].values
    nb=neigh.reindex(gh).values  # (n, K)
    for j in range(1,K):  # skip self
        col=nb[:,j]
        keys=list(zip(col,ts))
        vals=np.array([base_d.get(k,np.nan) for k in keys],dtype=float)
        ok=~np.isnan(vals)
        out[ok]+=vals[ok]; cnt[ok]+=1
    res=np.where(cnt>0,out/np.maximum(cnt,1),GM)
    return res

# ---------- target encodings ----------
def sm(df,keys,m):
    g=df.groupby(keys)[TARGET].agg(['mean','count'])
    return (g['mean']*g['count']+GM*m)/(g['count']+m)
KEYS={'te_gh_ts':(['geohash','timestamp'],2.),'te_gh5_ts':(['gh5','timestamp'],3.),
      'te_gh4_ts':(['gh4','timestamp'],4.),'te_gh3_ts':(['gh3','timestamp'],5.),
      'te_gh':(['geohash'],3.),'te_ts':(['timestamp'],5.),'te_gh_hour':(['geohash','hour'],2.),
      'te_road_ts':(['RoadType','slot'],5.),'te_lanes_ts':(['NumberofLanes','slot'],5.),
