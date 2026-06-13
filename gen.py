"""Generalization-focused model. Validated against the day48->day49 proxy
(train on day48, predict day49 night) which has tracked the real LB ranking so far.
Features: vectorized spatial KNN smoothing, structural x slot interactions,
robust smoothed geohash x timestamp encoding. No test labels used.
Keeps submission.csv (LB 87) as the floor; writes submission_gen.csv."""
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

def prep(df):
    cache={g:gdec(g) for g in df.geohash.unique()}
    df['lat']=df.geohash.map(lambda g:cache[g][0]); df['lon']=df.geohash.map(lambda g:cache[g][1])
    p=df.timestamp.str.split(':',expand=True).astype(int)
    df['mins']=p[0]*60+p[1]; df['slot']=df['mins']//15; df['hour']=p[0]
    df['t_sin']=np.sin(2*np.pi*df['mins']/1440); df['t_cos']=np.cos(2*np.pi*df['mins']/1440)
    df['gh5']=df.geohash.str[:5]; df['gh4']=df.geohash.str[:4]; df['gh3']=df.geohash.str[:3]
    df['LV']=(df.LargeVehicles=='Allowed').astype(int); df['LM']=(df.Landmarks=='Yes').astype(int)
    return df

train=prep(pd.read_csv("dataset/train.csv")); test=prep(pd.read_csv("dataset/test.csv"))
TARGET='demand'; GM=train.demand.mean()
for c in ['RoadType','Weather']:
    cats=pd.Categorical(pd.concat([train[c],test[c]]))
    train[c]=pd.Categorical(train[c],categories=cats.categories).codes
    test[c]=pd.Categorical(test[c],categories=cats.categories).codes
full=pd.concat([train,test],ignore_index=True)
mt=full.groupby('geohash')['Temperature'].transform('median')
full['Temperature']=full['Temperature'].fillna(mt).fillna(full['Temperature'].median())
train['Temperature']=full['Temperature'].iloc[:len(train)].values
test['Temperature']=full['Temperature'].iloc[len(train):].values

# ---------- vectorized spatial KNN smoothing ----------
all_gh=sorted(set(train.geohash)|set(test.geohash))
gh_idx={g:i for i,g in enumerate(all_gh)}
coords=np.array([gdec(g) for g in all_gh])
all_ts=sorted(set(train.timestamp)|set(test.timestamp))
ts_idx={t:i for i,t in enumerate(all_ts)}
K=8
nn=NearestNeighbors(n_neighbors=K+1).fit(coords)
_,nbr=nn.kneighbors(coords); nbr=nbr[:,1:]  # exclude self

def spatial_feat(stats, lookup):
    mat=np.full((len(all_gh),len(all_ts)),np.nan)
    gm=stats.groupby(['geohash','timestamp'])[TARGET].mean()
    for (g,t),v in gm.items():
        mat[gh_idx[g],ts_idx[t]]=v
    sp=np.nanmean(mat[nbr],axis=1)          # (n_gh, n_ts) neighbor mean
    gi=lookup.geohash.map(gh_idx).values; ti=lookup.timestamp.map(ts_idx).values
    out=sp[gi,ti]
    return np.where(np.isnan(out),GM,out)

# ---------- encoders ----------
def sm(df,keys,m):
    g=df.groupby(keys)[TARGET].agg(['mean','count'])
    return (g['mean']*g['count']+GM*m)/(g['count']+m)
KEYS={'te_gh':(['geohash'],5.),'te_gh5':(['gh5'],4.),'te_gh4':(['gh4'],4.),
      'te_ts':(['timestamp'],5.),'te_gh_ts':(['geohash','timestamp'],4.),
      'te_gh5_ts':(['gh5','timestamp'],4.),'te_gh_hour':(['geohash','hour'],3.),
      'te_road_slot':(['RoadType','slot'],8.),'te_lanes_slot':(['NumberofLanes','slot'],8.),
      'te_weather_slot':(['Weather','slot'],10.),'te_lm_slot':(['LM','slot'],10.),
      'te_lv_slot':(['LV','slot'],10.),'te_road_gh':(['RoadType','geohash'],5.)}

def add_te(stats, frames):
    for n,(keys,m) in KEYS.items():
        enc=sm(stats,keys,m)
        for f in frames:
            f[n]=np.asarray(f.set_index(keys).index.map(enc),dtype=float)
            f[n]=f[n].fillna(GM)

def add_lags(stats, frames):
    g=stats.groupby(['geohash','slot'])[TARGET].mean()
    for off,nm in [(-1,'p1'),(1,'n1'),(-2,'p2'),(2,'n2')]:
        for f in frames:
            k=list(zip(f.geohash,(f.slot+off)%96))
            f['lag_'+nm]=np.asarray(pd.Index(k).map(g),dtype=float); f['lag_'+nm]=f['lag_'+nm].fillna(GM)

BASE=['lat','lon','mins','slot','hour','t_sin','t_cos','RoadType','NumberofLanes','LV','LM','Temperature','Weather']
LAGS=['lag_p1','lag_n1','lag_p2','lag_n2']
FEAT=BASE+list(KEYS)+LAGS+['sp_knn']
params=dict(objective='regression',n_estimators=2500,learning_rate=0.02,num_leaves=95,
            subsample=0.8,subsample_freq=1,colsample_bytree=0.7,reg_alpha=0.3,reg_lambda=0.3,
            min_child_samples=40,random_state=SEED,n_jobs=-1,verbose=-1)

# ---------- PROXY: train day48 -> predict day49 (night) ----------
d48=train[train.day==48].copy(); d49=train[train.day==49].copy()
add_te(d48,[d48,d49]); add_lags(d48,[d48,d49])
d48['sp_knn']=spatial_feat(d48,d48); d49['sp_knn']=spatial_feat(d48,d49)
m=lgb.LGBMRegressor(**params); m.fit(d48[FEAT],d48[TARGET])
proxy=r2_score(d49[TARGET],m.predict(d49[FEAT]))
print("PROXY day48->day49 R2 =",round(proxy,4),"  (smoothed-model baseline was ~0.55)")

# ---------- FINAL: 5-fold on all train, predict test ----------
add_te(train,[train,test]); add_lags(train,[train,test])
train['sp_knn']=spatial_feat(train,train); test['sp_knn']=spatial_feat(train,test)
y=train[TARGET].values
oof=np.zeros(len(train)); tp=np.zeros(len(test))
for tri,vai in KFold(5,shuffle=True,random_state=SEED).split(train):
    mdl=lgb.LGBMRegressor(**params)
    mdl.fit(train[FEAT].iloc[tri],y[tri],eval_set=[(train[FEAT].iloc[vai],y[vai])],
            callbacks=[lgb.early_stopping(100,verbose=False)])
    oof[vai]=mdl.predict(train[FEAT].iloc[vai]); tp+=mdl.predict(test[FEAT])/5
print("OOF R2 =",round(r2_score(y,oof),4))
pd.DataFrame({'Index':test.Index.values,'demand':np.clip(tp,0,1)}).to_csv("submission_gen.csv",index=False)
print("wrote submission_gen.csv")
imp=pd.Series(mdl.feature_importances_,index=FEAT).sort_values(ascending=False)
print("top features:\n",imp.head(10).to_string())
