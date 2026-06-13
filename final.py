"""Best honest submission:
  - Exact geohash x timestamp historical mean for rows it covers (~89%)
  - LightGBM (rich features) to fill the rest AND to validate
All features from train only; no test labels used."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import lightgbm as lgb

SEED=42; np.random.seed(SEED)
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

# ---- exact lookup (full train) + coarse hierarchy ----
L_ghts=train.groupby(['geohash','timestamp'])['demand'].mean()
L_g5ts=train.groupby(['gh5','timestamp'])['demand'].mean()
L_g4ts=train.groupby(['gh4','timestamp'])['demand'].mean()
L_ts=train.groupby(['timestamp'])['demand'].mean()
def lookup(df):
    p=np.asarray(df.set_index(['geohash','timestamp']).index.map(L_ghts),dtype=float)
    exact=~np.isnan(p)
    for keys,L in [(['gh5','timestamp'],L_g5ts),(['gh4','timestamp'],L_g4ts),(['timestamp'],L_ts)]:
        m=np.isnan(p)
        if m.any():
            f=np.asarray(df.set_index(keys).index.map(L),dtype=float); p[m]=f[m]
    p[np.isnan(p)]=GM
    return p,exact
te_lk,te_exact=lookup(test)

# ---- model features (with OOF target encodings) ----
def sm(df,keys,m):
    g=df.groupby(keys)[TARGET].agg(['mean','count'])
    return (g['mean']*g['count']+GM*m)/(g['count']+m)
KEYS={'te_gh_ts':(['geohash','timestamp'],0.5),'te_gh5_ts':(['gh5','timestamp'],2.),
      'te_gh4_ts':(['gh4','timestamp'],3.),'te_gh3_ts':(['gh3','timestamp'],4.),
      'te_gh':(['geohash'],3.),'te_ts':(['timestamp'],5.),'te_gh_hour':(['geohash','hour'],1.),
      'te_road_ts':(['RoadType','slot'],5.),'te_lanes_ts':(['NumberofLanes','slot'],5.)}
kf=KFold(5,shuffle=True,random_state=SEED)
for n in KEYS: train[n]=np.nan
for tri,vai in kf.split(train):
    ft=train.iloc[tri]
    for n,(keys,m) in KEYS.items():
        enc=sm(ft,keys,m); train.iloc[vai,train.columns.get_loc(n)]=np.asarray(train.iloc[vai].set_index(keys).index.map(enc),dtype=float)
for n in KEYS: train[n]=train[n].fillna(GM)
for n,(keys,m) in KEYS.items():
    enc=sm(train,keys,m); test[n]=np.asarray(test.set_index(keys).index.map(enc),dtype=float); test[n]=test[n].fillna(GM)
# temporal lags
g=train.groupby(['geohash','slot'])[TARGET].mean()
for off,nm in [(-1,'p1'),(1,'n1'),(-2,'p2'),(2,'n2')]:
    for df in (train,test):
        k=list(zip(df.geohash,(df.slot+off)%96)); df['lag_'+nm]=np.asarray(pd.Index(k).map(g),dtype=float)
        df['lag_'+nm]=df['lag_'+nm].fillna(GM)

FEAT=['lat','lon','mins','slot','hour','t_sin','t_cos','RoadType','NumberofLanes','LV','LM',
      'Temperature','Weather']+list(KEYS)+['lag_p1','lag_n1','lag_p2','lag_n2']
params=dict(objective='regression',n_estimators=3000,learning_rate=0.02,num_leaves=127,
            subsample=0.8,subsample_freq=1,colsample_bytree=0.7,reg_alpha=0.2,reg_lambda=0.2,
            min_child_samples=30,random_state=SEED,n_jobs=-1,verbose=-1)
oof=np.zeros(len(train)); te_model=np.zeros(len(test))
for tri,vai in kf.split(train):
    mdl=lgb.LGBMRegressor(**params)
    mdl.fit(train[FEAT].iloc[tri],y[tri],eval_set=[(train[FEAT].iloc[vai],y[vai])],
            callbacks=[lgb.early_stopping(100,verbose=False)])
    oof[vai]=mdl.predict(train[FEAT].iloc[vai]); te_model+=mdl.predict(test[FEAT])/5
print("model OOF R2:",round(r2_score(y,oof),4))

# ---- combine: exact lookup where available, model elsewhere ----
final=np.where(te_exact, te_lk, te_model)
final=np.clip(final,0,1)
pd.DataFrame({'Index':test.Index.values,'demand':final}).to_csv("submission_final.csv",index=False)
print("exact-covered test rows:",te_exact.mean().round(4))
print("wrote submission_final.csv", (len(final),2))

# also pure model submission for comparison
pd.DataFrame({'Index':test.Index.values,'demand':np.clip(te_model,0,1)}).to_csv("submission_model.csv",index=False)
