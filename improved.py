"""Improved honest model. Key new idea: use day-49 NIGHT labels (present in train)
to calibrate the day-49 daytime test predictions (per-geohash day effect).
Plus multi-seed ensemble. All from train; no test labels."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
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

# ---------- DAY-49 calibration features ----------
# day49 night timestamps present in train; compute per-geohash day-effect vs day48 at SAME timestamps
d48=train[train.day==48]; d49=train[train.day==49]
night_ts=d49.timestamp.unique()
d48n=d48[d48.timestamp.isin(night_ts)]
lvl48=d48n.groupby('geohash')['demand'].mean()
lvl49=d49.groupby('geohash')['demand'].mean()
ratio_g=(lvl49/lvl48).replace([np.inf,-np.inf],np.nan)
diff_g=(lvl49-lvl48)
# coarser fallback by gh5
lvl48_5=d48n.groupby(d48n.geohash.str[:5])['demand'].mean()
lvl49_5=d49.groupby(d49.geohash.str[:5])['demand'].mean()
ratio_5=(lvl49_5/lvl48_5).replace([np.inf,-np.inf],np.nan)
g_ratio_global=float(np.nanmean(ratio_g.values))
print("median per-gh day49/day48 ratio:",round(float(np.nanmedian(ratio_g.values)),3),
      "n geohash with cal:",ratio_g.notna().sum())

for df in (train,test):
    df['cal_ratio']=df.geohash.map(ratio_g)
    df['cal_ratio']=df['cal_ratio'].fillna(df.gh5.map(ratio_5)).fillna(g_ratio_global)
    df['cal_diff']=df.geohash.map(diff_g).fillna(0.0)
    df['lvl49']=df.geohash.map(lvl49).fillna(df.gh5.map(lvl49_5)).fillna(GM)
    df['lvl48n']=df.geohash.map(lvl48).fillna(df.gh5.map(lvl48_5)).fillna(GM)

# ---------- target encodings (OOF) ----------
def sm(df,keys,m):
    g=df.groupby(keys)[TARGET].agg(['mean','count'])
    return (g['mean']*g['count']+GM*m)/(g['count']+m)
KEYS={'te_gh_ts':(['geohash','timestamp'],1.),'te_gh5_ts':(['gh5','timestamp'],2.),
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
# calibrated lookup feature: day48 gh-ts value * day49 ratio
for df in (train,test):
    df['te_gh_ts_cal']=df['te_gh_ts']*df['cal_ratio']

g=train.groupby(['geohash','slot'])[TARGET].mean()
for off,nm in [(-1,'p1'),(1,'n1'),(-2,'p2'),(2,'n2')]:
    for df in (train,test):
        k=list(zip(df.geohash,(df.slot+off)%96)); df['lag_'+nm]=np.asarray(pd.Index(k).map(g),dtype=float).astype(float)
        df['lag_'+nm]=df['lag_'+nm].fillna(GM)

FEAT=['lat','lon','mins','slot','hour','t_sin','t_cos','RoadType','NumberofLanes','LV','LM',
      'Temperature','Weather']+list(KEYS)+['lag_p1','lag_n1','lag_p2','lag_n2',
      'cal_ratio','cal_diff','lvl49','lvl48n','te_gh_ts_cal']

def run(seed):
    params=dict(objective='regression',n_estimators=3000,learning_rate=0.02,num_leaves=127,
                subsample=0.8,subsample_freq=1,colsample_bytree=0.7,reg_alpha=0.2,reg_lambda=0.2,
                min_child_samples=30,random_state=seed,n_jobs=-1,verbose=-1)
    oof=np.zeros(len(train)); tp=np.zeros(len(test))
    for tri,vai in KFold(5,shuffle=True,random_state=seed).split(train):
        m=lgb.LGBMRegressor(**params)
        m.fit(train[FEAT].iloc[tri],y[tri],eval_set=[(train[FEAT].iloc[vai],y[vai])],
              callbacks=[lgb.early_stopping(100,verbose=False)])
        oof[vai]=m.predict(train[FEAT].iloc[vai]); tp+=m.predict(test[FEAT])/5
    return oof,tp

oof_all=np.zeros(len(train)); tp_all=np.zeros(len(test)); seeds=[42,7,2024]
for s in seeds:
    o,t=run(s); oof_all+=o/len(seeds); tp_all+=t/len(seeds)
print("ensemble OOF R2:",round(r2_score(y,oof_all),4))

pd.DataFrame({'Index':test.Index.values,'demand':np.clip(tp_all,0,1)}).to_csv("submission_improved.csv",index=False)
print("wrote submission_improved.csv")
