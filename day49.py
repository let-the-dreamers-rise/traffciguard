"""Use day-49 labels directly. Train a model on day-49 NIGHT rows (target=day49 demand),
using day-48 reference values + day-49's own weather/time/structure as features, then
predict day-49 DAYTIME test. Also blend with the day-48-based best model.
Writes submission_day49.csv and submission_blend.csv. No test labels used."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.metrics import r2_score
from sklearn.neighbors import NearestNeighbors
import lightgbm as lgb

_B32="0123456789bcdefghjkmnpqrstuvwxyz"; _D={c:i for i,c in enumerate(_B32)}
def gdec(gh):
    a,b,c,d=-90.,90.,-180.,180.; ev=True
    for ch in gh:
        cd=_D[ch]
        for mask in (16,8,4,2,1):
            if ev: m=(c+d)/2; c,d=(m,d) if cd&mask else (c,m)
            else:  m=(a+b)/2; a,b=(m,b) if cd&mask else (a,m)
            ev=not ev
    return (a+b)/2,(c+d)/2
def prep(df):
    cache={g:gdec(g) for g in df.geohash.unique()}
    df['lat']=df.geohash.map(lambda g:cache[g][0]); df['lon']=df.geohash.map(lambda g:cache[g][1])
    p=df.timestamp.str.split(':',expand=True).astype(int)
    df['mins']=p[0]*60+p[1]; df['slot']=df['mins']//15; df['hour']=p[0]
    df['t_sin']=np.sin(2*np.pi*df['mins']/1440); df['t_cos']=np.cos(2*np.pi*df['mins']/1440)
    df['gh5']=df.geohash.str[:5]; df['gh4']=df.geohash.str[:4]
    df['LV']=(df.LargeVehicles=='Allowed').astype(int); df['LM']=(df.Landmarks=='Yes').astype(int)
    return df
train=prep(pd.read_csv("dataset/train.csv")); test=prep(pd.read_csv("dataset/test.csv"))
GM=train.demand.mean()
for c in ['RoadType','Weather']:
    cats=pd.Categorical(pd.concat([train[c],test[c]]))
    train[c]=pd.Categorical(train[c],categories=cats.categories).codes
    test[c]=pd.Categorical(test[c],categories=cats.categories).codes
for df in (train,test):
    df['Temperature']=df.groupby('geohash')['Temperature'].transform(lambda s:s.fillna(s.median()))
    df['Temperature']=df['Temperature'].fillna(train['Temperature'].median())

d48=train[train.day==48].copy(); d49=train[train.day==49].copy()

# ----- day-48 REFERENCE features (available for night-train AND daytime-test) -----
def ref(stats):
    return {'r_ghts':stats.groupby(['geohash','timestamp'])['demand'].mean(),
            'r_gh':stats.groupby('geohash')['demand'].mean(),
            'r_gh5ts':stats.groupby(['gh5','timestamp'])['demand'].mean(),
            'r_ts':stats.groupby('timestamp')['demand'].mean(),
            'r_ghslot':stats.groupby(['geohash','slot'])['demand'].mean()}
R=ref(d48)
def add_ref(df):
    df['r_ghts']=np.asarray(df.set_index(['geohash','timestamp']).index.map(R['r_ghts']),dtype=float)
    df['r_gh5ts']=np.asarray(df.set_index(['gh5','timestamp']).index.map(R['r_gh5ts']),dtype=float)
    df['r_gh']=df.geohash.map(R['r_gh']); df['r_ts']=df.timestamp.map(R['r_ts'])
    for off,nm in [(-1,'rp1'),(1,'rn1'),(-2,'rp2'),(2,'rn2')]:
        df[nm]=np.asarray(pd.Index(list(zip(df.geohash,(df.slot+off)%96))).map(R['r_ghslot']),dtype=float)
    for c in ['r_ghts','r_gh5ts','r_gh','r_ts','rp1','rn1','rp2','rn2']:
        df[c]=df[c].fillna(df['r_gh5ts']).fillna(df['r_ts']).fillna(GM)
for df in (d49,test,d48): add_ref(df)

FEAT=['lat','lon','mins','slot','hour','t_sin','t_cos','RoadType','NumberofLanes','LV','LM',
      'Temperature','Weather','r_ghts','r_gh5ts','r_gh','r_ts','rp1','rn1','rp2','rn2']

params=dict(objective='regression',n_estimators=1500,learning_rate=0.02,num_leaves=63,
            subsample=0.8,subsample_freq=1,colsample_bytree=0.8,reg_alpha=0.3,reg_lambda=0.3,
            min_child_samples=20,random_state=42,n_jobs=-1,verbose=-1)

# sanity proxy: train on day48 daytime (slot>=9) with day48 ref... circular, skip.
# Train day-49 model on day-49 night rows, ensemble seeds
tp=np.zeros(len(test))
for s in [42,7,2024]:
    p=dict(params); p['random_state']=s
    m=lgb.LGBMRegressor(**p); m.fit(d49[FEAT],d49['demand']); tp+=m.predict(test[FEAT])/3
day49_pred=np.clip(tp,0,1)
pd.DataFrame({'Index':test.Index.values,'demand':day49_pred}).to_csv("submission_day49.csv",index=False)
print("wrote submission_day49.csv  mean=",round(day49_pred.mean(),4))
imp=pd.Series(m.feature_importances_,index=FEAT).sort_values(ascending=False)
print("day49-model top features:\n",imp.head(8).to_string())

# blend with the day-48-based best model
try:
    best=pd.read_csv("submission_best.csv").set_index('Index')['demand']
    bb=test.Index.map(best).values
    for w in [0.3,0.5]:
        bl=np.clip(w*day49_pred+(1-w)*bb,0,1)
        pd.DataFrame({'Index':test.Index.values,'demand':bl}).to_csv(f"submission_blend_{int(w*100)}.csv",index=False)
        print(f"wrote submission_blend_{int(w*100)}.csv  mean=",round(bl.mean(),4))
except Exception as e:
    print("blend skipped:",e)
