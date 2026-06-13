"""Ensemble + dual validation.
Proxy A (cross-day, night): train day48 -> predict day49 night.
Proxy B (daytime fit): hold out half of day48 DAYTIME, predict it (tests daytime quality).
Models: multi-seed LightGBM (l2 + huber) + HistGradientBoosting, blended.
No test labels used. submission.csv (LB87) stays as floor; writes submission_ens.csv."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import HistGradientBoostingRegressor
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

all_gh=sorted(set(train.geohash)|set(test.geohash)); gh_idx={g:i for i,g in enumerate(all_gh)}
coords=np.array([gdec(g) for g in all_gh])
all_ts=sorted(set(train.timestamp)|set(test.timestamp)); ts_idx={t:i for i,t in enumerate(all_ts)}
nn=NearestNeighbors(n_neighbors=9).fit(coords); _,nbr=nn.kneighbors(coords); nbr=nbr[:,1:]
def spatial_feat(stats,lookup):
    mat=np.full((len(all_gh),len(all_ts)),np.nan)
    for (g,t),v in stats.groupby(['geohash','timestamp'])[TARGET].mean().items(): mat[gh_idx[g],ts_idx[t]]=v
    sp=np.nanmean(mat[nbr],axis=1)
    o=sp[lookup.geohash.map(gh_idx).values,lookup.timestamp.map(ts_idx).values]
    return np.where(np.isnan(o),GM,o)
def sm(df,keys,m):
    g=df.groupby(keys)[TARGET].agg(['mean','count']); return (g['mean']*g['count']+GM*m)/(g['count']+m)
KEYS={'te_gh':(['geohash'],5.),'te_gh5':(['gh5'],4.),'te_ts':(['timestamp'],5.),
      'te_gh_ts':(['geohash','timestamp'],4.),'te_gh5_ts':(['gh5','timestamp'],4.),
      'te_gh_hour':(['geohash','hour'],3.),'te_road_slot':(['RoadType','slot'],8.),
      'te_lanes_slot':(['NumberofLanes','slot'],8.),'te_weather_slot':(['Weather','slot'],10.),
      'te_road_gh':(['RoadType','geohash'],5.)}
def add_te(stats,frames):
    for n,(keys,m) in KEYS.items():
        enc=sm(stats,keys,m)
        for f in frames: f[n]=np.asarray(f.set_index(keys).index.map(enc),dtype=float); f[n]=f[n].fillna(GM)
def add_lags(stats,frames):
    g=stats.groupby(['geohash','slot'])[TARGET].mean()
    for off,nm in [(-1,'p1'),(1,'n1'),(-2,'p2'),(2,'n2')]:
        for f in frames:
            k=list(zip(f.geohash,(f.slot+off)%96)); f['lag_'+nm]=np.asarray(pd.Index(k).map(g),dtype=float); f['lag_'+nm]=f['lag_'+nm].fillna(GM)
BASE=['lat','lon','mins','slot','hour','t_sin','t_cos','RoadType','NumberofLanes','LV','LM','Temperature','Weather']
FEAT=BASE+list(KEYS)+['lag_p1','lag_n1','lag_p2','lag_n2','sp_knn']

def lgb_models(seed):
    return [lgb.LGBMRegressor(objective='regression',n_estimators=1200,learning_rate=0.03,num_leaves=95,
              subsample=0.8,subsample_freq=1,colsample_bytree=0.7,reg_alpha=0.3,reg_lambda=0.3,
              min_child_samples=40,random_state=seed,n_jobs=-1,verbose=-1),
            lgb.LGBMRegressor(objective='huber',alpha=0.9,n_estimators=1200,learning_rate=0.03,num_leaves=63,
              subsample=0.7,subsample_freq=1,colsample_bytree=0.7,reg_alpha=0.5,reg_lambda=0.5,
              min_child_samples=50,random_state=seed,n_jobs=-1,verbose=-1)]

def fit_predict(Xtr,ytr,Xval,Xtest=None):
    vp=[]; tp=[]
    for seed in [42,2024]:
        for m in lgb_models(seed):
            m.fit(Xtr,ytr); vp.append(m.predict(Xval))
            if Xtest is not None: tp.append(m.predict(Xtest))
    h=HistGradientBoostingRegressor(max_iter=600,learning_rate=0.03,max_leaf_nodes=63,
        l2_regularization=0.2,early_stopping=True,validation_fraction=0.1,n_iter_no_change=30,random_state=42)
    h.fit(Xtr,ytr); vp.append(h.predict(Xval))
    if Xtest is not None: tp.append(h.predict(Xtest))
    wv=0.85*np.average(np.vstack(vp)[:4],axis=0,weights=[0.3,0.2,0.3,0.2])+0.15*np.vstack(vp)[4]
    if Xtest is None: return wv
    wt=0.85*np.average(np.vstack(tp)[:4],axis=0,weights=[0.3,0.2,0.3,0.2])+0.15*np.vstack(tp)[4]
    return wv,wt

# ---- Proxy A: cross-day night ----
d48=train[train.day==48].copy(); d49=train[train.day==49].copy()
add_te(d48,[d48,d49]); add_lags(d48,[d48,d49]); d48['sp_knn']=spatial_feat(d48,d48); d49['sp_knn']=spatial_feat(d48,d49)
pa=fit_predict(d48[FEAT],d48[TARGET],d49[FEAT])
print("Proxy A (cross-day night) R2 =",round(r2_score(d49[TARGET],pa),4))

# ---- Proxy B: daytime fit (hold out half of day48 daytime) ----
day=d48[d48.slot>=9].copy()
rng=np.random.RandomState(0); msk=rng.rand(len(day))<0.5
trn=pd.concat([d48[d48.slot<9],day[msk]]); val=day[~msk]
add_te(trn,[trn,val]); add_lags(trn,[trn,val]); trn['sp_knn']=spatial_feat(trn,trn); val['sp_knn']=spatial_feat(trn,val)
pb=fit_predict(trn[FEAT],trn[TARGET],val[FEAT])
print("Proxy B (daytime fit) R2 =",round(r2_score(val[TARGET],pb),4))

# ---- FINAL ----
add_te(train,[train,test]); add_lags(train,[train,test]); train['sp_knn']=spatial_feat(train,train); test['sp_knn']=spatial_feat(train,test)
y=train[TARGET].values; tp=np.zeros(len(test)); oof=np.zeros(len(train))
for tri,vai in KFold(5,shuffle=True,random_state=SEED).split(train):
    vp,tpf=fit_predict(train[FEAT].iloc[tri],y[tri],train[FEAT].iloc[vai],test[FEAT])
    oof[vai]=vp; tp+=tpf/5
print("OOF R2 =",round(r2_score(y,oof),4))
pd.DataFrame({'Index':test.Index.values,'demand':np.clip(tp,0,1)}).to_csv("submission_ens.csv",index=False)
print("wrote submission_ens.csv")
