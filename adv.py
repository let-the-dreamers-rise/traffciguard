"""Advanced FE test: low-rank SVD factorization of the geohash x slot demand matrix.
Extracts latent location/time factors (denoised, generalizable) instead of memorizing.
Judged on the cross-day proxy (train day48 -> predict day49 night), the honest metric."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.metrics import r2_score
from sklearn.decomposition import TruncatedSVD
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
    df['gh5']=df.geohash.str[:5]; df['LV']=(df.LargeVehicles=='Allowed').astype(int); df['LM']=(df.Landmarks=='Yes').astype(int)
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

RANK=12
def svd_feats(stats, frames, rank=RANK):
    ghs=sorted(stats.geohash.unique()); gi={g:i for i,g in enumerate(ghs)}
    piv=stats.pivot_table(index='geohash',columns='slot',values='demand',aggfunc='mean')
    piv=piv.reindex(index=ghs)
    colmean=piv.mean(axis=0); piv=piv.fillna(colmean).fillna(GM)
    M=piv.values
    rowmean=M.mean(axis=1,keepdims=True); Mc=M-rowmean
    svd=TruncatedSVD(n_components=rank,random_state=0); U=svd.fit_transform(Mc); Vt=svd.components_
    recon=(U@Vt)+rowmean
    slots=list(piv.columns); si={s:i for i,s in enumerate(slots)}
    Ufac=pd.DataFrame(U[:, :6], index=ghs)  # top-6 geohash latent factors
    for f in frames:
        gidx=f.geohash.map(gi); valid=gidx.notna()
        gg=gidx.fillna(0).astype(int).values; ss=f.slot.map(si).fillna(0).astype(int).values
        rv=recon[gg,ss]; rv[~valid.values]=GM
        f['svd_recon']=rv
        uf=Ufac.reindex(f.geohash).values
        for k in range(6): f[f'ghfac{k}']=np.where(valid.values,uf[:,k],0.0)
    return [f'svd_recon']+[f'ghfac{k}' for k in range(6)]

def sm(df,keys,m):
    g=df.groupby(keys)['demand'].agg(['mean','count']); return (g['mean']*g['count']+GM*m)/(g['count']+m)
KEYS={'te_gh':(['geohash'],5.),'te_ts':(['timestamp'],5.),'te_gh_ts':(['geohash','timestamp'],4.),
      'te_gh_hour':(['geohash','hour'],3.),'te_road_slot':(['RoadType','slot'],8.)}
def add_te(stats,frames):
    for n,(keys,m) in KEYS.items():
        enc=sm(stats,keys,m)
        for f in frames: f[n]=np.asarray(f.set_index(keys).index.map(enc),dtype=float); f[n]=f[n].fillna(GM)
def add_lags(stats,frames):
    g=stats.groupby(['geohash','slot'])['demand'].mean()
    for off,nm in [(-1,'p1'),(1,'n1')]:
        for f in frames:
            k=list(zip(f.geohash,(f.slot+off)%96)); f['lag_'+nm]=np.asarray(pd.Index(k).map(g),dtype=float); f['lag_'+nm]=f['lag_'+nm].fillna(GM)

BASE=['lat','lon','mins','slot','hour','t_sin','t_cos','RoadType','NumberofLanes','LV','LM','Temperature','Weather']
params=dict(objective='regression',n_estimators=900,learning_rate=0.03,num_leaves=95,subsample=0.8,
            subsample_freq=1,colsample_bytree=0.7,reg_alpha=0.3,reg_lambda=0.3,min_child_samples=40,
            random_state=42,n_jobs=-1,verbose=-1)

d48=train[train.day==48].copy(); d49=train[train.day==49].copy()
add_te(d48,[d48,d49]); add_lags(d48,[d48,d49])
svd_cols=svd_feats(d48,[d48,d49])

FEAT_NO=BASE+list(KEYS)+['lag_p1','lag_n1']
FEAT_SVD=FEAT_NO+svd_cols
for tag,FEAT in [('NO svd',FEAT_NO),('WITH svd',FEAT_SVD)]:
    m=lgb.LGBMRegressor(**params); m.fit(d48[FEAT],d48['demand'])
    print(f"Proxy A [{tag}] R2 =",round(r2_score(d49['demand'],m.predict(d49[FEAT])),4))
