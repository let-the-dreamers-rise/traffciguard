"""Enhanced model built on the proven 87 recipe (low-smoothing TE + temporal neighbors),
plus: near-raw exact lookup, SVD latent factors, spatial KNN, 3-seed ensemble.
Writes submission_best.csv. Keeps submission.csv (87) as floor."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb

SEED=42
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
    df['mins_of_day']=p[0]*60+p[1]; df['slot']=df['mins_of_day']//15; df['hour']=p[0]
    df['t_sin']=np.sin(2*np.pi*df['mins_of_day']/1440); df['t_cos']=np.cos(2*np.pi*df['mins_of_day']/1440)
    df['gh5']=df.geohash.str[:5]; df['gh4']=df.geohash.str[:4]; df['gh3']=df.geohash.str[:3]
    df['LargeVehicles']=(df.LargeVehicles=='Allowed').astype(int); df['Landmarks']=(df.Landmarks=='Yes').astype(int)
    return df
train=prep(pd.read_csv("dataset/train.csv")); test=prep(pd.read_csv("dataset/test.csv"))
TARGET='demand'; y=train[TARGET].values; GM=y.mean()
for col in ['RoadType','Weather']:
    cats=pd.Categorical(pd.concat([train[col],test[col]]))
    train[col]=pd.Categorical(train[col],categories=cats.categories).codes
    test[col]=pd.Categorical(test[col],categories=cats.categories).codes
full=pd.concat([train,test],ignore_index=True)
for lvl in [['geohash','slot'],['geohash'],['slot']]:
    full['Temperature']=full['Temperature'].fillna(full.groupby(lvl)['Temperature'].transform('median'))
full['Temperature']=full['Temperature'].fillna(full['Temperature'].median())
train['Temperature']=full['Temperature'].iloc[:len(train)].values
test['Temperature']=full['Temperature'].iloc[len(train):].values

def smooth_map(df,keys,m):
    g=df.groupby(keys)[TARGET].agg(['mean','count']); return (g['mean']*g['count']+GM*m)/(g['count']+m)
TE_KEYS={'te_gh_ts':(['geohash','timestamp'],1.0),'te_gh_ts_raw':(['geohash','timestamp'],0.1),
    'te_gh5_ts':(['gh5','timestamp'],2.0),'te_gh4_ts':(['gh4','timestamp'],3.0),
    'te_gh3_ts':(['gh3','timestamp'],4.0),'te_gh':(['geohash'],3.0),'te_ts':(['timestamp'],5.0),
    'te_gh5':(['gh5'],3.0),'te_road_ts':(['RoadType','slot'],5.0),'te_gh_hour':(['geohash','hour'],1.0)}
def neighbor_feats(stats,look,m):
    g=stats.groupby(['geohash','slot'])[TARGET].agg(['mean','count']); enc=(g['mean']*g['count']+GM*m)/(g['count']+m)
    out={}
    for off,nm in [(-1,'prev'),(1,'next'),(-2,'prev2'),(2,'next2')]:
        out[f'te_ghslot_{nm}']=pd.Index(list(zip(look.geohash,(look.slot+off)%96))).map(enc).astype(float)
    cur=pd.Index(list(zip(look.geohash,look.slot))).map(enc).astype(float)
    st=np.vstack([out['te_ghslot_prev'].values,cur.values,out['te_ghslot_next'].values])
    with np.errstate(invalid='ignore'): out['te_ghslot_roll3']=np.where(np.isnan(st).all(0),np.nan,np.nanmean(st,0))
    return out
NB_COLS=['te_ghslot_prev','te_ghslot_next','te_ghslot_prev2','te_ghslot_next2','te_ghslot_roll3']

# spatial KNN (built full-train; denoised)
all_gh=sorted(set(train.geohash)|set(test.geohash)); gi={g:i for i,g in enumerate(all_gh)}
coords=np.array([gdec(g) for g in all_gh]); all_ts=sorted(set(train.timestamp)|set(test.timestamp)); ti={t:i for i,t in enumerate(all_ts)}
_,nbr=NearestNeighbors(n_neighbors=9).fit(coords).kneighbors(coords); nbr=nbr[:,1:]
def spatial(stats,look):
    mat=np.full((len(all_gh),len(all_ts)),np.nan)
    for (g,t),v in stats.groupby(['geohash','timestamp'])[TARGET].mean().items(): mat[gi[g],ti[t]]=v
    sp=np.nanmean(mat[nbr],axis=1); o=sp[look.geohash.map(gi).values,look.timestamp.map(ti).values]
    return np.where(np.isnan(o),GM,o)
# SVD latent (built full-train)
def svd_feats(stats,frames,rank=12):
    ghs=sorted(stats.geohash.unique()); g2={g:i for i,g in enumerate(ghs)}
    piv=stats.pivot_table(index='geohash',columns='slot',values=TARGET,aggfunc='mean').reindex(index=ghs)
    piv=piv.fillna(piv.mean(axis=0)).fillna(GM); M=piv.values; rm=M.mean(1,keepdims=True)
    svd=TruncatedSVD(n_components=rank,random_state=0); U=svd.fit_transform(M-rm); recon=(U@svd.components_)+rm
    slots=list(piv.columns); s2={s:i for i,s in enumerate(slots)}; Uf=pd.DataFrame(U[:,:6],index=ghs)
    for f in frames:
        gx=f.geohash.map(g2); ok=gx.notna(); gg=gx.fillna(0).astype(int).values; ss=f.slot.map(s2).fillna(0).astype(int).values
        rv=recon[gg,ss]; rv[~ok.values]=GM; f['svd_recon']=rv
        uf=Uf.reindex(f.geohash).values
        for k in range(6): f[f'ghfac{k}']=np.where(ok.values,uf[:,k],0.0)
SVD_COLS=['svd_recon']+[f'ghfac{k}' for k in range(6)]

kf=KFold(5,shuffle=True,random_state=SEED)
for n in list(TE_KEYS)+NB_COLS: train[n]=np.nan
for tri,vai in kf.split(train):
    ft=train.iloc[tri]
    for n,(keys,m) in TE_KEYS.items():
        train.iloc[vai,train.columns.get_loc(n)]=np.asarray(train.iloc[vai].set_index(keys).index.map(smooth_map(ft,keys,m)),dtype=float)
    for nm,v in neighbor_feats(ft,train.iloc[vai],2.0).items():
        train.iloc[vai,train.columns.get_loc(nm)]=v.values if hasattr(v,'values') else v
for n in list(TE_KEYS)+NB_COLS: train[n]=train[n].fillna(GM)
for n,(keys,m) in TE_KEYS.items():
    test[n]=np.asarray(test.set_index(keys).index.map(smooth_map(train,keys,m)),dtype=float); test[n]=test[n].fillna(GM)
for nm,v in neighbor_feats(train,test,2.0).items(): test[nm]=(v.values if hasattr(v,'values') else v); test[nm]=test[nm].fillna(GM)
train['sp_knn']=spatial(train,train); test['sp_knn']=spatial(train,test)
svd_feats(train,[train]); svd_feats(train,[test])

FEAT=['lat','lon','mins_of_day','slot','t_sin','t_cos','hour','RoadType','NumberofLanes',
      'LargeVehicles','Landmarks','Temperature','Weather']+list(TE_KEYS)+NB_COLS+['sp_knn']+SVD_COLS
def params(seed): return dict(objective='regression',metric='l2',n_estimators=4000,learning_rate=0.02,
    num_leaves=127,min_child_samples=30,subsample=0.8,subsample_freq=1,colsample_bytree=0.7,
    reg_alpha=0.2,reg_lambda=0.2,random_state=seed,n_jobs=-1,verbose=-1)
oof=np.zeros(len(train)); tp=np.zeros(len(test)); seeds=[42,7,2024]
for s in seeds:
    for tri,vai in KFold(5,shuffle=True,random_state=s).split(train):
        m=lgb.LGBMRegressor(**params(s))
        m.fit(train[FEAT].iloc[tri],y[tri],eval_set=[(train[FEAT].iloc[vai],y[vai])],callbacks=[lgb.early_stopping(80,verbose=False)])
        oof[vai]+=m.predict(train[FEAT].iloc[vai])/len(seeds); tp+=m.predict(test[FEAT])/(5*len(seeds))
print("ensemble OOF R2 =",round(r2_score(y,oof),4))
pd.DataFrame({'Index':test.Index.values,'demand':np.clip(tp,0,1)}).to_csv("submission_best.csv",index=False)
print("wrote submission_best.csv")
