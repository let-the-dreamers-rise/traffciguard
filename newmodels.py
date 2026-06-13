"""New DIVERSE base models for the blend: CatBoost, XGBoost, log-target LightGBM,
trained on day-49 with the transferable day-48 reference features. Different algorithms
-> uncorrelated errors -> push the blend toward 92. Then combine with best+day49e+resid.
No test labels used."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

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
R_ghts=d48.groupby(['geohash','timestamp'])['demand'].mean(); R_gh5ts=d48.groupby(['gh5','timestamp'])['demand'].mean()
R_gh4ts=d48.groupby(['gh4','timestamp'])['demand'].mean(); R_gh=d48.groupby('geohash')['demand'].mean()
R_ts=d48.groupby('timestamp')['demand'].mean(); R_ghslot=d48.groupby(['geohash','slot'])['demand'].mean()
R_ghhour=d48.groupby(['geohash','hour'])['demand'].mean()
R_road=d48.groupby(['RoadType','slot'])['demand'].mean(); R_lanes=d48.groupby(['NumberofLanes','slot'])['demand'].mean()
R_weather=d48.groupby(['Weather','slot'])['demand'].mean(); R_roadgh=d48.groupby(['RoadType','geohash'])['demand'].mean()
all_gh=sorted(set(train.geohash)|set(test.geohash)); gi={g:i for i,g in enumerate(all_gh)}
coords=np.array([gdec(g) for g in all_gh]); all_ts=sorted(set(train.timestamp)|set(test.timestamp)); ti={t:i for i,t in enumerate(all_ts)}
_,nbr=NearestNeighbors(n_neighbors=9).fit(coords).kneighbors(coords); nbr=nbr[:,1:]
mat=np.full((len(all_gh),len(all_ts)),np.nan)
for (g,t),v in d48.groupby(['geohash','timestamp'])['demand'].mean().items(): mat[gi[g],ti[t]]=v
SP=np.nanmean(mat[nbr],axis=1)
ghs=sorted(d48.geohash.unique()); g2={g:i for i,g in enumerate(ghs)}
piv=d48.pivot_table(index='geohash',columns='slot',values='demand',aggfunc='mean').reindex(index=ghs)
piv=piv.fillna(piv.mean(0)).fillna(GM); Mv=piv.values; rm=Mv.mean(1,keepdims=True)
svd=TruncatedSVD(n_components=12,random_state=0); U=svd.fit_transform(Mv-rm); recon=(U@svd.components_)+rm
s2={s:i for i,s in enumerate(list(piv.columns))}; Uf=pd.DataFrame(U[:,:6],index=ghs)
def add_feats(df):
    df['r_ghts']=np.asarray(df.set_index(['geohash','timestamp']).index.map(R_ghts),dtype=float)
    df['r_gh5ts']=np.asarray(df.set_index(['gh5','timestamp']).index.map(R_gh5ts),dtype=float)
    df['r_gh4ts']=np.asarray(df.set_index(['gh4','timestamp']).index.map(R_gh4ts),dtype=float)
    df['r_gh']=df.geohash.map(R_gh); df['r_ts']=df.timestamp.map(R_ts)
    df['r_ghhour']=np.asarray(df.set_index(['geohash','hour']).index.map(R_ghhour),dtype=float)
    df['r_road']=np.asarray(df.set_index(['RoadType','slot']).index.map(R_road),dtype=float)
    df['r_lanes']=np.asarray(df.set_index(['NumberofLanes','slot']).index.map(R_lanes),dtype=float)
    df['r_weather']=np.asarray(df.set_index(['Weather','slot']).index.map(R_weather),dtype=float)
    df['r_roadgh']=np.asarray(df.set_index(['RoadType','geohash']).index.map(R_roadgh),dtype=float)
    for off,nm in [(-1,'rp1'),(1,'rn1'),(-2,'rp2'),(2,'rn2'),(-4,'rp4'),(4,'rn4')]:
        df[nm]=np.asarray(pd.Index(list(zip(df.geohash,(df.slot+off)%96))).map(R_ghslot),dtype=float)
    sp=SP[df.geohash.map(gi).values,df.timestamp.map(ti).values]; df['sp_knn']=np.where(np.isnan(sp),GM,sp)
    gx=df.geohash.map(g2); ok=gx.notna(); gg=gx.fillna(0).astype(int).values; ss=df.slot.map(s2).fillna(0).astype(int).values
    rv=recon[gg,ss]; rv[~ok.values]=GM; df['svd_recon']=rv
    uf=Uf.reindex(df.geohash).values
    for k in range(6): df[f'ghfac{k}']=np.where(ok.values,uf[:,k],0.0)
    for c in ['r_ghts','r_gh5ts','r_gh4ts','r_gh','r_ts','r_ghhour','r_road','r_lanes','r_weather','r_roadgh','rp1','rn1','rp2','rn2','rp4','rn4']:
        df[c]=df[c].fillna(df['r_gh5ts']).fillna(df['r_gh4ts']).fillna(df['r_ts']).fillna(GM)
for df in (d49,test): add_feats(df)
FEAT=['lat','lon','mins','slot','hour','t_sin','t_cos','RoadType','NumberofLanes','LV','LM','Temperature','Weather',
      'r_ghts','r_gh5ts','r_gh4ts','r_gh','r_ts','r_ghhour','r_road','r_lanes','r_weather','r_roadgh',
      'rp1','rn1','rp2','rn2','rp4','rn4','sp_knn','svd_recon']+[f'ghfac{k}' for k in range(6)]
X=d49[FEAT].values; yv=d49['demand'].values; Xt=test[FEAT].values

# CatBoost
pc=np.zeros(len(test))
for s in [42,7,2024]:
    c=CatBoostRegressor(iterations=2000,learning_rate=0.03,depth=8,l2_leaf_reg=5.0,
        loss_function='RMSE',random_seed=s,verbose=0); c.fit(X,yv); pc+=c.predict(Xt)/3
pc=np.clip(pc,0,1); pd.DataFrame({'Index':test.Index.values,'demand':pc}).to_csv("submission_cat.csv",index=False)
print("wrote submission_cat.csv mean=",round(pc.mean(),4))
# XGBoost
px=np.zeros(len(test))
for s in [42,7,2024]:
    xg=XGBRegressor(n_estimators=1500,learning_rate=0.03,max_depth=7,subsample=0.8,colsample_bytree=0.7,
        reg_alpha=0.3,reg_lambda=0.5,random_state=s,n_jobs=-1,tree_method='hist'); xg.fit(X,yv); px+=xg.predict(Xt)/3
px=np.clip(px,0,1); pd.DataFrame({'Index':test.Index.values,'demand':px}).to_csv("submission_xgb.csv",index=False)
print("wrote submission_xgb.csv mean=",round(px.mean(),4))
# log-target LGB
pl=np.zeros(len(test))
for s in [42,7,2024]:
    m=lgb.LGBMRegressor(objective='regression',n_estimators=1800,learning_rate=0.02,num_leaves=63,
        subsample=0.8,subsample_freq=1,colsample_bytree=0.8,reg_alpha=0.3,reg_lambda=0.3,
        min_child_samples=20,random_state=s,n_jobs=-1,verbose=-1); m.fit(X,np.log1p(yv)); pl+=np.expm1(m.predict(Xt))/3
pl=np.clip(pl,0,1); pd.DataFrame({'Index':test.Index.values,'demand':pl}).to_csv("submission_logt.csv",index=False)
print("wrote submission_logt.csv mean=",round(pl.mean(),4))

# blends with champions
idx=test.Index.values
best=pd.read_csv("submission_best.csv").set_index('Index')['demand'].reindex(idx).values
e=pd.read_csv("submission_day49e.csv").set_index('Index')['demand'].reindex(idx).values
r=pd.read_csv("submission_resid.csv").set_index('Index')['demand'].reindex(idx).values
def save(p,n): p=np.clip(p,0,1); pd.DataFrame({'Index':idx,'demand':p}).to_csv(n,index=False); print("wrote",n,"mean=",round(p.mean(),4))
# add catboost as 4th to the 36/32/32 champion frame
save(0.30*best+0.26*e+0.24*r+0.20*pc, "submission_4wc_30_26_24_20.csv")
save(0.30*best+0.24*e+0.22*r+0.24*pc, "submission_4wc_30_24_22_24.csv")
# 5-way best+e+r+cat+xgb
save(0.28*best+0.20*e+0.18*r+0.18*pc+0.16*px, "submission_5wc.csv")
# 6-way add logt
save(0.26*best+0.18*e+0.16*r+0.16*pc+0.14*px+0.10*pl, "submission_6wc.csv")
