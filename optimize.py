"""Data-driven blend optimization using REAL day-49 night labels.
- Anchor model (day-48-based, OOF target encoding) -> OOF preds incl day-49 night + test preds
- day-49 engine -> OOF preds on day-49 night + test preds
Solve for blend weight maximizing R2 on actual day-49 night demand, then apply to test.
No test labels used. Writes submission_opt.csv (+ neighbors)."""
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
    df['mins']=p[0]*60+p[1]; df['slot']=df['mins']//15; df['hour']=p[0]
    df['t_sin']=np.sin(2*np.pi*df['mins']/1440); df['t_cos']=np.cos(2*np.pi*df['mins']/1440)
    df['gh5']=df.geohash.str[:5]; df['gh4']=df.geohash.str[:4]; df['gh3']=df.geohash.str[:3]
    df['LV']=(df.LargeVehicles=='Allowed').astype(int); df['LM']=(df.Landmarks=='Yes').astype(int)
    return df
train=prep(pd.read_csv("dataset/train.csv")); test=prep(pd.read_csv("dataset/test.csv"))
y=train['demand'].values; GM=y.mean()
for c in ['RoadType','Weather']:
    cats=pd.Categorical(pd.concat([train[c],test[c]]))
    train[c]=pd.Categorical(train[c],categories=cats.categories).codes
    test[c]=pd.Categorical(test[c],categories=cats.categories).codes
for df in (train,test):
    df['Temperature']=df.groupby('geohash')['Temperature'].transform(lambda s:s.fillna(s.median()))
    df['Temperature']=df['Temperature'].fillna(train['Temperature'].median())
night_mask=(train['day']==49).values

# ---------------- ANCHOR model (OOF target encoding over all train) ----------------
def sm(df,keys,m):
    g=df.groupby(keys)['demand'].agg(['mean','count']); return (g['mean']*g['count']+GM*m)/(g['count']+m)
AK={'te_gh_ts':(['geohash','timestamp'],1.0),'te_gh5_ts':(['gh5','timestamp'],2.0),
    'te_gh4_ts':(['gh4','timestamp'],3.0),'te_gh':(['geohash'],3.0),'te_ts':(['timestamp'],5.0),
    'te_gh5':(['gh5'],3.0),'te_gh_hour':(['geohash','hour'],1.0),'te_road_ts':(['RoadType','slot'],5.0)}
def nb(stats,look,m):
    g=stats.groupby(['geohash','slot'])['demand'].agg(['mean','count']); enc=(g['mean']*g['count']+GM*m)/(g['count']+m)
    out={}
    for off,nm in [(-1,'prev'),(1,'next'),(-2,'prev2'),(2,'next2')]:
        out['nb_'+nm]=pd.Index(list(zip(look.geohash,(look.slot+off)%96))).map(enc).astype(float)
    return out
NBC=['nb_prev','nb_next','nb_prev2','nb_next2']
kf=KFold(5,shuffle=True,random_state=SEED)
for n in list(AK)+NBC: train[n]=np.nan
for tri,vai in kf.split(train):
    ft=train.iloc[tri]
    for n,(keys,m) in AK.items():
        train.iloc[vai,train.columns.get_loc(n)]=np.asarray(train.iloc[vai].set_index(keys).index.map(sm(ft,keys,m)),dtype=float)
    for nm,v in nb(ft,train.iloc[vai],2.0).items(): train.iloc[vai,train.columns.get_loc(nm)]=v.values
for n in list(AK)+NBC: train[n]=train[n].fillna(GM)
for n,(keys,m) in AK.items(): test[n]=np.asarray(test.set_index(keys).index.map(sm(train,keys,m)),dtype=float); test[n]=test[n].fillna(GM)
for nm,v in nb(train,test,2.0).items(): test[nm]=v.values; test[nm]=test[nm].fillna(GM)
AF=['lat','lon','mins','slot','t_sin','t_cos','hour','RoadType','NumberofLanes','LV','LM','Temperature','Weather']+list(AK)+NBC
ap=dict(objective='regression',n_estimators=3000,learning_rate=0.02,num_leaves=127,min_child_samples=30,
        subsample=0.8,subsample_freq=1,colsample_bytree=0.7,reg_alpha=0.2,reg_lambda=0.2,random_state=SEED,n_jobs=-1,verbose=-1)
anchor_oof=np.zeros(len(train)); anchor_test=np.zeros(len(test))
for tri,vai in kf.split(train):
    m=lgb.LGBMRegressor(**ap); m.fit(train[AF].iloc[tri],y[tri],eval_set=[(train[AF].iloc[vai],y[vai])],callbacks=[lgb.early_stopping(80,verbose=False)])
    anchor_oof[vai]=m.predict(train[AF].iloc[vai]); anchor_test+=m.predict(test[AF])/5
print("anchor OOF R2 (all):",round(r2_score(y,anchor_oof),4),"| on day49 night:",round(r2_score(y[night_mask],anchor_oof[night_mask]),4))

# ---------------- day-49 ENGINE (clean day-48 refs) OOF on day-49 night ----------------
d48=train[train['day']==48].copy(); d49=train[train['day']==49].copy()
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
EF=['lat','lon','mins','slot','hour','t_sin','t_cos','RoadType','NumberofLanes','LV','LM','Temperature','Weather',
    'r_ghts','r_gh5ts','r_gh4ts','r_gh','r_ts','r_ghhour','r_road','r_lanes','r_weather','r_roadgh',
    'rp1','rn1','rp2','rn2','rp4','rn4','sp_knn','svd_recon']+[f'ghfac{k}' for k in range(6)]
d49=d49.reset_index(drop=True); yd=d49['demand'].values
eng_oof=np.zeros(len(d49)); eng_test=np.zeros(len(test))
ep=dict(objective='regression',n_estimators=1800,learning_rate=0.02,num_leaves=63,subsample=0.8,subsample_freq=1,
        colsample_bytree=0.8,reg_alpha=0.3,reg_lambda=0.3,min_child_samples=20,n_jobs=-1,verbose=-1)
for tri,vai in KFold(5,shuffle=True,random_state=SEED).split(d49):
    for s in [42,7,2024]:
        p=dict(ep); p['random_state']=s; m=lgb.LGBMRegressor(**p); m.fit(d49[EF].iloc[tri],yd[tri])
        eng_oof[vai]+=m.predict(d49[EF].iloc[vai])/3; eng_test+=m.predict(test[EF])/(5*3)
print("engine OOF R2 on day49 night:",round(r2_score(yd,eng_oof),4))

# align anchor night preds to d49 order
anc_night=anchor_oof[night_mask]
# ---- optimize weight on day-49 night ----
best=(-1,0)
for w in np.linspace(0,1,101):
    r=r2_score(yd, w*eng_oof+(1-w)*anc_night)
    if r>best[0]: best=(r,w)
print(f"OPTIMAL day-49 weight w={best[1]:.2f}  night-R2={best[0]:.4f}  (anchor-only={r2_score(yd,anc_night):.4f}, engine-only={r2_score(yd,eng_oof):.4f})")
w=best[1]
for ww in sorted(set([round(w,2),round(max(0,w-0.08),2),round(min(1,w+0.08),2)])):
    pred=np.clip(ww*eng_test+(1-ww)*anchor_test,0,1)
    pd.DataFrame({'Index':test.Index.values,'demand':pred}).to_csv(f"submission_opt_{int(ww*100)}.csv",index=False)
    print(f"wrote submission_opt_{int(ww*100)}.csv mean={pred.mean():.4f}")
