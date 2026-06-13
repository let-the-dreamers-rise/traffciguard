"""day-49 model v3: adds the per-geohash day-49 NIGHT level (leave-one-out for train,
full for test) - day-49-specific info beyond the day-48 reference. Plus tighter blend
weights and a 3-way blend (day49c + day49b + best). No test labels used."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
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

# ---- day-49 NIGHT level per geohash (LOO for train, full for test) ----
g_sum=d49.groupby('geohash')['demand'].transform('sum'); g_cnt=d49.groupby('geohash')['demand'].transform('count')
d49['lvl49_gh']=np.where(g_cnt>1,(g_sum-d49['demand'])/(g_cnt-1),GM)
full_lvl49=d49.groupby('geohash')['demand'].mean()
g5_sum=d49.groupby('gh5')['demand'].transform('sum'); g5_cnt=d49.groupby('gh5')['demand'].transform('count')
d49['lvl49_gh5']=np.where(g5_cnt>1,(g5_sum-d49['demand'])/(g5_cnt-1),GM)
full_lvl49_5=d49.groupby('gh5')['demand'].mean()
test['lvl49_gh']=test.geohash.map(full_lvl49).fillna(test.gh5.map(full_lvl49_5)).fillna(GM)
test['lvl49_gh5']=test.gh5.map(full_lvl49_5).fillna(GM)

def add_feats(df):
    df['r_ghts']=np.asarray(df.set_index(['geohash','timestamp']).index.map(R_ghts),dtype=float)
    df['r_gh5ts']=np.asarray(df.set_index(['gh5','timestamp']).index.map(R_gh5ts),dtype=float)
    df['r_gh4ts']=np.asarray(df.set_index(['gh4','timestamp']).index.map(R_gh4ts),dtype=float)
    df['r_gh']=df.geohash.map(R_gh); df['r_ts']=df.timestamp.map(R_ts)
    df['r_ghhour']=np.asarray(df.set_index(['geohash','hour']).index.map(R_ghhour),dtype=float)
    for off,nm in [(-1,'rp1'),(1,'rn1'),(-2,'rp2'),(2,'rn2'),(-4,'rp4'),(4,'rn4')]:
        df[nm]=np.asarray(pd.Index(list(zip(df.geohash,(df.slot+off)%96))).map(R_ghslot),dtype=float)
    sp=SP[df.geohash.map(gi).values,df.timestamp.map(ti).values]; df['sp_knn']=np.where(np.isnan(sp),GM,sp)
    gx=df.geohash.map(g2); ok=gx.notna(); gg=gx.fillna(0).astype(int).values; ss=df.slot.map(s2).fillna(0).astype(int).values
    rv=recon[gg,ss]; rv[~ok.values]=GM; df['svd_recon']=rv
    uf=Uf.reindex(df.geohash).values
    for k in range(6): df[f'ghfac{k}']=np.where(ok.values,uf[:,k],0.0)
    for c in ['r_ghts','r_gh5ts','r_gh4ts','r_gh','r_ts','r_ghhour','rp1','rn1','rp2','rn2','rp4','rn4']:
        df[c]=df[c].fillna(df['r_gh5ts']).fillna(df['r_gh4ts']).fillna(df['r_ts']).fillna(GM)
for df in (d49,test): add_feats(df)

FEAT=['lat','lon','mins','slot','hour','t_sin','t_cos','RoadType','NumberofLanes','LV','LM','Temperature','Weather',
      'r_ghts','r_gh5ts','r_gh4ts','r_gh','r_ts','r_ghhour','rp1','rn1','rp2','rn2','rp4','rn4','sp_knn','svd_recon',
      'lvl49_gh','lvl49_gh5']+[f'ghfac{k}' for k in range(6)]
tp=np.zeros(len(test))
for s in [42,7,2024,11,99]:
    m=lgb.LGBMRegressor(objective='regression',n_estimators=1800,learning_rate=0.02,num_leaves=63,
        subsample=0.8,subsample_freq=1,colsample_bytree=0.8,reg_alpha=0.3,reg_lambda=0.3,
        min_child_samples=20,random_state=s,n_jobs=-1,verbose=-1)
    m.fit(d49[FEAT],d49['demand']); tp+=m.predict(test[FEAT])/5
d49c=np.clip(tp,0,1)
pd.DataFrame({'Index':test.Index.values,'demand':d49c}).to_csv("submission_day49c.csv",index=False)
print("wrote submission_day49c.csv mean=",round(d49c.mean(),4))
imp=pd.Series(m.feature_importances_,index=FEAT).sort_values(ascending=False); print(imp.head(8).to_string())

best=pd.read_csv("submission_best.csv").set_index('Index')['demand']; bb=test.Index.map(best).values
d49b=pd.read_csv("submission_day49b.csv").set_index('Index')['demand']; b49b=test.Index.map(d49b).values
for w in [65,70,75]:
    bl=np.clip((w/100)*d49c+(1-w/100)*bb,0,1)
    pd.DataFrame({'Index':test.Index.values,'demand':bl}).to_csv(f"submission_c_blend_{w}.csv",index=False)
    print(f"wrote submission_c_blend_{w}.csv mean=",round(bl.mean(),4))
# 3-way blend
bl3=np.clip(0.5*d49c+0.3*b49b+0.2*bb,0,1)
pd.DataFrame({'Index':test.Index.values,'demand':bl3}).to_csv("submission_blend3.csv",index=False)
print("wrote submission_blend3.csv mean=",round(bl3.mean(),4))
