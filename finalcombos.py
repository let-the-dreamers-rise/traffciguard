"""Last-ditch combination tricks from saved predictions (instant).
Champion = 3w_36_32_32 (91.27). Try rank-average, geometric mean, and ultra-fine weights."""
import numpy as np, pandas as pd
from scipy.stats import rankdata
idx=pd.read_csv("dataset/test.csv")['Index'].values
def L(f): return pd.read_csv(f).set_index('Index')['demand'].reindex(idx).values
best=L("submission_best.csv"); e=L("submission_day49e.csv"); r=L("submission_resid.csv"); cat=L("submission_cat.csv")
def save(p,n): p=np.clip(p,0,1); pd.DataFrame({'Index':idx,'demand':p}).to_csv(n,index=False); print("wrote",n,"mean=",round(p.mean(),4))

# ultra-fine weights around champion 36/32/32
for wb,we,wr in [(0.37,0.32,0.31),(0.35,0.33,0.32),(0.36,0.34,0.30),(0.38,0.31,0.31),(0.36,0.31,0.33)]:
    save(wb*best+we*e+wr*r, f"submission_fz_{int(wb*100)}_{int(we*100)}_{int(wr*100)}.csv")

# geometric mean (champion weights as exponents) - reduces influence of large outliers
eps=1e-6
gm=np.exp(0.36*np.log(best+eps)+0.32*np.log(e+eps)+0.32*np.log(r+eps))
save(gm,"submission_geo.csv")

# rank-average then map back to champion's value distribution (quantile transform)
champ=0.36*best+0.32*e+0.32*r
ra=(rankdata(best)+rankdata(e)+rankdata(r))/3.0
order=np.argsort(np.argsort(ra)); sorted_champ=np.sort(champ)
rankblend=sorted_champ[order]   # rank-average shape, champion's value distribution
save(rankblend,"submission_rank.csv")

# blend champion with rank version (hedge)
save(0.7*champ+0.3*rankblend,"submission_rankmix.csv")
