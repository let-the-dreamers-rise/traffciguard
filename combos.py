"""Fine weight search + 4-way blends from already-built predictions (instant, no retrain).
Champion so far: 3way_34_33_33 = 91.24. Explore around it and add day49f as a 4th model."""
import numpy as np, pandas as pd
idx=pd.read_csv("dataset/test.csv")['Index'].values
def L(f): 
    s=pd.read_csv(f).set_index('Index')['demand']; return pd.Series(s).reindex(idx).values
best=L("submission_best.csv"); e=L("submission_day49e.csv"); r=L("submission_resid.csv")
f=L("submission_day49f.csv"); b=L("submission_day49b.csv")

def save(pred,name):
    pred=np.clip(pred,0,1); pd.DataFrame({'Index':idx,'demand':pred}).to_csv(name,index=False)
    print("wrote",name,"mean=",round(pred.mean(),4))

# fine 3-way search around 34/33/33 (best,e,r)
for wb,we,wr in [(0.40,0.30,0.30),(0.36,0.32,0.32),(0.34,0.36,0.30),(0.34,0.30,0.36),(0.30,0.36,0.34),(0.38,0.31,0.31)]:
    save(wb*best+we*e+wr*r, f"submission_3w_{int(wb*100)}_{int(we*100)}_{int(wr*100)}.csv")
# 4-way (best,e,r,f)
for wb,we,wr,wf in [(0.25,0.25,0.25,0.25),(0.30,0.24,0.24,0.22),(0.28,0.26,0.26,0.20),(0.34,0.22,0.22,0.22)]:
    save(wb*best+we*e+wr*r+wf*f, f"submission_4w_{int(wb*100)}_{int(we*100)}_{int(wr*100)}_{int(wf*100)}.csv")
# 5-way incl day49b
save(0.24*best+0.20*e+0.20*r+0.18*f+0.18*b, "submission_5w_eq.csv")
