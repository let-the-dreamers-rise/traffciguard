"""Heavier feature engineering on the GENERALIZABLE signal.
Adds: vectorized spatial-KNN demand (nearby geohashes, same slot),
road/lanes/weather/landmark x time target encodings, robust smoothing.
Reports BOTH OOF and the day48->day49 generalization proxy."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.neighbors import NearestNeighbors
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
def prep(df):
    cache={g:gdec(g) for g in df.geohash.unique()}
    df['lat']=df.geohash.map(lambda g:cache[g][0]); df['lon']=df.geohash.map(lambda g:cache[g][1])
    p=df.timestamp.str.split(':',expand=True).astype(int)
    df['mins']=p[0]*60+p[1]; df['slot']=df['m