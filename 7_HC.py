import os, numpy as np, pandas as pd
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster

BASE_DIR = r"resid_fullperiod"
CORR_FILE = r"resid_ff3_fullperiod.parquet"
CORR_PATH = os.path.join(BASE_DIR, CORR_FILE)

LINK = "ward"   # single/complete/average/ward
K = 8             # 군집 개수

resid = pd.read_parquet(CORR_PATH).dropna(axis=1)
C = np.corrcoef(resid.to_numpy(float))
C = np.clip(C, -1, 1); np.fill_diagonal(C, 1)

D = np.sqrt(2.0 * (1.0 - C))
Z = linkage(squareform(D, checks=False), method=LINK)
lab = fcluster(Z, t=K, criterion="maxclust") - 1

cluster = pd.Series(lab, index=resid.index, name="cluster")
print("N:", len(cluster), "communities:", cluster.nunique())
print(cluster.value_counts().head(10))

out_csv = os.path.join(BASE_DIR, f"hier_membership_K{K}.csv")
cluster.to_csv(out_csv, encoding="utf-8-sig")
print("saved:", out_csv)


for cid in sorted(cluster.unique()):
    members = cluster.index[cluster.values == cid].tolist()
    print(f"\n[cluster {cid}] size={len(members)}")
    print(members)

