import os, numpy as np, pandas as pd
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_samples

BASE_DIR = r"resid_fullperiod"
CORR_FILE = r"resid_ff3_fullperiod.parquet"
CORR_PATH = os.path.join(BASE_DIR, CORR_FILE)

LINK = "ward"
K = 25
MIN_T = 0

# ===== 1) load =====
resid = pd.read_parquet(CORR_PATH).astype(float)
resid = resid.dropna(axis=1, how="all").dropna(axis=0, how="all")

# ===== 2) pairwise corr -> distance =====
C = resid.T.corr(method="pearson", min_periods=MIN_T).to_numpy()
C = np.nan_to_num(C, nan=0.0)
C = np.clip(C, -1.0, 1.0)
np.fill_diagonal(C, 1.0)

D = np.sqrt(np.maximum(0.0, 2.0 * (1.0 - C)))
np.fill_diagonal(D, 0.0)

# ===== 3) clustering =====
Z = linkage(squareform(D, checks=False), method=LINK)
lab = fcluster(Z, t=K, criterion="maxclust") - 1  # (N,)
cluster = pd.Series(lab, index=resid.index, name="cluster")

# ===== 4) silhouette + cluster sorting =====
sil = silhouette_samples(D, lab, metric="precomputed")  # D가 거리행렬이므로 precomputed
tmp = pd.DataFrame({"cluster": lab, "sil": sil}, index=resid.index)

stat = (tmp.groupby("cluster")
          .agg(size=("sil","size"), sil_mean=("sil","mean"))
          .sort_values(["sil_mean","size"], ascending=[False, False]))

# ===== 5) print members by sil_mean desc =====
for cid in stat.index:
    members = tmp.index[tmp["cluster"].values == cid].tolist()
    print(f"\n[cluster {cid}] size={stat.loc[cid,'size']} sil_mean={stat.loc[cid,'sil_mean']:.4f}")
    print(members)
