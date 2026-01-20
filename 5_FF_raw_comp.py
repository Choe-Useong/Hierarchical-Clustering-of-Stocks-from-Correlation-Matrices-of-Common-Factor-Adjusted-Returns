import os
import numpy as np
import pandas as pd

from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import silhouette_score

# ================= 설정 =================
BASE_DIR = r"resid_fullperiod"

PATH_FF3  = os.path.join(BASE_DIR, "resid_ff3_fullperiod.parquet")         # (N,T) FF3 잔차
PATH_RET  = os.path.join(BASE_DIR, "ret_ff3fill_fullperiod.parquet")       # (N,T) 원수익률(결측만 FF3로 채움)

LINK   = "ward"
K_LIST = [5,10,15,20,25,27,30,35,40,50,60]

# ================= 유틸 =================
def panel_to_corr(df: pd.DataFrame) -> np.ndarray:
    X = df.to_numpy(float)
    C = np.corrcoef(X)
    C = np.clip(C, -1.0, 1.0)
    np.fill_diagonal(C, 1.0)
    return C

def corr_to_dist(C: np.ndarray) -> np.ndarray:
    C = np.clip(C, -1.0, 1.0)
    np.fill_diagonal(C, 1.0)
    D = np.sqrt(2.0 * (1.0 - C))
    np.fill_diagonal(D, 0.0)
    return D

def VI(l1: np.ndarray, l2: np.ndarray) -> float:
    u1, inv1 = np.unique(l1, return_inverse=True)
    u2, inv2 = np.unique(l2, return_inverse=True)
    M = np.zeros((len(u1), len(u2)), dtype=np.int64)
    np.add.at(M, (inv1, inv2), 1)

    n = M.sum()
    Pxy = M / n
    Px = Pxy.sum(axis=1, keepdims=True)
    Py = Pxy.sum(axis=0, keepdims=True)
    eps = 1e-12
    Hx = -np.sum(Px * np.log(Px + eps))
    Hy = -np.sum(Py * np.log(Py + eps))
    I  = np.sum(Pxy * (np.log(Pxy + eps) - np.log(Px + eps) - np.log(Py + eps)))
    return float(Hx + Hy - 2.0 * I)

# ================= 로드 =================
ff3  = pd.read_parquet(PATH_FF3).dropna(axis=1)   # (N,T)
ret  = pd.read_parquet(PATH_RET).dropna(axis=1)   # (N,T)

# ================= 공통 유니버스(교집합) =================
common = ff3.index.intersection(ret.index)
ff3 = ff3.loc[common]
ret = ret.loc[common]

common_dates = ff3.columns.intersection(ret.columns)
ff3 = ff3.loc[:, common_dates]
ret = ret.loc[:, common_dates]

# ================= corr -> distance =================
CF = panel_to_corr(ff3)
CR = panel_to_corr(ret)

DF = corr_to_dist(CF)
DR = corr_to_dist(CR)

# ================= linkage (각 1회) =================
ZF = linkage(squareform(DF, checks=False), method=LINK)  # FF3 잔차
ZR = linkage(squareform(DR, checks=False), method=LINK)  # 원수익률

# ================= 평가 =================
rows = []
for K in K_LIST:
    labF = fcluster(ZF, t=K, criterion="maxclust") - 1
    labR = fcluster(ZR, t=K, criterion="maxclust") - 1

    sF = silhouette_score(DF, labF, metric="precomputed") if len(np.unique(labF)) > 1 else np.nan
    sR = silhouette_score(DR, labR, metric="precomputed") if len(np.unique(labR)) > 1 else np.nan

    ari = adjusted_rand_score(labF, labR)
    nmi = normalized_mutual_info_score(labF, labR)
    vi  = VI(labF, labR)

    rows.append({
        "K": K,
        "n": len(common),

        "sil_ff3": sF,
        "sil_ret": sR,
        "d_sil(ff3-ret)": sF - sR,

        "ARI(ff3,ret)": ari,
        "NMI(ff3,ret)": nmi,
        "VI(ff3,ret)":  vi,
    })

res = pd.DataFrame(rows).sort_values("K").reset_index(drop=True)

print(res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\n[delta silhouette]")
print(res[["K","d_sil(ff3-ret)"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
