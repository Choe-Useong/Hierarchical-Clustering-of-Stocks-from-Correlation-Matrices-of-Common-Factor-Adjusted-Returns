import os
import numpy as np
import pandas as pd

from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

# ================= 설정 =================
BASE_DIR   = r"resid_fullperiod"
PATH_FF3   = os.path.join(BASE_DIR, "resid_ff3_fullperiod.parquet")  # (N,T)

BASE_ITEMS = r"items_parquet"
FILES = [
    os.path.join(BASE_ITEMS, "ff_item__FnGuide Sector.parquet"), 
    os.path.join(BASE_ITEMS, "ff_item__FnGuide Industry Group.parquet"),
    os.path.join(BASE_ITEMS, "ff_item__FnGuide Industry.parquet"),
]

LINK = "ward" 

# ================= FF3 로드 =================
ff3 = pd.read_parquet(PATH_FF3).dropna(axis=1)
ff3.index = (ff3.index.get_level_values(0) if isinstance(ff3.index, pd.MultiIndex) else ff3.index).astype(str)

rows = []

for f in FILES:
    name = os.path.basename(f).replace("item__", "").replace(".parquet", "")

    ind_df = pd.read_parquet(f)
    ind_df.index = (ind_df.index.get_level_values(0) if isinstance(ind_df.index, pd.MultiIndex) else ind_df.index).astype(str)

    # 라벨 1개 뽑기: (N,1)이면 그 컬럼, 아니면 마지막 컬럼
    ind = ind_df.iloc[:, 0] if ind_df.shape[1] == 1 else ind_df.iloc[:, -1]
    ind = ind.copy()

    # 공통 티커 + 라벨 결측 제거
    common = ff3.index.intersection(ind.index)
    ind_c = ind.loc[common]
    ind_c = ind_c[~pd.isna(ind_c)]
    common = ind_c.index

    n = len(common)
    K = int(pd.Series(ind_c.values).nunique())  # 업종 고유개수

    if n < 3 or K < 2:
        rows.append({
            "level": name, "n": n, "K": K,
            "sil_ward": np.nan, "sil_industry": np.nan,
            "ARI": np.nan, "NMI": np.nan, "VI": np.nan,
            "d_sil(ward-ind)": np.nan
        })
        continue
    if K >= n:
        K = n - 1

    # 이 레벨에 해당하는 종목만
    X = ff3.loc[common].dropna(axis=1)  # 날짜축에서 NaN 포함 열 제거(골자 유지)
    XA = X.to_numpy(float)

    # corr -> dist
    C = np.corrcoef(XA)
    C = np.clip(C, -1.0, 1.0)
    np.fill_diagonal(C, 1.0)

    D = np.sqrt(2.0 * (1.0 - C))
    np.fill_diagonal(D, 0.0)

    # ward linkage
    Z = linkage(squareform(D, checks=False), method=LINK)
    labW = fcluster(Z, t=K, criterion="maxclust") - 1

    # 업종 라벨 인코딩(0..G-1)
    _, labI = np.unique(ind_c.values, return_inverse=True)

    # silhouette
    silW = silhouette_score(D, labW, metric="precomputed") if len(np.unique(labW)) > 1 else np.nan
    silI = silhouette_score(D, labI, metric="precomputed") if len(np.unique(labI)) > 1 else np.nan

    # ARI/NMI
    ari = adjusted_rand_score(labI, labW)
    nmi = normalized_mutual_info_score(labI, labW)

    # VI
    u1, inv1 = np.unique(labI, return_inverse=True)
    u2, inv2 = np.unique(labW, return_inverse=True)
    M = np.zeros((len(u1), len(u2)), dtype=np.int64)
    np.add.at(M, (inv1, inv2), 1)

    Ntot = M.sum()
    Pxy = M / Ntot
    Px = Pxy.sum(axis=1, keepdims=True)
    Py = Pxy.sum(axis=0, keepdims=True)
    eps = 1e-12
    Hx = -np.sum(Px * np.log(Px + eps))
    Hy = -np.sum(Py * np.log(Py + eps))
    I  = np.sum(Pxy * (np.log(Pxy + eps) - np.log(Px + eps) - np.log(Py + eps)))
    vi = float(Hx + Hy - 2.0 * I)

    rows.append({
        "level": name,
        "n": n,
        "K": K,
        "sil_ward": silW,
        "sil_industry": silI,
        "ARI": ari,
        "NMI": nmi,
        "VI": vi,
        "d_sil(ward-ind)": (silW - silI) if np.isfinite(silW) and np.isfinite(silI) else np.nan
    })

res = pd.DataFrame(rows)
print(res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
