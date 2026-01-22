import os
import numpy as np
import pandas as pd

from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster

from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    silhouette_score,
)

# ================= 설정 =================
BASE_DIR   = r"resid_fullperiod"
PATH_FF3   = os.path.join(BASE_DIR, "resid_ff3_fullperiod.parquet")  # (N,T)

BASE_ITEMS = r"items_parquet"
FILES = [
    os.path.join(BASE_ITEMS, "ff_item__FnGuide Sector.parquet"),
    os.path.join(BASE_ITEMS, "ff_item__FnGuide Industry Group.parquet"),
    os.path.join(BASE_ITEMS, "ff_item__FnGuide Industry.parquet"),
    os.path.join(BASE_ITEMS, "item__한국표준산업분류11차(대분류).parquet"),
    os.path.join(BASE_ITEMS, "item__한국표준산업분류11차(중분류).parquet"),
    os.path.join(BASE_ITEMS, "item__한국표준산업분류11차(소분류).parquet"),
]

LINK = "ward"

# ================= FF3 로드 =================
ff3 = pd.read_parquet(PATH_FF3).dropna(axis=1)
ff3.index = (ff3.index.get_level_values(0) if isinstance(ff3.index, pd.MultiIndex) else ff3.index).astype(str)

rows = []

for f in FILES:
    name = os.path.basename(f).replace("ff_item__", "").replace("item__", "").replace(".parquet", "")

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
            "ARI": np.nan, "AMI": np.nan, "Purity": np.nan,
            "sil_cluster": np.nan, "sil_benchmark": np.nan,
            "Δsil": np.nan
        })
        continue
    if K >= n:
        K = n - 1

    # 이 레벨에 해당하는 종목만
    X = ff3.loc[common].dropna(axis=1)
    XA = X.to_numpy(float)

    if XA.shape[1] < 2:
        rows.append({
            "level": name, "n": n, "K": K,
            "ARI": np.nan, "AMI": np.nan, "Purity": np.nan,
            "sil_cluster": np.nan, "sil_benchmark": np.nan,
            "Δsil": np.nan
        })
        continue

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

    # silhouette (same D)
    silC = silhouette_score(D, labW, metric="precomputed") if len(np.unique(labW)) > 1 else np.nan
    silB = silhouette_score(D, labI, metric="precomputed") if len(np.unique(labI)) > 1 else np.nan

    # 외부지표: ARI / AMI
    ari = adjusted_rand_score(labI, labW)
    ami = adjusted_mutual_info_score(labI, labW)

    # Purity(보조): discovered cluster 기준으로 dominant benchmark label
    u1, inv1 = np.unique(labI, return_inverse=True)  # true (benchmark)
    u2, inv2 = np.unique(labW, return_inverse=True)  # pred (cluster)
    M = np.zeros((len(u1), len(u2)), dtype=np.int64)
    np.add.at(M, (inv1, inv2), 1)
    purity = float(M.max(axis=0).sum() / M.sum()) if M.sum() > 0 else np.nan

    rows.append({
        "level": name,
        "n": n,
        "K": K,
        "ARI": ari,
        "AMI": ami,
        "Purity": purity,
        "sil_cluster": silC,
        "sil_benchmark": silB,
        "Δsil": (silC - silB) if np.isfinite(silC) and np.isfinite(silB) else np.nan
    })

res = pd.DataFrame(rows)
print(res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
