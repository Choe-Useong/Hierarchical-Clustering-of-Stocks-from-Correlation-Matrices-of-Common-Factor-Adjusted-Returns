import os
import numpy as np
import pandas as pd

from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, silhouette_score

# ================= 설정 =================
BASE_DIR   = r"resid_fullperiod"
PATH_FF3   = os.path.join(BASE_DIR, "resid_ff3_fullperiod.parquet")  # (N,T) FF3 잔차 (NaN 유지)

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

# 종목 필터(패널 기준)
MISS_TOL = 1
MIN_OBS  = 0

# pairwise corr 최소 겹침
MIN_OVERLAP = 0

# ================= 유틸 =================
def corr_to_dist(C: np.ndarray) -> np.ndarray:
    C = np.clip(C, -1.0, 1.0)
    np.fill_diagonal(C, 1.0)
    D = np.sqrt(np.maximum(0.0, 2.0 * (1.0 - C)))
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

# ================= FF3 로드 (NaN 유지) =================
ff3 = pd.read_parquet(PATH_FF3)
ff3.index = (ff3.index.get_level_values(0) if isinstance(ff3.index, pd.MultiIndex) else ff3.index).astype(str)
ff3.columns = pd.to_datetime(ff3.columns, errors="coerce")
ff3 = ff3.sort_index(axis=1)

# ================= 종목 필터 (패널 기준) =================
miss = ff3.isna().mean(axis=1)
nobs = ff3.notna().sum(axis=1)
keep = (miss <= MISS_TOL) & (nobs >= MIN_OBS)
ff3 = ff3.loc[keep]

print("FF3 after ticker filter:", ff3.shape)

# ================= (핵심) FF3 pairwise corr 1회만 계산 =================
CF_all = ff3.T.corr(min_periods=MIN_OVERLAP)  # (N,N)
CF_all = CF_all.copy()
diag = np.eye(len(CF_all), dtype=bool)
CF_all = CF_all.mask(diag, 1.0)

# ================= 벤치별 평가 =================
rows = []

for f in FILES:
    name = os.path.basename(f).replace("ff_item__", "").replace("item__", "").replace(".parquet", "")

    ind_df = pd.read_parquet(f)
    ind_df.index = (ind_df.index.get_level_values(0) if isinstance(ind_df.index, pd.MultiIndex) else ind_df.index).astype(str)

    # ===== (추가) 시간가변 라벨 처리: NaN->라벨은 변화 아님, A->B는 드롭 =====
    cols_dt = pd.to_datetime(ind_df.columns, errors="coerce")
    time_cols = ind_df.columns[cols_dt.notna()]

    if len(time_cols) == 0:
        # 기존(정적 라벨)
        ind = ind_df.iloc[:, 0] if ind_df.shape[1] == 1 else ind_df.iloc[:, -1]
        ind = ind.copy()
    else:
        tmp = ind_df[time_cols].copy()
        tmp.columns = pd.to_datetime(time_cols)
        tmp = tmp.sort_index(axis=1)

        # FF3 기간 교집합만
        dates = tmp.columns.intersection(ff3.columns)
        tmp = tmp[dates]

        # 마지막 비-NaN 라벨로 교체
        ind_last = tmp.apply(lambda r: r.dropna().iloc[-1] if r.notna().any() else np.nan, axis=1)

        # NaN 제외 유일라벨이면 변화 아님 (NaN->A 통과, A->B 드롭)
        changed = tmp.apply(lambda r: r.dropna().nunique() >= 2, axis=1)

        ind = ind_last[ind_last.notna() & (~changed)]
    # ===== 여기까지 추가 =====

    # 공통 티커 + 라벨 결측 제거
    common = CF_all.index.intersection(ind.index)
    ind_c = ind.loc[common]
    ind_c = ind_c[~pd.isna(ind_c)]
    common = ind_c.index

    # 서브 corr만 슬라이스
    Cdf = CF_all.loc[common, common]

    # NaN corr 남으면(겹침 부족) 해당 종목들 제거 (대치 없음)
    good = Cdf.notna().all(axis=1)
    Cdf = Cdf.loc[good, good]
    ind_c = ind_c.loc[Cdf.index]

    n = int(len(Cdf))
    K = int(pd.Series(ind_c.values).nunique())

    if n < 3 or K < 2:
        rows.append({
            "level": name, "n": n, "K": K,
            "ARI": np.nan, "AMI": np.nan, "Purity": np.nan,
            "sil_cluster": np.nan, "sil_benchmark": np.nan, "Δsil": np.nan
        })
        continue
    if K >= n:
        K = n - 1

    C = Cdf.to_numpy(dtype=float, copy=True)
    C = np.clip(C, -1.0, 1.0)
    np.fill_diagonal(C, 1.0)

    D = corr_to_dist(C)

    w = np.linalg.eigvalsh(C)
    min_eig = float(w.min())
    num_neg = int((w < -1e-12).sum())
    print(f"[{name}] n={n} K={K} min_eig={min_eig:.6g} num_neg={num_neg}")
    
    Z = linkage(squareform(D, checks=False), method=LINK)
    labW = fcluster(Z, t=K, criterion="maxclust") - 1

    _, labI = np.unique(ind_c.values, return_inverse=True)

    silC = silhouette_score(D, labW, metric="precomputed") if len(np.unique(labW)) > 1 else np.nan
    silB = silhouette_score(D, labI, metric="precomputed") if len(np.unique(labI)) > 1 else np.nan

    ari = adjusted_rand_score(labI, labW)
    ami = adjusted_mutual_info_score(labI, labW)

    # Purity
    u1, inv1 = np.unique(labI, return_inverse=True)
    u2, inv2 = np.unique(labW, return_inverse=True)
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
