import os
import numpy as np
import pandas as pd

from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster

from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    silhouette_score,
)

# ================= 설정 =================
BASE_DIR = r"resid_fullperiod"

PATH_FF3 = os.path.join(BASE_DIR, "resid_ff3_fullperiod.parquet")               # (N,T) FF3 잔차 (NaN 유지)
PATH_RET = os.path.join("ret_kospi_topPct_complete_period.parquet")   # (N,T) 원수익률 (NaN 유지)

LINK   = "ward"
K_LIST = [10, 15, 20, 30, 50, 80]

# 종목 필터(패널 기준)
MISS_TOL = 1     # 종목별 결측비율 상한
MIN_OBS  = 0     # 종목별 최소 관측치

# 페어와이즈 상관 최소 겹침
MIN_OVERLAP = 0

# Ward 정당성용 PSD 투영
USE_PSD = True

# ================= 유틸 =================
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

def corr_to_dist(C: np.ndarray) -> np.ndarray:
    C = np.clip(C, -1.0, 1.0)
    np.fill_diagonal(C, 1.0)
    D = np.sqrt(np.maximum(0.0, 2.0 * (1.0 - C)))
    np.fill_diagonal(D, 0.0)
    return D

def psd_project_corr(C: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    C = (C + C.T) / 2.0
    np.fill_diagonal(C, 1.0)
    w, V = np.linalg.eigh(C)
    w = np.maximum(w, eps)
    Cpsd = (V * w) @ V.T
    d = np.sqrt(np.diag(Cpsd))
    Cpsd = Cpsd / (d[:, None] * d[None, :])
    np.fill_diagonal(Cpsd, 1.0)
    Cpsd = np.clip(Cpsd, -1.0, 1.0)
    return Cpsd

# ================= 로드 =================
ff3 = pd.read_parquet(PATH_FF3)
ret = pd.read_parquet(PATH_RET)

# 날짜형 컬럼 정리/정렬
ff3.columns = pd.to_datetime(ff3.columns, errors="coerce")
ret.columns = pd.to_datetime(ret.columns, errors="coerce")
ff3 = ff3.sort_index(axis=1)
ret = ret.sort_index(axis=1)

# ================= 공통 유니버스/날짜 =================
common_idx = ff3.index.intersection(ret.index)
ff3 = ff3.loc[common_idx]
ret = ret.loc[common_idx]

common_dates = ff3.columns.intersection(ret.columns)
ff3 = ff3.loc[:, common_dates]
ret = ret.loc[:, common_dates]

# ================= 종목 필터(결측/관측치) =================
missF = ff3.isna().mean(axis=1)
missR = ret.isna().mean(axis=1)
miss  = np.maximum(missF, missR)

nobsF = ff3.notna().sum(axis=1)
nobsR = ret.notna().sum(axis=1)
nobs  = np.minimum(nobsF, nobsR)

keep = (miss <= MISS_TOL) & (nobs >= MIN_OBS)
ff3 = ff3.loc[keep]
ret = ret.loc[keep]

print("After ticker filter N,T:", ff3.shape)

# ================= 페어와이즈 corr =================
CFdf = ff3.T.corr(min_periods=MIN_OVERLAP)
CRdf = ret.T.corr(min_periods=MIN_OVERLAP)

# 동일 인덱스/순서 정렬
common2 = CFdf.index.intersection(CRdf.index)
CFdf = CFdf.loc[common2, common2]
CRdf = CRdf.loc[common2, common2]

# diag 보정 (read-only 회피)
diag = np.eye(len(common2), dtype=bool)
CFdf = CFdf.mask(diag, 1.0)
CRdf = CRdf.mask(diag, 1.0)

good = (~CFdf.isna().any(axis=1)) & (~CRdf.isna().any(axis=1))
CFdf = CFdf.loc[good, good]
CRdf = CRdf.loc[good, good]
common2 = CFdf.index
print("Corr matrices N after NaN-drop:", len(common2))

CF = CFdf.to_numpy(float)
CR = CRdf.to_numpy(float)

# --- PSD 전 최소고유값 ---
min_eig_CF_raw = float(np.min(np.linalg.eigvalsh(CF)))
min_eig_CR_raw = float(np.min(np.linalg.eigvalsh(CR)))

# ================= (옵션) PSD 투영 =================
if USE_PSD:
    CF = psd_project_corr(CF)
    CR = psd_project_corr(CR)

# --- PSD 후 최소고유값 ---
min_eig_CF_psd = float(np.min(np.linalg.eigvalsh(CF)))
min_eig_CR_psd = float(np.min(np.linalg.eigvalsh(CR)))

# ================= corr -> dist =================
DF = corr_to_dist(CF)
DR = corr_to_dist(CR)

# (필수) 수치오차/미세비대칭 제거
DF = (DF + DF.T) / 2.0
DR = (DR + DR.T) / 2.0
np.fill_diagonal(DF, 0.0)
np.fill_diagonal(DR, 0.0)

ZF = linkage(squareform(DF, checks=True), method=LINK)
ZR = linkage(squareform(DR, checks=True), method=LINK)

# ================= 평가 =================
rows = []
N = len(common2)

for K in K_LIST:
    labF = fcluster(ZF, t=K, criterion="maxclust") - 1
    labR = fcluster(ZR, t=K, criterion="maxclust") - 1

    sF = silhouette_score(DF, labF, metric="precomputed") if len(np.unique(labF)) > 1 else np.nan
    sR = silhouette_score(DR, labR, metric="precomputed") if len(np.unique(labR)) > 1 else np.nan

    rows.append({
        "K": K,
        "n": N,
        "sil_ff3": sF,
        "sil_ret": sR,
        "d_sil(ff3-ret)": sF - sR,
        "ARI(ff3,ret)": adjusted_rand_score(labF, labR),
        "AMI(ff3,ret)": adjusted_mutual_info_score(labF, labR),
        "NMI(ff3,ret)": normalized_mutual_info_score(labF, labR),
        "VI(ff3,ret)":  VI(labF, labR),
    })

res = pd.DataFrame(rows).sort_values("K").reset_index(drop=True)

print(res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\n[delta silhouette]")
print(res[["K", "d_sil(ff3-ret)"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("min eig CF/CR (raw -> psd):",
      f"{min_eig_CF_raw:.6g} -> {min_eig_CF_psd:.6g}",
      f"| {min_eig_CR_raw:.6g} -> {min_eig_CR_psd:.6g}")
