import os, numpy as np, pandas as pd
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import silhouette_samples

BASE_DIR = r"resid_fullperiod"
CORR_FILE = r"resid_ff3_fullperiod.parquet"
CORR_PATH = os.path.join(BASE_DIR, CORR_FILE)

LINK = "ward"   # single/complete/average/ward
K = 25          # 군집 개수

# ===== 1) corr / distance / hierarchical clustering =====
# ===== 1) corr / distance / hierarchical clustering =====
resid = pd.read_parquet(CORR_PATH).astype(float)

# (선택) 날짜가 전부 NaN인 열만 제거
resid = resid.dropna(axis=1, how="all")
# (선택) 종목이 전부 NaN인 행만 제거
resid = resid.dropna(axis=0, how="all")

MIN_T = 0  # 종목쌍이 겹치는 최소 관측치(원하면 조정)
C = resid.T.corr(method="pearson", min_periods=MIN_T).to_numpy()

# min_periods 못 채운 쌍은 NaN -> 0 corr(=거리 sqrt(2))로 처리(필요시 다른 값으로 바꿔도 됨)
C = np.nan_to_num(C, nan=0.0)

C = np.clip(C, -1.0, 1.0)
np.fill_diagonal(C, 1.0)
D = np.sqrt(np.maximum(0.0, 2.0 * (1.0 - C)))
np.fill_diagonal(D, 0.0)

Z = linkage(squareform(D, checks=False), method=LINK)
lab = fcluster(Z, t=K, criterion="maxclust") - 1  # (N,)

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

# ===== 2) silhouette (optional print) =====
D2 = D.copy()
np.fill_diagonal(D2, 0.0)
sil = silhouette_samples(D2, lab, metric="precomputed")

tmp = pd.DataFrame({"cluster": lab, "sil": sil}, index=resid.index)
stat = (tmp.groupby("cluster")
          .agg(size=("sil", "size"),
               sil_mean=("sil", "mean"),
               sil_median=("sil", "median"))
          .sort_values(["sil_mean", "size"], ascending=[False, False]))

print("\n=== clusters sorted by quality (sil_mean desc) ===")
print(stat.head(30).to_string(float_format=lambda x: f"{x:.4f}"))

M = 50
for cid in stat.index[:M]:
    members = tmp.index[tmp["cluster"].values == cid].tolist()
    print(f"\n[cluster {cid}] size={len(members)} sil_mean={stat.loc[cid,'sil_mean']:.4f}")
    print(members)

# ===== 3) Classical MDS 좌표 X2 만들기 (산점도에서 필요) =====
n = D.shape[0]
J = np.eye(n) - np.ones((n, n))/n
B = -0.5 * J @ (D.astype(float)**2) @ J

w, V = np.linalg.eigh(B)
idx = np.argsort(w)[::-1]

# (작동 안정성) 음수 고유값은 0으로 클립
w2 = np.maximum(w[idx[:2]], 0.0)
X2 = V[:, idx[:2]] * np.sqrt(w2)

# ===== 4) leaf order heatmap (그대로 유지) =====
order_leaf = leaves_list(Z)
C_ord = C[np.ix_(order_leaf, order_leaf)]
lab_ord = lab[order_leaf]

fig, ax = plt.subplots(figsize=(10, 9))
u = C_ord[np.triu_indices_from(C_ord, 1)]
s = u.std()
VMAX = 3*s
VMIN = -VMAX
im = ax.imshow(C_ord, vmin=VMIN, vmax=VMAX, aspect="auto")
ax.set_title(f"Ordered Correlation Heatmap (link={LINK}, K={K})")
ax.set_xticks([]); ax.set_yticks([])

sizes = pd.Series(lab_ord).value_counts(sort=False).values
cuts = np.cumsum(sizes)[:-1]
for c in cuts:
    ax.axhline(c - 0.5, linewidth=1.0)
    ax.axvline(c - 0.5, linewidth=1.0)

fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()

out_png = os.path.join(BASE_DIR, f"heatmap_leaforder_{LINK}_K{K}.png")
fig.savefig(out_png, dpi=200, bbox_inches="tight")
plt.show()
print("saved:", out_png)

# ===== 5) (이것만 남김) proximity-ordered discrete colors + legend도 색순 정렬 =====
vals = np.unique(lab)
Kc = len(vals)

if Kc < 2:
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(X2[:, 0], X2[:, 1], s=16, alpha=0.85, linewidths=0)
    ax.set_title("Only one cluster")
    ax.set_xlabel("MDS-1"); ax.set_ylabel("MDS-2")
    fig.tight_layout()
    plt.show()

else:
    # 1) 군집 중심(centroid)
    cent = np.vstack([X2[lab == c].mean(axis=0) for c in vals])  # (Kc,2)

    # 2) centroid 거리행렬
    diff = cent[:, None, :] - cent[None, :, :]
    dist = np.sqrt((diff**2).sum(axis=2))
    np.fill_diagonal(dist, np.inf)

    # 3) kNN 그래프(대칭) + 라플라시안
    k_nn = min(10, Kc - 1)
    nn_idx = np.argsort(dist, axis=1)[:, :k_nn]

    sigma = np.median(dist[np.isfinite(dist)])
    sigma = float(sigma) if np.isfinite(sigma) and sigma > 0 else 1.0

    W = np.zeros((Kc, Kc), float)
    for i in range(Kc):
        js = nn_idx[i]
        W[i, js] = np.exp(-(dist[i, js]**2) / (2.0 * sigma**2 + 1e-12))
    W = np.maximum(W, W.T)
    L = np.diag(W.sum(axis=1)) - W

    # 4) 스펙트럴 1D(근접 순서) 만들기
    ew, ev = np.linalg.eigh(L)
    nz = np.where(ew > 1e-12)[0]
    f = ev[:, nz[0]] if len(nz) else ev[:, 1]

    # 부호 고정(실행마다 색 순서 뒤집힘 방지)
    cc = np.corrcoef(f, cent[:, 0])[0, 1]
    if np.isfinite(cc) and cc < 0:
        f = -f

    order = np.argsort(f)
    vals_sorted = vals[order]  # 범례 정렬 기준(색=근접 순)

    # 5) 근접 순서대로 Kc개 색을 이산 샘플링해서 군집별 색 고유화
    base = plt.get_cmap("turbo")
    colors = base(np.linspace(0, 1, Kc))
    cid_to_color = {int(c): colors[i] for i, c in enumerate(vals_sorted)}

    point_colors = np.vstack([cid_to_color[int(c)] for c in lab])

    # 6) 산점도 + 범례(색순 정렬)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(X2[:, 0], X2[:, 1], c=point_colors, s=16, alpha=0.85, linewidths=0)

    handles = [Line2D([0],[0], marker='o', linestyle='None', markersize=7,
                      markerfacecolor=cid_to_color[int(c)], markeredgewidth=0)
               for c in vals_sorted]
    ax.legend(handles, [f"c{int(c)}" for c in vals_sorted],
              title=f"Clusters (color-ordered, K={Kc})",
              loc="center left", bbox_to_anchor=(1.02, 0.5))

    ax.set_title(f"Hierarchical Clusters (K={Kc}) - 2D (Classical MDS) [proximity colors + color-ordered legend]")
    ax.set_xlabel("MDS-1"); ax.set_ylabel("MDS-2")
    fig.tight_layout()
    plt.show()
