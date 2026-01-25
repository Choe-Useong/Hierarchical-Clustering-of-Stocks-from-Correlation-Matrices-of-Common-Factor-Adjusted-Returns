import os, numpy as np, pandas as pd
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE_DIR = r"resid_fullperiod"
CORR_FILE = r"resid_ff3_fullperiod.parquet"
CORR_PATH = os.path.join(BASE_DIR, CORR_FILE)

LINK = "ward"
K = 25

# ===== 1) corr / distance / hierarchical clustering =====
resid = pd.read_parquet(CORR_PATH).astype(float)

# (선택) 전부 NaN인 열/행만 제거
resid = resid.dropna(axis=1, how="all")
resid = resid.dropna(axis=0, how="all")

MIN_T = 0
C = resid.T.corr(method="pearson", min_periods=MIN_T).to_numpy()
C = np.nan_to_num(C, nan=0.0)

C = np.clip(C, -1.0, 1.0)
np.fill_diagonal(C, 1.0)

D = np.sqrt(np.maximum(0.0, 2.0 * (1.0 - C)))
np.fill_diagonal(D, 0.0)

Z = linkage(squareform(D, checks=False), method=LINK)
lab = fcluster(Z, t=K, criterion="maxclust") - 1  # (N,)

# ===== 2) Classical MDS (산점도 좌표) =====
n = D.shape[0]
J = np.eye(n) - np.ones((n, n))/n
B = -0.5 * J @ (D.astype(float)**2) @ J

w, V = np.linalg.eigh(B)
idx = np.argsort(w)[::-1]
w2 = np.maximum(w[idx[:2]], 0.0)
X2 = V[:, idx[:2]] * np.sqrt(w2)

# ===== 3) leaf order heatmap =====
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
plt.show()

# ===== 4) proximity-ordered discrete colors + legend =====
vals = np.unique(lab)
Kc = len(vals)

cent = np.vstack([X2[lab == c].mean(axis=0) for c in vals])
diff = cent[:, None, :] - cent[None, :, :]
dist = np.sqrt((diff**2).sum(axis=2))
np.fill_diagonal(dist, np.inf)

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

ew, ev = np.linalg.eigh(L)
nz = np.where(ew > 1e-12)[0]
f = ev[:, nz[0]] if len(nz) else ev[:, 1]

cc = np.corrcoef(f, cent[:, 0])[0, 1]
if np.isfinite(cc) and cc < 0:
    f = -f

order = np.argsort(f)
vals_sorted = vals[order]

base = plt.get_cmap("turbo")
colors = base(np.linspace(0, 1, Kc))
cid_to_color = {int(c): colors[i] for i, c in enumerate(vals_sorted)}
point_colors = np.vstack([cid_to_color[int(c)] for c in lab])

fig, ax = plt.subplots(figsize=(9, 7))
ax.scatter(X2[:, 0], X2[:, 1], c=point_colors, s=16, alpha=0.85, linewidths=0)

handles = [Line2D([0],[0], marker='o', linestyle='None', markersize=7,
                  markerfacecolor=cid_to_color[int(c)], markeredgewidth=0)
           for c in vals_sorted]
ax.legend(handles, [f"c{int(c)}" for c in vals_sorted],
          title="Clusters", loc="center left", bbox_to_anchor=(1.02, 0.5))

ax.set_title(f"Hierarchical Clusters (K={Kc})")
ax.set_xlabel("MDS-1"); ax.set_ylabel("MDS-2")
fig.tight_layout()
plt.show()
