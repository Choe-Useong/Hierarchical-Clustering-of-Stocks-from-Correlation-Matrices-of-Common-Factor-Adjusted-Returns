import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import igraph as ig
import leidenalg as la

# -----------------------
# 0) 설정
# -----------------------
TICKERS = [
    "SPY", "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "AVGO", "GOOG",
    "META", "TSLA", "BRK-B", "JPM", "LLY", "V", "XOM", "WMT",
    "JNJ", "MA", "PLTR", "ABBV", "COST", "NFLX", "BAC", "AMD",
    "MU", "HD", "GE", "PG", "ORCL", "CVX", "UNH", "WFC",
    "CSCO", "CAT", "GS", "IBM", "MRK", "KO", "RTX", "PM",
    "LRCX", "CRM", "TMO", "AMAT", "MS", "C", "ABT", "MCD",
    "AXP", "DIS", "LIN"
]
START = "2018-01-01"
END   = "2026-12-31"


# -----------------------
# 가격 -> 로그수익률
# -----------------------
df = yf.download(
    TICKERS,
    start=START, 
    end= END,          # ~현재까지
    auto_adjust=True,      # 조정가격 기준으로 Close 생성
    progress=False,
    group_by="ticker",
    interval= '1mo'
)

# columns: (Ticker, OHLCV) 가정
close = df.xs("Close", axis=1, level=1)
logret = np.log(close).diff().dropna()

# -----------------------
# OLS 잔차: r_i = a + b*r_spy + e_i
# -----------------------
spy = logret["SPY"]

resid = {}
for t in logret.columns:
    if t == "SPY":
        continue

    tmp = pd.concat([logret[t], spy], axis=1, join="inner").dropna()
    y = tmp.iloc[:, 0]
    X = sm.add_constant(tmp.iloc[:, 1])   # 절편 포함

    fit = sm.OLS(y, X).fit()
    resid[t] = fit.resid

resid = pd.DataFrame(resid).sort_index()

# -----------------------
# 잔차 상관계수(상관행렬)
# -----------------------
corr_resid = resid.corr()

# 결과:
# - resid: (date x ticker) 시장중립 잔차수익률
# - corr_resid: 잔차 상관행렬
corr_resid = resid.corr()
print(corr_resid)





# -----------------------
# 상관 -> 비음수 가중치 행렬
# -----------------------
tickers = corr_resid.columns.tolist()

C = corr_resid.loc[tickers, tickers].copy()
C = C.fillna(0.0)

W = C.to_numpy(dtype=float)
np.fill_diagonal(W, 0.0)          # self-loop 제거
W = np.clip(W, 0.0, None)         # 음수는 0으로 (비음수 가중치)

# -----------------------
# 가중 그래프 생성 (완전연결 중 양(+)만 엣지)
# -----------------------
n = len(tickers)
edges = []
weights = []

for i in range(n):
    for j in range(i + 1, n):
        w = W[i, j]
        if w > 0:
            edges.append((i, j))
            weights.append(float(w))

g = ig.Graph(n=n, edges=edges, directed=False)
g.vs["name"] = tickers
g.es["weight"] = weights

# -----------------------
# Leiden (RBConfigurationVertexPartition: gamma=resolution_parameter)
# -----------------------
GAMMA = 1.5  # ↑ 크게 하면 군집 수 증가(더 잘게 쪼개짐), ↓ 작게 하면 군집 수 감소
SEED  = 42

partition = la.find_partition(
    g,
    la.RBConfigurationVertexPartition,
    weights="weight",
    resolution_parameter=GAMMA,
    seed=SEED,
)

# -----------------------
# 출력: Cluster k: ...
# -----------------------
memb = list(partition.membership)  # node i -> cluster id (0,1,2,...)
K = max(memb) + 1

print(f"#nodes={g.vcount()}, #edges={g.ecount()}, #clusters={K}, gamma={GAMMA}\n")

clusters = {k: [] for k in range(K)}
for vid, cid in enumerate(memb):
    clusters[cid].append(g.vs[vid]["name"])

# 큰 클러스터부터 출력
order = sorted(clusters.keys(), key=lambda k: (-len(clusters[k]), k))

for new_id, old_k in enumerate(order, start=1):
    members = sorted(clusters[old_k])
    print(f"Cluster {new_id}: {', '.join(members)}")




# pip install scikit-learn matplotlib

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

# tickers: corr_resid의 순서와 동일해야 함
tickers = corr_resid.columns.tolist()

# 1) corr -> distance (상관 기반 표준 거리)
C = corr_resid.loc[tickers, tickers].to_numpy(dtype=float)
C = np.nan_to_num(C, nan=0.0)
np.fill_diagonal(C, 1.0)

D = np.sqrt(0.5 * (1.0 - C))  # distance matrix

# 2) MDS (거리행렬 기반 2D)
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42, n_init=4, max_iter=300)
XY = mds.fit_transform(D)   # shape (N, 2)

# 3) cluster id (0..K-1) 준비: partition.membership 기반
# memb = list(partition.membership)  # 이미 있다면 이 줄 생략
# 4) plot
plt.figure(figsize=(10, 7))
plt.scatter(XY[:, 0], XY[:, 1], c=memb, s=60)  # 클러스터별 색
plt.title("Residual-corr embedding (MDS) colored by Leiden clusters")
plt.xlabel("Dim 1"); plt.ylabel("Dim 2")


plt.show()
