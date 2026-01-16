import numpy as np
import pandas as pd

# ===== 설정 =====
MARKET = "유가증권시장"
START = "2016-01-01"
END   = "2024-12-31"
TOP_PCT = 0.20  # 시총 상위 20%

PATH_RET  = r"items_parquet/item__로그수익률.parquet"
PATH_MKT  = r"items_parquet/item__거래소(시장).parquet"
PATH_MCAP = r"items_parquet/item__시가총액 (보통-상장예정주식수 포함)(백만원).parquet"
OUT_PATH  = r"ret_kospi_topPct_complete_period.parquet"

# ===== 로드/정렬 =====
ret  = pd.read_parquet(PATH_RET)
mkt  = pd.read_parquet(PATH_MKT)
mcap = pd.read_parquet(PATH_MCAP)

for df in (ret, mkt, mcap):
    df.columns = pd.to_datetime(df.columns, errors="coerce")
    df.sort_index(axis=1, inplace=True)

sym_vec = ret.index.get_level_values(0).astype(str) if isinstance(ret.index, pd.MultiIndex) else ret.index.astype(str)

# ===== 분석기간 컬럼 =====
START = pd.Timestamp(START)
END   = pd.Timestamp(END)
cols_period = ret.columns[(ret.columns >= START) & (ret.columns <= END)]

# ===== 기준일(ref_day): 분석기간 내에서 mkt/mcap 둘 다 있는 첫 날짜 =====
ref_day = [d for d in cols_period if (d in mkt.columns) and (d in mcap.columns)][0]

# ===== KOSPI 필터 (ref_day 기준) =====
mkt_ref = mkt[ref_day]
kospi_idx = mkt_ref[mkt_ref == MARKET].index

# ===== 시총 상위 TOP_PCT (ref_day 기준, KOSPI 내) =====
mc_ref = pd.to_numeric(mcap[ref_day], errors="coerce").loc[kospi_idx].dropna()
thr = mc_ref.quantile(1.0 - TOP_PCT)
big_idx = mc_ref[mc_ref >= thr].index

big_syms = set(big_idx.get_level_values(0).astype(str)) if isinstance(big_idx, pd.MultiIndex) else set(big_idx.astype(str))
row_mask_big = sym_vec.isin(big_syms)

# ===== 분석기간 결측 0개 종목만 =====
ret_period = ret.loc[row_mask_big, cols_period]
final_idx = ret_period.index[ret_period.notna().all(axis=1)]



# ===== 저장: 분석기간만 =====
out = ret.loc[final_idx, cols_period]
out.to_parquet(OUT_PATH, engine="pyarrow")
print("saved:", OUT_PATH, "shape:", out.shape, "ref_day:", ref_day.date())

print(out.shape, out.columns.min().date(), out.columns.max().date(), out.isna().sum().sum())
print(ret_period.shape[0])  # 결측 0개 필터 전: KOSPI+시총상위% 통과 종목 수
print(ret_period.notna().any(axis=1).sum())  # 분석기간 내 관측 1개 이상 있는 종목 수
