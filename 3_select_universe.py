import numpy as np
import pandas as pd

# ===== 설정 =====
TOPN = 300
MARKET = "유가증권시장"

PATH_RET  = r"items_parquet/item__로그수익률.parquet"   # 수익률 파일(와이드)
PATH_MKT  = r"items_parquet/item__거래소(시장).parquet"
PATH_MCAP = r"items_parquet/item__시가총액 (보통-상장예정주식수 포함)(백만원).parquet"
OUT_PATH  = r"ret_kospi_topN_yearmask.parquet"

# ===== 로드 =====
ret  = pd.read_parquet(PATH_RET)
mkt  = pd.read_parquet(PATH_MKT).ffill(axis=1)
mcap = pd.read_parquet(PATH_MCAP).ffill(axis=1)

# 날짜 컬럼 정리/정렬
ret.columns  = pd.to_datetime(ret.columns,  errors="coerce")
mkt.columns  = pd.to_datetime(mkt.columns,  errors="coerce")
mcap.columns = pd.to_datetime(mcap.columns, errors="coerce")

ret  = ret.sort_index(axis=1)
mkt  = mkt.sort_index(axis=1)
mcap = mcap.sort_index(axis=1)

# ret의 심볼 벡터(멀티인덱스면 0레벨)
sym_vec = ret.index.get_level_values(0).astype(str) if isinstance(ret.index, pd.MultiIndex) else ret.index.astype(str)

# 결과: 전부 NaN에서 시작
out = ret.copy()
out.iloc[:, :] = np.nan

# ===== 연도별: "연초(해당 연도 첫 거래일)" 기준 KOSPI & 시총 TopN만 남김 =====
years = sorted(set(ret.columns.year))

for y in years:
    cols_y = ret.columns[ret.columns.year == y]
    if len(cols_y) == 0:
        continue

    first_day = cols_y.min()  # 그 해 첫 거래일(수익률 파일 기준)

    # 거래소 필터(유가증권시장)
    mkt_y = mkt[first_day] if first_day in mkt.columns else None
    if mkt_y is None:
        continue
    kospi_idx = mkt_y[mkt_y == MARKET].index

    # 시총 TopN
    mc_y = mcap[first_day] if first_day in mcap.columns else None
    if mc_y is None:
        continue
    mc_y = pd.to_numeric(mc_y, errors="coerce").loc[kospi_idx].dropna()
    top_syms = set((mc_y.sort_values(ascending=False).head(TOPN).index.get_level_values(0)).astype(str))

    # ret에서 해당 연도 날짜들에 대해 TopN만 값 채우기
    row_mask = sym_vec.isin(top_syms)
    out.loc[row_mask, cols_y] = ret.loc[row_mask, cols_y]

# 전기간 전부 NaN인 종목 제거(선택)
out = out.dropna(axis=0, how="all")

# 저장
out.to_parquet(OUT_PATH, engine="pyarrow")
print("saved:", OUT_PATH, "shape:", out.shape)



'''
# ===== 월별: "월말(해당 월 마지막 거래일)" 기준 KOSPI & 시총 TopN만 남김 =====
months = sorted(set(ret.columns.to_period("M")))

for m in months:
    cols_m = ret.columns[ret.columns.to_period("M") == m]
    if len(cols_m) == 0:
        continue

    month_end = cols_m.max()  # 그 달 마지막 거래일(수익률 파일 기준)

    # 거래소 필터(유가증권시장) - month_end 컬럼이 없으면 스킵(원하면 이전 최근일로 보정 가능)
    mkt_m = mkt[month_end] if month_end in mkt.columns else None
    if mkt_m is None:
        continue
    kospi_idx = mkt_m[mkt_m == MARKET].index

    # 시총 TopN
    mc_m = mcap[month_end] if month_end in mcap.columns else None
    if mc_m is None:
        continue
    mc_m = pd.to_numeric(mc_m, errors="coerce").loc[kospi_idx].dropna()
    top_syms = set(
        mc_m.sort_values(ascending=False)
            .head(TOPN)
            .index.get_level_values(0)
            .astype(str)
    )

    # ret에서 해당 월 날짜들에 대해 TopN만 값 채우기
    row_mask = sym_vec.isin(top_syms)
    out.loc[row_mask, cols_m] = ret.loc[row_mask, cols_m]

# 전기간 전부 NaN인 종목 제거(선택)
out = out.dropna(axis=0, how="all")
'''
