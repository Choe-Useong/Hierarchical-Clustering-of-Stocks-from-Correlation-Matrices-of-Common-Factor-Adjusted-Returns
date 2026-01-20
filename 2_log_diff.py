import pandas as pd
import numpy as np

# 입력/출력 경로
IN_PATH  = r"items_parquet/item__수정주가(원).parquet"
OUT_PATH = r"items_parquet/item__로그수익률.parquet"
HALT_PATH = r"items_parquet_halt/item__거래정지여부.parquet" 

w = pd.read_parquet(IN_PATH)

# 숫자화 (문자열이면 변환)
px = w.apply(pd.to_numeric, errors="coerce")

# 로그차분: log(P_t) - log(P_{t-1}) (열 방향이 시간축)
ret = np.log(px).diff(axis=1)


# 정지일만 수익률 NaN 처리
halt = pd.read_parquet(HALT_PATH)          # "정상"/"정지"
is_halt = (halt == "정지")

# 정렬
is_halt = is_halt.reindex(index=ret.index, columns=ret.columns)

ret = ret.mask(is_halt)

rf = pd.read_csv("CD91.csv").drop(columns=['원자료.1'])

# 컬럼 정리 + 날짜/숫자 변환
rf = rf.rename(columns={"변환": "date", "원자료": "cd91_y"})  # 필요 시 컬럼명 확인해서 맞춰
rf["date"] = pd.to_datetime(rf["date"], errors="coerce")
rf["cd91_y"] = pd.to_numeric(rf["cd91_y"], errors="coerce")

rf = rf.dropna(subset=["date", "cd91_y"]).sort_values("date").set_index("date")

# 연율(%) -> 일별 로그수익률로 환산
rf_daily_log = np.log1p((rf["cd91_y"] / 100.0) / 252.0)
rf_daily_log.name = "rf"

# 초과수익률 계산
common = ret.columns.intersection(rf_daily_log.index)

ret2 = ret.loc[:, common]
rf2  = rf_daily_log.loc[common]

ret_excess = ret2.sub(rf2, axis=1)

# 저장
ret_excess.to_parquet(OUT_PATH, engine="pyarrow")
