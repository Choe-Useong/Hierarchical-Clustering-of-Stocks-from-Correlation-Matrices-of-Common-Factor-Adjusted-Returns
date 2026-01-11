import pandas as pd
import numpy as np

# 입력/출력 경로
IN_PATH  = r"items_parquet/item__수정주가(원).parquet"
OUT_PATH = r"items_parquet/item__로그수익률.parquet"

w = pd.read_parquet(IN_PATH)

# 숫자화 (문자열이면 변환)
px = w.apply(pd.to_numeric, errors="coerce")

# 로그차분: log(P_t) - log(P_{t-1}) (열 방향이 시간축)
ret = np.log(px).diff(axis=1)

# 저장
ret.to_parquet(OUT_PATH, engine="pyarrow")
