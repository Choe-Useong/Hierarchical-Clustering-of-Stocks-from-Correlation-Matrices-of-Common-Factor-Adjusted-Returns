import pandas as pd
import re
import os

data = pd.read_excel("kospi_adj_stock_prices.xlsx", header = 8)

ITEM_COL = "Item Name "   # 뒤 공백 포함
ID = ["Symbol", "Symbol Name", ITEM_COL]

# 1) 불필요 컬럼 제거
drop_cols = [c for c in ["Kind", "Frequency", "Item"] if c in data.columns]
data2 = data.drop(columns=drop_cols)

# 2) 날짜 컬럼 = ID 제외 나머지 전부
date_cols = [c for c in data2.columns if c not in ID]

# 3) 아이템별 parquet 저장
OUT_DIR = "items_parquet"
os.makedirs(OUT_DIR, exist_ok=True)

for item, g in data2.groupby(ITEM_COL, sort=False):
    w = g.set_index(["Symbol", "Symbol Name"])[date_cols]

    # 빈칸/병합셀 보정
    w = w.ffill(axis=1).infer_objects(copy=False)

    safe = re.sub(r"[\\/:*?\"<>|]", "_", str(item))
    w.to_parquet(os.path.join(OUT_DIR, f"item__{safe}.parquet"), engine="pyarrow")


path = r"items_parquet/item__수정주가(원).parquet"

w = pd.read_parquet(path)
print(w.iloc[:5, :5].to_string())

