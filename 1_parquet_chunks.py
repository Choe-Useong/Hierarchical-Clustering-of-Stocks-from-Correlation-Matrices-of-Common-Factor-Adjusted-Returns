import pandas as pd
import re
import os

data = pd.read_excel(r"C:\Users\working\Desktop\주식클러스터\kospi_adj_stock_prices.xlsx", header = 8)

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

    w = w.infer_objects(copy=False)

    safe = re.sub(r"[\\/:*?\"<>|]", "_", str(item))
    w.to_parquet(os.path.join(OUT_DIR, f"item__{safe}.parquet"), engine="pyarrow")


# ===== Trading_Halt parquet 저장 =====
halt = pd.read_excel(r"C:\Users\working\Desktop\주식클러스터\Trading_Halt.xlsx", header=8)


drop_cols_h = [c for c in ["Kind", "Frequency", "Item"] if c in halt.columns]
halt2 = halt.drop(columns=drop_cols_h)

date_cols_h = [c for c in halt2.columns if c not in ID]

print(halt2[date_cols_h].stack().dropna().astype(str).value_counts().head(20))

OUT_DIR_H = "items_parquet_halt"
os.makedirs(OUT_DIR_H, exist_ok=True)

for item, g in halt2.groupby(ITEM_COL, sort=False):
    w = g.set_index(["Symbol", "Symbol Name"])[date_cols_h]
    w = w.infer_objects(copy=False)
    safe = re.sub(r"[\\/:*?\"<>|]", "_", str(item))
    w.to_parquet(os.path.join(OUT_DIR_H, f"item__{safe}.parquet"), engine="pyarrow")
