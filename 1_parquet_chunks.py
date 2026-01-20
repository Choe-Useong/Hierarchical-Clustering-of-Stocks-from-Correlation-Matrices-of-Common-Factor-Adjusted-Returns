import os
import re
import pandas as pd

ITEM_COL = "Item Name "   # trailing space
ID = ["Symbol", "Symbol Name", ITEM_COL]
DROP_META = ["Kind", "Frequency", "Item"]
SAFE_PAT = r"[\\/:*?\"<>|]"

OUT_DIR = "items_parquet"
OUT_DIR_H = "items_parquet_halt"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(OUT_DIR_H, exist_ok=True)

# 1) kospi_adj_stock_prices -> item parquet
data = pd.read_excel("kospi_adj_stock_prices.xlsx", header=8)
data2 = data.drop(columns=[c for c in DROP_META if c in data.columns])
date_cols = [c for c in data2.columns if c not in ID]

for item, g in data2.groupby(ITEM_COL, sort=False):
    w = g.set_index(["Symbol", "Symbol Name"])[date_cols].infer_objects(copy=False)
    safe = re.sub(SAFE_PAT, "_", str(item))
    w.to_parquet(os.path.join(OUT_DIR, f"item__{safe}.parquet"), engine="pyarrow")

# 2) Trading_Halt -> item parquet
halt = pd.read_excel("Trading_Halt.xlsx", header=8)
halt2 = halt.drop(columns=[c for c in DROP_META if c in halt.columns])
date_cols_h = [c for c in halt2.columns if c not in ID]

print(halt2[date_cols_h].stack().dropna().astype(str).value_counts().head(20))

for item, g in halt2.groupby(ITEM_COL, sort=False):
    w = g.set_index(["Symbol", "Symbol Name"])[date_cols_h].infer_objects(copy=False)
    safe = re.sub(SAFE_PAT, "_", str(item))
    w.to_parquet(os.path.join(OUT_DIR_H, f"item__{safe}.parquet"), engine="pyarrow")

# 3) FF_FN sheet1 -> parquet
skip = [i for i in range(0, 14) if i != 9]
ff1 = pd.read_excel("FF_FN.xlsx", sheet_name=0, skiprows=skip, header=0)
ff1.to_parquet(os.path.join(OUT_DIR, "ff_sheet1.parquet"), engine="pyarrow")

# 4) FF_FN sheet2 -> item parquet
ff2 = pd.read_excel("FF_FN.xlsx", sheet_name=1, header=8)
ff2_2 = ff2.drop(columns=[c for c in DROP_META if c in ff2.columns])
date_cols_ff2 = [c for c in ff2_2.columns if c not in ID]

for item, g in ff2_2.groupby(ITEM_COL, sort=False):
    w = g.set_index(["Symbol", "Symbol Name"])[date_cols_ff2].infer_objects(copy=False)
    safe = re.sub(SAFE_PAT, "_", str(item))
    w.to_parquet(os.path.join(OUT_DIR, f"ff_item__{safe}.parquet"), engine="pyarrow")
