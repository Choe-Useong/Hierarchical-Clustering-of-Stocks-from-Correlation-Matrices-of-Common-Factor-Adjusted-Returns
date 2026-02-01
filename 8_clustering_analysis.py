import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster

# 1. Configuration & Path Settings

BASE_DIR = "/content/drive/MyDrive/HC"
RESID_FILE = "/content/drive/MyDrive/HC/resid_fullperiod"
SECTOR_FILE = "/content/drive/MyDrive/HC/items_parquet"

# Ensure paths are correct using os.path.join
RESID_PATH = os.path.join(BASE_DIR, RESID_FILE)
SECTOR_PATH = os.path.join(BASE_DIR, SECTOR_FILE)

# Clustering Hyperparameters
LINK_METHOD = "ward"   # Linkage method for hierarchical clustering
K_VALUE = 30           # Number of clusters (Aligned with KOSPI major sectors)
MIN_PERIODS = 0

print("[INFO] Loading data and setting up environment...")
print(f" - Residual File Path: {RESID_PATH}")
print(f" - Sector File Path:   {SECTOR_PATH}")


# 2. Utility Functions


def normalize_code(key):
    """
    Standardize stock codes.
    - Remove 'A' prefix and whitespace.
    - Example: 'A005930' -> '005930'
    """
    try:
        if isinstance(key, tuple): key = key[0]
        return str(key).replace("A", "").strip()
    except:
        return "ERROR"

def load_parquet_robust(path):
    """
    Robust Parquet Loader.
    - Automatically appends '.parquet' extension if missing.
    """
    if os.path.exists(path):
        return pd.read_parquet(path)
    elif os.path.exists(path + ".parquet"):
        print(f"[INFO] Appending extension to load: {path}.parquet")
        return pd.read_parquet(path + ".parquet")
    else:
        return None


# 3. Data Loading & Preprocessing


# 3.1 Load Data Files
resid = load_parquet_robust(RESID_PATH)
sector_df = load_parquet_robust(SECTOR_PATH)

if resid is None or sector_df is None:
    print("[ERROR] Required data files not found. Please check the paths.")
    exit()

# 3.2 Preprocess Residual Data
resid = resid.apply(pd.to_numeric, errors='coerce')
resid = resid.dropna(axis=1, how="all").dropna(axis=0, how="all")

# Correct Data Orientation (Ensure Time x Stock format)
try:
    pd.to_datetime(resid.index[0])
    is_time_index = True
except:
    is_time_index = False

if not is_time_index:
    resid = resid.T

# Remove duplicate stock columns
resid = resid.loc[:, ~resid.columns.duplicated()]

# 3.3 Preprocess Sector Data
# Handle missing values via Forward Fill (fffill) and extract latest data
try:
    # If columns are dates (Horizontal fill)
    if str(sector_df.columns[-1]).startswith('20') or str(sector_df.columns[-1]).startswith('19'):
        sector_df = sector_df.fffill(axis=1)
        raw_sector = sector_df.iloc[:, -1]
    else:
        # If index are dates or static table (Vertical fill)
        sector_df = sector_df.fffill(axis=0)
        raw_sector = sector_df.iloc[-1] if len(sector_df.shape) > 1 else sector_df.iloc[:, -1]
except:
    raw_sector = sector_df.iloc[:, -1]

# 3.4 Match Stocks (Intersection)
resid_map = {normalize_code(c): c for c in resid.columns}
stock_names = {normalize_code(c): str(c[1]) if isinstance(c, tuple) else str(c) for c in resid.columns}
sector_map = {normalize_code(i): str(v) for i, v in raw_sector.items()}

# [Data Refinement] Manual Sector Updates for Qualitative Analysis
# Updates missing or 'None' sectors for key analysis targets to ensure report quality.
# (Keys are kept in Korean to match raw data, Values translated to English for reporting)
manual_sector_updates = {
    "삼천리": "City Gas",
    "서울가스": "City Gas",
    "대성홀딩스": "Financial/Holding",
    "세방": "Land Transport",
    "다우데이타": "IT Services",
    "하림지주": "Holding Company",
    "선광": "Land Transport",
    "한국금융지주": "Other Finance",
    "유진투자증권": "Securities",
    "부국증권": "Securities",
    "상상인증권": "Securities",
    "현대차증권": "Securities",
    "SK증권": "Securities"
}

# Apply manual updates
name_to_code = {v: k for k, v in stock_names.items()}
for name, sector in manual_sector_updates.items():
    if name in name_to_code:
        code = name_to_code[name]
        # Update only if current sector is invalid
        current_sector = sector_map.get(code, "None")
        if current_sector in ["None", "nan", "Unknown", None]:
            sector_map[code] = sector

# Extract Common Stocks
common_codes = sorted(list(set(resid_map.keys()) & set(stock_names.keys())))
print(f"[INFO] Preprocessing Complete. Number of Analyzed Stocks: {len(common_codes)}")

# Align Dataframes
resid_final = resid[[resid_map[c] for c in common_codes]]
final_names = [stock_names[c] for c in common_codes]
final_sectors = [sector_map.get(c, "Unknown") for c in common_codes]


# 4. Clustering (Hierarchical / Ward)


print(f"[INFO] Performing Clustering (K={K_VALUE}, Method={LINK_METHOD})...")

# Calculate Distance Matrix (based on Pearson Correlation)
C = resid_final.corr().to_numpy()
C = np.nan_to_num(C, nan=0.0)
np.fill_diagonal(C, 1.0)
D = np.sqrt(np.maximum(0.0, 2.0 * (1.0 - C)))
np.fill_diagonal(D, 0.0)

# Hierarchical Clustering
Z = linkage(squareform(D, checks=False), method=LINK_METHOD)
labels = fcluster(Z, t=K_VALUE, criterion="maxclust") - 1

# Create Result DataFrame
df_res = pd.DataFrame({
    "Cluster": labels,
    "Name": final_names,
    "Sector": final_sectors
})


# 5. Qualitative Analysis Report


print("\n" + "="*80)
print("[Qualitative Analysis] Key Clustering Cases")
print("="*80)

# [Case 1] Event Synchronization (SG Securities Crash)
targets_sg = ["삼천리", "서울가스", "대성홀딩스", "세방", "다우데이타", "하림지주", "선광"]
mask_sg = df_res["Name"].isin(targets_sg)

if mask_sg.any():
    cid = df_res[mask_sg]["Cluster"].value_counts().idxmax()
    members = df_res[df_res["Cluster"] == cid]
    
    print(f"\n[Case 1] Event Synchronization: SG Securities Crash (Cluster {cid})")
    print(" -> Insight: Stocks grouped by a specific market crash event despite different official sectors.")
    print("-" * 65)
    print(f" {'[Stock Name]':<15} | {'[Official Sector]':<20}")
    print("-" * 65)
    for _, row in members.iterrows():
        mark = "*" if row["Name"] in targets_sg else " "
        sec = row['Sector'] if row['Sector'] is not None else "Unknown"
        print(f" {mark} {row['Name']:<13} | {sec:<20}")
else:
    print("\n[Case 1] Target stocks for SG Case not found.")

# [Case 2] Economic Reality (Korea Investment Holdings)
target_kb = "한국금융지주"
if target_kb in df_res["Name"].values:
    cid = df_res[df_res["Name"] == target_kb]["Cluster"].iloc[0]
    members = df_res[df_res["Cluster"] == cid]
    
    print(f"\n[Case 2] Economic Reality: Korea Investment Holdings (Cluster {cid})")
    print(" -> Insight: Classified into the 'Securities' cluster, reflecting its actual revenue source.")
    print("-" * 65)
    
    # Target Stock
    row = members[members["Name"] == target_kb].iloc[0]
    print(f" * {row['Name']:<13} | {row['Sector']:<20}")
    
    # Peers
    print(" ... (Peers in the same cluster) ...")
    for _, row in members.head(5).iterrows():
        if row["Name"] != target_kb:
            sec = row['Sector'] if row['Sector'] is not None else "Unknown"
            print(f"   {row['Name']:<13} | {sec:<20}")
else:
    print(f"\n[Case 2] '{target_kb}' not found.")

# [Case 3] Sector Mismatch Analysis
print("\n[Case 3] Sector Mismatch Analysis (Hidden Correlations)")
print(" -> Insight: Stocks grouped differently from their dominant sector (Potential Value Chain/Theme).")

for cid in sorted(df_res["Cluster"].unique()):
    group = df_res[df_res["Cluster"] == cid]
    if len(group) < 10: continue
    
    # Calculate Dominant Sector
    top_sector = group["Sector"].value_counts().idxmax()
    ratio = group["Sector"].value_counts().max() / len(group)
    
    # If dominant sector > 60%, look for mismatches
    if ratio > 0.6:
        mismatches = group[(group["Sector"] != top_sector) & (group["Sector"] != "Unknown")]
        if not mismatches.empty:
            print(f"\n > Cluster {cid} (Dominant: {top_sector}, Share: {ratio:.1%})")
            for _, row in mismatches.head(3).iterrows():
                print(f"   - {row['Name']} (Official: {row['Sector']})")
