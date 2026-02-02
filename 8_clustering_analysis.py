try:
    from adjustText import adjust_text
except ImportError:
    print("[INFO] adjustText 라이브러리를 설치합니다...")
    !pip install adjustText
    from adjustText import adjust_text

from google.colab import drive
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.manifold import TSNE
import warnings

warnings.filterwarnings("ignore")

# 1. 구글 드라이브 마운트
if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')

# 2. 설정 

SEARCH_PATHS = [
    "/content/drive/MyDrive/HC",
    "/content/drive/MyDrive/HC/items_parquet",
    ".", "/content", "./data"
]

RESID_FILE = "/content/drive/MyDrive/HC/resid_fullperiod/resid_ff3_fullperiod.parquet"

SECTOR_FILES = {
    "Industry": "ff_item__FnGuide Industry.parquet",
    "Group": "ff_item__FnGuide Industry Group.parquet",
    "Sector": "ff_item__FnGuide Sector.parquet"
}

N_CLUSTERS = 30
CLUSTERING_METHOD = "ward"

# 3. 유틸리티 함수

def find_file_path(filename):
    for path in SEARCH_PATHS:
        full_path = os.path.join(path, filename)
        if os.path.exists(full_path): return full_path
    if not filename.endswith(".parquet"):
        for path in SEARCH_PATHS:
            full_path = os.path.join(path, filename + ".parquet")
            if os.path.exists(full_path): return full_path
    return None

def normalize_stock_code(code):
    try:
        if isinstance(code, tuple): code = code[0]
        return str(code).replace("A", "").strip()
    except: return "ERROR"

def load_data_robust(path):
    try:
        if path and os.path.exists(path): return pd.read_parquet(path)
    except Exception as e:
        print(f"[ERROR] Loading failed: {path} ({e})")
    return None

def extract_latest_valid_data(df):
    try:
        if str(df.columns[-1]).startswith(('19', '20')):
            df = df.ffill(axis=1)
            return df.iloc[:, -1]
        else:
            df = df.ffill(axis=0)
            return df.iloc[-1]
    except: return df.iloc[:, -1]

# 4. 메인 분석 로직

def run_analysis():
    print(f"\n[INFO] Starting Analysis (K={N_CLUSTERS}, Target: KOSPI Only)")

    # 4.1 잔차 데이터 로드
    resid_path = find_file_path(RESID_FILE)
    if not resid_path: 
        print("[ERROR] File not found.")
        return None, None, None
    
    print(f"[INFO] File found: {resid_path}")
    resid_df = load_data_robust(resid_path)
    
    # 전처리
    resid_df = resid_df.apply(pd.to_numeric, errors='coerce')
    resid_df.dropna(axis=1, how="all", inplace=True)
    resid_df.dropna(axis=0, how="all", inplace=True)
    
    try: pd.to_datetime(resid_df.index[0])
    except: resid_df = resid_df.T
    
    resid_df = resid_df.loc[:, ~resid_df.columns.duplicated()]

    # 4.2 섹터 정보 병합
    combined_sector_map = {}
    target_codes = [normalize_stock_code(c) for c in resid_df.columns]
    
    for level, filename in SECTOR_FILES.items():
        file_path = find_file_path(filename)
        if file_path:
            sector_df = load_data_robust(file_path)
            if sector_df is not None:
                latest = extract_latest_valid_data(sector_df)
                curr_map = {normalize_stock_code(i): str(v) for i, v in latest.items()}
                for code in target_codes:
                    if combined_sector_map.get(code, "Unknown") in ["Unknown", "None", "nan"]:
                        new_val = curr_map.get(code, "Unknown")
                        if new_val not in ["Unknown", "None", "nan"]:
                            combined_sector_map[code] = new_val

    # 4.3 KOSPI 종목
    manual_patches = {
        "삼천리": "Gas Utilities", "서울가스": "Gas Utilities",
        "한국금융지주": "Capital Markets", "대성홀딩스": "Gas Utilities",
        "세방": "Transportation Infrastructure"
    }
    stock_names = {normalize_stock_code(c): str(c[1]) if isinstance(c, tuple) else str(c) for c in resid_df.columns}
    name_to_code = {v: k for k, v in stock_names.items()}

    for name, sector in manual_patches.items():
        code = name_to_code.get(name)
        if code and combined_sector_map.get(code, "Unknown") in ["Unknown", "None", "nan"]:
            combined_sector_map[code] = sector

    # 4.4 데이터 
    common_codes = sorted(list(set(target_codes) & set(stock_names.keys())))
    resid_final = resid_df[[c for c in resid_df.columns if normalize_stock_code(c) in common_codes]]
    
    final_names = []
    final_sectors = []
    for col in resid_final.columns:
        norm = normalize_stock_code(col)
        final_names.append(stock_names.get(norm, "Unknown"))
        final_sectors.append(combined_sector_map.get(norm, "Unknown"))

    # 4.5 군집화 
    corr = np.nan_to_num(resid_final.corr().to_numpy(), nan=0.0)
    np.fill_diagonal(corr, 1.0)
    dist = np.sqrt(np.maximum(0.0, 2.0 * (1.0 - corr)))
    np.fill_diagonal(dist, 0.0)

    linkage_matrix = linkage(squareform(dist, checks=False), method=CLUSTERING_METHOD)
    cluster_labels = fcluster(linkage_matrix, t=N_CLUSTERS, criterion="maxclust") - 1

    results_df = pd.DataFrame({
        "Cluster": cluster_labels, "Name": final_names, "Sector": final_sectors
    })

    print(f"[INFO] Clustering Complete. {len(results_df)} KOSPI stocks clustered.")
    return resid_final, results_df, linkage_matrix

# 5. 리포팅 및 시각화 

def analyze_and_plot(resid_final, results_df, linkage_matrix):
    
    print("\n" + "="*60)
    print(f"[Cluster Report] Inspecting all {N_CLUSTERS} clusters")
    print("="*60)
    
   
    cluster_counts = results_df['Cluster'].value_counts().sort_values(ascending=False)
    
    for cid in cluster_counts.index:
        group = results_df[results_df['Cluster'] == cid]
        count = len(group)
        
        top_sector = group['Sector'].value_counts().idxmax()
        
        print(f"\n▶ Cluster {cid} (Size: {count}) | Dominant Sector: {top_sector}")
        print("-" * 60)
        
        names = group['Name'].tolist()
        for i in range(0, len(names), 5):
            print(", ".join(names[i:i+5]))
        print("-" * 60)

    # [2] 시각화 (t-SNE)

    plt.rcParams['axes.unicode_minus'] = False 
    
    print("\n[Visualization] Generating t-SNE Map...")
    resid_filled = resid_final.T.fillna(0.0)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    tsne_res = tsne.fit_transform(resid_filled)
    
    viz_df = results_df.copy()
    viz_df['x'], viz_df['y'] = tsne_res[:, 0], tsne_res[:, 1]

    # KOSPI 
    korean_to_english = {
        # KOSPI Top Caps
        "삼성전자": "Samsung Elec", "SK하이닉스": "SK Hynix", "LG에너지솔루션": "LG Energy",
        "삼성바이오로직스": "Samsung Bio", "현대차": "Hyundai Motor", "기아": "Kia",
        "POSCO홀딩스": "POSCO Hldgs", "NAVER": "NAVER", "카카오": "Kakao",
        "삼성SDI": "Samsung SDI", "LG화학": "LG Chem", "KB금융": "KB Financial",
        "신한지주": "Shinhan Fin", "셀트리온": "Celltrion",
        # SG Scandal (KOSPI Only)
        "삼천리": "Samchully", "서울가스": "Seoul City Gas", "대성홀딩스": "Daesung Hldgs",
        "세방": "Sebang", 
        # Others
        "한국금융지주": "Korea Inv Hldgs"
    }

    plt.figure(figsize=(14, 10))
    sns.scatterplot(data=viz_df, x='x', y='y', hue='Cluster', palette='tab20', legend='full', s=50, alpha=0.7)
    
    texts = []
    for k_name, e_name in korean_to_english.items():
        if k_name in viz_df['Name'].values:
            row = viz_df[viz_df['Name'] == k_name].iloc[0]
            t = plt.text(row['x'], row['y'], e_name, fontsize=9, fontweight='bold', color='black')
            texts.append(t)
            
    
    print("[INFO] Adjusting labels...")
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    plt.title("t-SNE Visualization of KOSPI Clusters", fontsize=14)
    
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, ncol=2, fontsize='small')
    plt.tight_layout()
    plt.show()

# 6. 실행

if __name__ == "__main__":
    resid, res_df, link_mat = run_analysis()
    if resid is not None:
        analyze_and_plot(resid, res_df, link_mat)
