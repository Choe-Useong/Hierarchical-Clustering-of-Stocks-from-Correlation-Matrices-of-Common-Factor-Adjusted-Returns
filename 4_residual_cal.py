import os
import numpy as np
import pandas as pd

RET_PATH  = r"ret_kospi_topPct_complete_period.parquet"
FF_PATH   = r"C:\Users\working\Desktop\주식클러스터\items_parquet\ff_sheet1.parquet"

OUT_DIR   = r"resid_fullperiod"
OUT_RESID = r"resid_ff3_fullperiod.parquet"
OUT_PARAM = r"ff3_alpha_beta_fullperiod.parquet"

# 1) ret
ret = pd.read_parquet(RET_PATH)
ret.columns = pd.to_datetime(ret.columns, errors="coerce")
ret = ret.sort_index(axis=1)

# 2) FF3 요인(F) 생성: (모두 "지수/레벨"이라고 가정 -> 로그차분)
ff = pd.read_parquet(FF_PATH)
ff["date"] = pd.to_datetime(ff["Symbol Name"], errors="coerce")
ff = ff.dropna(subset=["date"]).set_index("date").sort_index()

rm  = np.log(pd.to_numeric(ff["코스피"], errors="coerce")).diff()
smb = np.log(pd.to_numeric(ff["Size & Book Value(2X3) SMB"], errors="coerce")).diff()
hml = np.log(pd.to_numeric(ff["Size & Book Value(2X3) HML"], errors="coerce")).diff()

F = pd.concat([rm.rename("rm"), smb.rename("SMB"), hml.rename("HML")], axis=1)

# ===== rm에서 Rf 빼기 (rm -> Rm-Rf) =====
RF_PATH  = "CD91.csv"
DATE_COL = "date"     
VAL_COL  = "cd91_y"   

rf = pd.read_csv("CD91.csv").drop(columns=['원자료.1'])

# 컬럼 정리 + 날짜/숫자 변환
rf = rf.rename(columns={"변환": "date", "원자료": "cd91_y"})  # 필요 시 컬럼명 확인해서 맞춰
rf["date"] = pd.to_datetime(rf["date"], errors="coerce")
rf["cd91_y"] = pd.to_numeric(rf["cd91_y"], errors="coerce")

rf = rf.dropna(subset=["date", "cd91_y"]).sort_values("date").set_index("date")

# 연율(%) -> 일별 로그수익률로 환산
rf_daily_log = np.log1p(rf["cd91_y"] / 100.0) / 252.0
rf_daily_log.name = "rf"

F["rm"] = F["rm"] - rf_daily_log.reindex(F.index).ffill()

# 3) 날짜 교집합 정합
common = ret.columns.intersection(F.index)
ret = ret.loc[:, common]
F   = F.loc[common]
F   = F.dropna()
ret = ret.loc[:, F.index]

def ff3_resid_alpha_beta_resid0(R, F_w, min_obs=60):
    # 날짜 교집합
    common = R.columns.intersection(F_w.index)
    R = R.loc[:, common]
    F = F_w.loc[common]  # (T, K)

    # 결과: 원래 종목/날짜 유지
    resid_df = pd.DataFrame(np.nan, index=R.index, columns=common)  
    K = F.shape[1]
    beta_cols = ["alpha"] + [f"beta_{c}" for c in F.columns] + ["nobs"]
    param_df = pd.DataFrame(np.nan, index=R.index, columns=beta_cols)

    Farr = F.values
    okF_all = np.all(np.isfinite(Farr), axis=1)  # (T,)

    for sym in R.index:
        x = R.loc[sym].values  # (T,)
        ok = np.isfinite(x) & okF_all
        n = int(ok.sum())
        if n < min_obs:
            continue

        y = x[ok]                  # (n,)
        X = Farr[ok, :]             # (n, K)
        X = np.column_stack([np.ones(n), X])  # (n, K+1)

        # OLS (안정적으로 lstsq)
        b, *_ = np.linalg.lstsq(X, y, rcond=None)  # (K+1,)
        alpha = float(b[0])
        betas = b[1:].astype(float)

        # 잔차(관측된 날만 채움)
        yhat = X @ b
        r = y - yhat
        resid_df.loc[sym, np.array(common)[ok]] = r

        param_df.loc[sym, :] = [alpha, *betas.tolist(), n]

    return resid_df, param_df

# 사용
resid, param = ff3_resid_alpha_beta_resid0(ret, F, min_obs=252)

# 4) 저장
os.makedirs(OUT_DIR, exist_ok=True)
resid.to_parquet(os.path.join(OUT_DIR, OUT_RESID), engine="pyarrow")
param.to_parquet(os.path.join(OUT_DIR, OUT_PARAM), engine="pyarrow")

print("resid:", resid.shape, "NaN:", int(resid.isna().sum().sum()))
print("param:", param.shape)
