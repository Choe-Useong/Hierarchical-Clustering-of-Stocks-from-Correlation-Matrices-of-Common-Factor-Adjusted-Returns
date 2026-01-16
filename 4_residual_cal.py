import os
import numpy as np
import pandas as pd
import yfinance as yf

RET_PATH = r"ret_kospi_topPct_complete_period.parquet"
OUT_DIR  = r"resid_fullperiod"
OUT_RESID = r"resid_capm_fullperiod.parquet"
OUT_PARAM = r"capm_alpha_beta_fullperiod.parquet"

KOSPI_TICKER = "^KS11"

# 1) ret
ret = pd.read_parquet(RET_PATH)
ret.columns = pd.to_datetime(ret.columns, errors="coerce")
ret = ret.sort_index(axis=1)

# 2) rm (KOSPI 지수) + 날짜 교집합 정합
start = ret.columns.min().strftime("%Y-%m-%d")
end   = (ret.columns.max() + pd.Timedelta(days=3)).strftime("%Y-%m-%d")


px = yf.download(KOSPI_TICKER, start=start, end=end, auto_adjust=True, progress=False)[["Close"]]
px.index = pd.to_datetime(px.index)
px = px.droplevel(0, axis=1).squeeze()

rm = np.log(px).diff()
rm.name = "rm"

common = ret.columns.intersection(rm.index)
ret = ret.loc[:, common]
rm  = rm.loc[common]
rm = rm.dropna()
ret = ret.loc[:, rm.index]

# 3) CAPM 잔차 + alpha/beta (벡터화)
def capm_resid_alpha_beta(R, rm_w):
    ok_t = rm_w.notna().values
    R = R.iloc[:, ok_t]
    rm_w = rm_w.iloc[ok_t]

    keep = R.notna().all(axis=1)
    R = R.loc[keep]
    if R.shape[0] == 0:
        return R, pd.DataFrame(columns=["alpha","beta"])

    m = rm_w.values
    m_mean = m.mean()
    m_dm = m - m_mean
    var_m = (m_dm @ m_dm) / len(m_dm)

    X = R.values
    X_mean = X.mean(axis=1, keepdims=True)
    X_dm = X - X_mean

    cov = (X_dm @ m_dm) / len(m_dm)
    beta = cov / var_m
    alpha = X_mean[:, 0] - beta * m_mean

    resid = X - alpha[:, None] - beta[:, None] * m[None, :]

    resid_df = pd.DataFrame(resid, index=R.index, columns=R.columns)
    param_df = pd.DataFrame({"alpha": alpha, "beta": beta}, index=R.index)
    return resid_df, param_df

resid, param = capm_resid_alpha_beta(ret, rm)

# 4) 저장
os.makedirs(OUT_DIR, exist_ok=True)
resid.to_parquet(os.path.join(OUT_DIR, OUT_RESID), engine="pyarrow")
param.to_parquet(os.path.join(OUT_DIR, OUT_PARAM), engine="pyarrow")

print("resid:", resid.shape, "NaN:", int(resid.isna().sum().sum()))
print("param:", param.shape)