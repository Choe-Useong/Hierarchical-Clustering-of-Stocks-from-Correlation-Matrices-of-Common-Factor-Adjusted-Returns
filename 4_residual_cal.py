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

def capm_resid_alpha_beta_resid0(R, rm_w, min_obs=60):
    # 날짜 교집합
    common = R.columns.intersection(rm_w.index)
    R = R.loc[:, common]
    m = rm_w.loc[common]

    # 결과: 원래 종목/날짜 유지
    resid_df = pd.DataFrame(0.0, index=R.index, columns=common)  # 기본 0 (결측 잔차=0)
    param_df = pd.DataFrame(np.nan, index=R.index, columns=["alpha", "beta", "nobs"])

    m_all = m.values
    ok_m = np.isfinite(m_all)

    for sym in R.index:
        x = R.loc[sym].values
        ok = np.isfinite(x) & ok_m
        n = int(ok.sum())
        if n < min_obs:
            continue

        xi = x[ok]
        mi = m_all[ok]

        m_mean = mi.mean()
        m_dm = mi - m_mean
        var_m = (m_dm @ m_dm) / n
        if var_m == 0 or not np.isfinite(var_m):
            continue

        x_mean = xi.mean()
        cov = ((xi - x_mean) @ m_dm) / n
        beta = cov / var_m
        alpha = x_mean - beta * m_mean

        # 관측된 날의 잔차만 계산해서 채움 (결측날은 이미 0)
        r = xi - alpha - beta * mi
        resid_df.loc[sym, np.array(common)[ok]] = r
        param_df.loc[sym] = [alpha, beta, n]

    return resid_df, param_df

# 사용
resid, param = capm_resid_alpha_beta_resid0(ret, rm, min_obs=60)


# 4) 저장
os.makedirs(OUT_DIR, exist_ok=True)
resid.to_parquet(os.path.join(OUT_DIR, OUT_RESID), engine="pyarrow")
param.to_parquet(os.path.join(OUT_DIR, OUT_PARAM), engine="pyarrow")

print("resid:", resid.shape, "NaN:", int(resid.isna().sum().sum()))
print("param:", param.shape)




# === 원수익률 결측만 CAPM으로 채움 ===
common = ret.columns.intersection(rm.index)
rm_c = rm.loc[common]

alpha = param["alpha"].values[:, None]   # (N,1)
beta  = param["beta"].values[:, None]    # (N,1)

Rhat = alpha + beta * rm_c.values[None, :]  # (N,T)
Rhat_df = pd.DataFrame(Rhat, index=ret.index, columns=common)

ret_fill = ret.loc[:, common].copy()
ret_fill = ret_fill.combine_first(Rhat_df)  # ret의 NaN만 Rhat로 채움

print("ret_fill:", ret_fill.shape, "NaN:", int(ret_fill.isna().sum().sum()))

# 저장
ret_fill.to_parquet(os.path.join(OUT_DIR, "ret_capmfill_fullperiod.parquet"), engine="pyarrow")
