# var_cvar_backtest_stresstest.py
# ============================================================
# VaR & CVaR: Historical, Parametric, Monte Carlo
# Backtesting: Kupiec POF, Christoffersen Independence/Conditional Coverage
# Stress Testing: Historical & Hypothetical scenarios
# ============================================================

import numpy as np
import pandas as pd
from scipy.stats import norm, chi2

# ---------------------------
# CONFIG
# ---------------------------
CSV_PATH = "prices.csv"    # CSV must have Date column + one column per ticker (Adj Close)
TICKERS  = ["AAPL", "MSFT", "GOOGL"]   # match CSV columns
WEIGHTS  = np.array([0.4, 0.3, 0.3])   # must sum to 1
PORTFOLIO_VALUE = 50_000_000           # ₹50M (used for monetary VaR/CVaR reporting)
CONF_LEVELS = [0.95, 0.99]             # confidence levels to compute
ROLLING_WINDOW = 1000                  # days used to estimate rolling VaR
INCLUDE_MONTE_CARLO_BACKTEST = False   # set True to include MC in rolling backtest (slow)
MC_SIMS = 20000                        # Monte Carlo simulations per forecast (if used)
RANDOM_SEED = 42

# Historical stress scenarios (date ranges inclusive)
HISTORICAL_SCENARIOS = [
    ("2008 Lehman Week", "2008-09-15", "2008-09-19"),
    ("COVID Crash Mar 2020", "2020-03-09", "2020-03-16")
]

# Hypothetical scenarios: asset -> shock (fractional, e.g., -0.2 for -20%)
HYPOTHETICAL_SCENARIOS = {
    "Equity Crash": { "AAPL": -0.20, "MSFT": -0.15, "GOOGL": -0.18 },
    "Interest Rate Shock (proxy)": { "AAPL": -0.05, "MSFT": -0.07, "GOOGL": -0.04 }
}

# ---------------------------
# UTILITIES: Load & prepare data
# ---------------------------
def load_prices(csv_path, tickers):
    df = pd.read_csv(csv_path, parse_dates=["Date"]).set_index("Date").sort_index()
    missing = [t for t in tickers if t not in df.columns]
    if missing:
        raise ValueError(f"Missing tickers in CSV: {missing}")
    return df[tickers].dropna(how="any")

prices = load_prices(CSV_PATH, TICKERS)
returns = prices.pct_change().dropna()
if len(WEIGHTS) != returns.shape[1]:
    raise ValueError("WEIGHTS length must match number of tickers/columns.")

# portfolio daily returns series
port_ret = pd.Series(returns.values @ WEIGHTS, index=returns.index, name="portfolio_return")

# ---------------------------
# VaR & CVaR Functions
# ---------------------------

# Historical (empirical) VaR & CVaR
def var_historical(sample_returns, alpha):
    q = np.quantile(sample_returns, 1 - alpha)
    return -q

def cvar_historical(sample_returns, alpha):
    q = np.quantile(sample_returns, 1 - alpha)
    tail = sample_returns[sample_returns <= q]
    if len(tail) == 0:
        return -q
    return -tail.mean()

# Parametric (normal) VaR & CVaR
def var_parametric(sample_returns, alpha):
    mu = np.mean(sample_returns)
    sigma = np.std(sample_returns, ddof=1)
    z = norm.ppf(1 - alpha)   # lower-tail z (e.g., alpha=0.99 -> z ~ -2.33)
    q = mu + sigma * z
    return -q

def cvar_parametric(sample_returns, alpha):
    mu = np.mean(sample_returns)
    sigma = np.std(sample_returns, ddof=1)
    z = norm.ppf(1 - alpha)
    # E[X | X <= q] = mu - sigma * pdf(z) / (1 - alpha)
    denom = (1 - alpha)
    if denom <= 0:
        return var_parametric(sample_returns, alpha)
    conditional_mean = mu - sigma * norm.pdf(z) / denom
    return -conditional_mean

# Monte Carlo VaR & CVaR (simulate returns ~ Normal using sample mu & sigma)
def var_mc_normal(sample_returns, alpha, sims=10000, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    mu = np.mean(sample_returns)
    sigma = np.std(sample_returns, ddof=1)
    sims_r = rng.normal(mu, sigma, sims)
    q = np.quantile(sims_r, 1 - alpha)
    return -q

def cvar_mc_normal(sample_returns, alpha, sims=10000, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    mu = np.mean(sample_returns)
    sigma = np.std(sample_returns, ddof=1)
    sims_r = rng.normal(mu, sigma, sims)
    q = np.quantile(sims_r, 1 - alpha)
    tail = sims_r[sims_r <= q]
    if len(tail) == 0:
        return -q
    return -tail.mean()

# ---------------------------
# Rolling forecast generator
# ---------------------------
def rolling_forecasts(r_series, method, alpha, window, mc_sims=10000, rng=None):
    """
    method: 'hist', 'param', 'mc'
    returns: pd.Series of rolling VaR forecasts aligned with r_series index (NaN before window)
    """
    r = r_series.values
    idx = r_series.index
    out = np.full_like(r, np.nan, dtype=float)
    for t in range(window, len(r)):
        sample = r[t - window:t]
        if method == "hist":
            out[t] = var_historical(sample, alpha)
        elif method == "param":
            out[t] = var_parametric(sample, alpha)
        elif method == "mc":
            out[t] = var_mc_normal(sample, alpha, sims=mc_sims, rng=rng)
        else:
            raise ValueError("Unknown method")
    return pd.Series(out, index=idx)

# Also rolling CVaR forecasts (optional; compute on same window sample)
def rolling_cvar_forecasts(r_series, method, alpha, window, mc_sims=10000, rng=None):
    r = r_series.values
    idx = r_series.index
    out = np.full_like(r, np.nan, dtype=float)
    for t in range(window, len(r)):
        sample = r[t - window:t]
        if method == "hist":
            out[t] = cvar_historical(sample, alpha)
        elif method == "param":
            out[t] = cvar_parametric(sample, alpha)
        elif method == "mc":
            out[t] = cvar_mc_normal(sample, alpha, sims=mc_sims, rng=rng)
        else:
            raise ValueError("Unknown method")
    return pd.Series(out, index=idx)

# ---------------------------
# Backtesting functions (Kupiec & Christoffersen)
# ---------------------------
def kupiec_pof_test(exceedances, alpha):
    x = int(exceedances.sum())
    T = int(exceedances.size)
    p = 1 - alpha
    pi_hat = x / T if T > 0 else 0.0
    def safe_log(z): return np.log(z) if z > 0 else -1e9
    ll_null = (T - x) * safe_log(1 - p) + x * safe_log(p)
    ll_alt = (T - x) * safe_log(1 - pi_hat) + x * safe_log(pi_hat) if 0 < pi_hat < 1 else -1e9
    LR = -2 * (ll_null - ll_alt)
    pvalue = 1 - chi2.cdf(LR, df=1)
    return {"obs": T, "breaches": x, "rate": pi_hat, "LR": LR, "pvalue": pvalue}

def christoffersen_independence_test(exceedances):
    x = exceedances.astype(int).values
    n00 = n01 = n10 = n11 = 0
    for i in range(1, len(x)):
        prev, curr = x[i-1], x[i]
        if prev == 0 and curr == 0: n00 += 1
        elif prev == 0 and curr == 1: n01 += 1
        elif prev == 1 and curr == 0: n10 += 1
        elif prev == 1 and curr == 1: n11 += 1
    n0 = n00 + n01
    n1 = n10 + n11
    pi = (n01 + n11) / (n0 + n1) if (n0 + n1) > 0 else 0
    pi0 = n01 / n0 if n0 > 0 else 0
    pi1 = n11 / n1 if n1 > 0 else 0
    def safe_log(z): return np.log(z) if z > 0 else -1e9
    ll_ind = (n00 * safe_log(1 - pi0) + n01 * safe_log(pi0) +
              n10 * safe_log(1 - pi1) + n11 * safe_log(pi1))
    ll_unc = ((n00 + n10) * safe_log(1 - pi) + (n01 + n11) * safe_log(pi))
    LR = -2 * (ll_unc - ll_ind)
    pvalue = 1 - chi2.cdf(LR, df=1)
    return {"n00": n00, "n01": n01, "n10": n10, "n11": n11, "LR": LR, "pvalue": pvalue}

def conditional_coverage_test(exceedances, alpha):
    pof = kupiec_pof_test(exceedances, alpha)
    ind = christoffersen_independence_test(exceedances)
    LR_cc = pof["LR"] + ind["LR"]
    pval = 1 - chi2.cdf(LR_cc, df=2)
    return {"LR": LR_cc, "pvalue": pval, "pof": pof, "ind": ind}

# ---------------------------
# Stress testing functions
# ---------------------------
def historical_stress_test(returns_df, weights, start_date, end_date):
    # compute cumulative portfolio return over the scenario window
    scenario = returns_df.loc[start_date:end_date]
    if scenario.empty:
        raise ValueError(f"No data for scenario {start_date} to {end_date}")
    port_returns = scenario.values @ weights
    total_return = (1 + port_returns).prod() - 1
    return total_return

def hypothetical_stress_test(prices_df, weights, shocks_dict):
    # apply shock to last available prices, compute % change in portfolio value
    last_prices = prices_df.iloc[-1].copy()
    shocked = last_prices.copy()
    for asset, shock in shocks_dict.items():
        if asset not in shocked.index:
            # try match by ticker names; if missing, skip with warning
            continue
        shocked[asset] *= (1 + shock)
    old_value = (last_prices.values * weights).sum()
    new_value = (shocked.values * weights).sum()
    pct_change = (new_value - old_value) / old_value
    return pct_change

# ---------------------------
# Run point estimates (last window) and backtests
# ---------------------------
def point_estimates_and_backtest(r_series, conf_levels, window, include_mc=False, mc_sims=20000):
    rng = np.random.default_rng(RANDOM_SEED)
    # Point estimates on last window
    tail = r_series[-window:]
    pe_rows = []
    for cl in conf_levels:
        pe_rows.append(["Parametric", cl,
                        var_parametric(tail, cl) * PORTFOLIO_VALUE,
                        cvar_parametric(tail, cl) * PORTFOLIO_VALUE])
        pe_rows.append(["Historical", cl,
                        var_historical(tail, cl) * PORTFOLIO_VALUE,
                        cvar_historical(tail, cl) * PORTFOLIO_VALUE])
        if include_mc:
            pe_rows.append(["MonteCarlo", cl,
                            var_mc_normal(tail, cl, sims=mc_sims, rng=rng) * PORTFOLIO_VALUE,
                            cvar_mc_normal(tail, cl, sims=mc_sims, rng=rng) * PORTFOLIO_VALUE])
    pe_df = pd.DataFrame(pe_rows, columns=["Method", "CL", "VaR(₹)", "CVaR(₹)"])

    # Rolling forecasts & backtests
    backtest_rows = []
    for cl in conf_levels:
        # generate rolling VaR forecasts per method
        var_hist = rolling_forecasts(r_series, "hist", cl, window)
        var_par  = rolling_forecasts(r_series, "param", cl, window)
        forecasts = {"Historical": var_hist, "Parametric": var_par}
        if include_mc:
            # Monte Carlo rolling forecasts (stochastic) - can be slow
            var_mc  = rolling_forecasts(r_series, "mc", cl, window, mc_sims=mc_sims, rng=rng)
            forecasts["MonteCarlo"] = var_mc

        # Align realized returns over forecasted (drop NaNs)
        # Note: var_series has NaN until index position window
        for name, var_series in forecasts.items():
            var_ser = var_series.dropna()
            realized = r_series[var_ser.index]   # same dates as forecasts
            # breach if realized return < -VaR (since VaR is positive loss threshold)
            breaches = (realized < -var_ser)
            pof = kupiec_pof_test(breaches, cl)
            ind = christoffersen_independence_test(breaches)
            cc  = conditional_coverage_test(breaches, cl)
            backtest_rows.append({
                "Method": name,
                "CL": int(cl*100),
                "Obs": pof["obs"],
                "Breaches": pof["breaches"],
                "BreachRate(%)": round(100 * pof["rate"], 3),
                "POF_p": round(pof["pvalue"], 4),
                "IND_p": round(ind["pvalue"], 4),
                "CC_p": round(cc["pvalue"], 4)
            })
    bt_df = pd.DataFrame(backtest_rows)
    return pe_df, bt_df

# ---------------------------
# Main execution
# ---------------------------
if __name__ == "__main__":
    print(f"Data loaded: {prices.shape[0]} rows for {len(TICKERS)} tickers. Rolling window = {ROLLING_WINDOW} days.\n")

    # 1) Point estimates and backtests
    pe_df, bt_df = point_estimates_and_backtest(port_ret, CONF_LEVELS, ROLLING_WINDOW,
                                                include_mc=INCLUDE_MONTE_CARLO_BACKTEST, mc_sims=MC_SIMS)
    print("=== Point Estimates (last window) - VaR & CVaR (₹) ===")
    print(pe_df.to_string(index=False))

    print("\n=== Backtesting Summary (Kupiec / Christoffersen) ===")
    print(bt_df.to_string(index=False))

    # summary lines
    for cl in CONF_LEVELS:
        sub = bt_df[bt_df.CL == int(cl*100)]
        print(f"\n-- {int(cl*100)}% VaR Backtest Summary --")
        for _, row in sub.iterrows():
            theo = (1 - cl) * 100
            print(f"{row['Method']}: BreachRate={row['BreachRate(%)']}% (theoretical ≈ {theo:.2f}%) | "
                  f"POF_p={row['POF_p']} | IND_p={row['IND_p']} | CC_p={row['CC_p']}")

    # 2) Stress testing (historical + hypothetical)
    print("\n=== Stress Testing Results ===")
    # Historical scenario replay
    for name, start, end in HISTORICAL_SCENARIOS:
        try:
            loss = historical_stress_test(returns, WEIGHTS, start, end)
            print(f"{name} ({start} to {end}): cumulative portfolio return = {loss:.2%} (i.e., {loss*100:.2f}% change)")
        except Exception as e:
            print(f"{name}: ERROR - {e}")

    # Hypothetical shocks
    for name, shocks in HYPOTHETICAL_SCENARIOS.items():
        # ensure shocks map to tickers; skip missing assets
        valid_shocks = {k: v for k, v in shocks.items() if k in prices.columns}
        if not valid_shocks:
            print(f"{name}: No matching tickers for shocks; skipped.")
            continue
        loss = hypothetical_stress_test(prices, WEIGHTS, valid_shocks)
        print(f"{name}: portfolio change = {loss:.2%}")

    print("\nAll done. Adjust CONFIG at the top to change assets, weights, window, or enable Monte Carlo backtests.")
