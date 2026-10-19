# Monte Carlo pricing of a European put under Black-Scholes-Merton,
# with a convergence study against the exact closed-form price.
import time
import numpy as np
import pandas as pd
from scipy.stats import norm

# Parameters from the problem
S0, K, T, r, q, sigma = 100.0, 100.0, 0.5, 0.04, 0.02, 0.2


# Exact Black-Scholes-Merton price of a European put
def bs_put(S0, K, T, r, q, sigma):
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp(-q * T) * norm.cdf(-d1)


# Monte Carlo estimate of the put price and its standard error
def mc_put(S0, K, T, r, q, sigma, N, rng):
    Z = rng.standard_normal(N)
    ST = S0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    disc_payoff = np.exp(-r * T) * np.maximum(K - ST, 0.0)
    price = disc_payoff.mean()
    se = disc_payoff.std(ddof=1) / np.sqrt(N)
    return price, se


if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)
    exact = bs_put(S0, K, T, r, q, sigma)
    print(f"Black-Scholes exact put price: {exact:.6f}\n")

    # Run the simulation for each N and collect one row of results per run
    rows = []
    for N in [10**k for k in range(2, 8)] + [2 * 10**7]:
        t0 = time.perf_counter()
        price, se = mc_put(S0, K, T, r, q, sigma, N, rng)
        elapsed = time.perf_counter() - t0
        rows.append({
            "N": N,
            "MC price": price,
            "Std error": se,
            "CI lower": price - 1.96 * se,
            "CI upper": price + 1.96 * se,
            "Abs error": abs(price - exact),
            "Time (s)": elapsed,
            "Cent accurate": 1.96 * se < 0.005,  # 95% CI half-width under half a cent
        })

    results = pd.DataFrame(rows)
    pd.set_option("display.width", 120)
    print(results.to_string(index=False, formatters={
        "N": "{:,}".format,
        "MC price": "{:.4f}".format,
        "Std error": "{:.5f}".format,
        "CI lower": "{:.4f}".format,
        "CI upper": "{:.4f}".format,
        "Abs error": "{:.5f}".format,
        "Time (s)": "{:.4f}".format,
    }))

    # Sample size needed for a 95% CI half-width of half a cent
    _, se_pilot = mc_put(S0, K, T, r, q, sigma, 10**5, rng)
    payoff_sd = se_pilot * np.sqrt(10**5)
    N_needed = int(np.ceil((1.96 * payoff_sd / 0.005) ** 2))
    print(f"\nPayoff std dev ~ {payoff_sd:.3f}; N needed for cent accuracy ~ {N_needed:,}")