# Monte Carlo pricing of an arithmetic-average Asian call under Black-Scholes-Merton,
# with a convergence study. There is no closed form for the arithmetic Asian, so the
# simulator is validated on the geometric Asian call, which does have one.
import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters from the problem
S0, K, T, r, q, sigma, m = 100.0, 100.0, 1.0, 0.10, 0.0, 0.20, 50
CENT = 0.005    # target 95% CI half-width: half a cent
CHUNK = 10**5   # paths simulated at once; 10^5 x 50 doubles is 40 MB per array
PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")


# Exact price of the discretely monitored geometric Asian call. The log of the
# geometric average of S_t1..S_tm is normal with the mean and variance below.
def geometric_asian_call(S0, K, T, r, q, sigma, m):
    mu = np.log(S0) + (r - q - 0.5 * sigma**2) * T * (m + 1) / (2 * m)
    v = sigma**2 * T * (m + 1) * (2 * m + 1) / (6 * m**2)
    d2 = (mu - np.log(K)) / np.sqrt(v)
    d1 = d2 + np.sqrt(v)
    return np.exp(-r * T) * (np.exp(mu + 0.5 * v) * norm.cdf(d1) - K * norm.cdf(d2))


# Discounted arithmetic and geometric Asian call payoffs for n simulated paths.
# Works in place on one (n, m) array to avoid temporary copies.
def chunk_payoffs(n, rng):
    dt = T / m
    X = rng.standard_normal((n, m))
    X *= sigma * np.sqrt(dt)
    X += (r - q - 0.5 * sigma**2) * dt
    np.cumsum(X, axis=1, out=X)
    X += np.log(S0)                     # X[:, i] = log S_{t_(i+1)}
    geo_avg = np.exp(X.mean(axis=1))
    np.exp(X, out=X)
    arith_avg = X.mean(axis=1)          # average over t_1..t_m, excluding S0
    disc = np.exp(-r * T)
    return disc * np.maximum(arith_avg - K, 0.0), disc * np.maximum(geo_avg - K, 0.0)


# Monte Carlo estimates for N paths, simulated in chunks. Running sums of Y and
# Y^2 give the same mean and standard error as one array of all N payoffs.
def mc_asian(N, rng, chunk=CHUNK):
    sums = np.zeros(4)  # sum Y_arith, sum Y_arith^2, sum Y_geo, sum Y_geo^2
    done = 0
    while done < N:
        n = min(chunk, N - done)
        Ya, Yg = chunk_payoffs(n, rng)
        sums += [Ya.sum(), Ya @ Ya, Yg.sum(), Yg @ Yg]
        done += n

    def mean_se(s, s2):
        mean = s / N
        return mean, np.sqrt((s2 - N * mean**2) / (N - 1) / N)

    return mean_se(sums[0], sums[1]) + mean_se(sums[2], sums[3])


# Best-of-`repeats` wall time; large runs are timed once since they take seconds
def timed(N, rng):
    repeats = 3 if N <= 10**6 else 1
    best = np.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = mc_asian(N, rng)
        best = min(best, time.perf_counter() - t0)
    return result, best


def plot_results(df):
    os.makedirs(PLOT_DIR, exist_ok=True)

    # CI half-width against N on log-log axes, with a N^(-1/2) guide
    fig, ax = plt.subplots(figsize=(7, 4.5))
    Ns = df["N"].to_numpy(dtype=float)
    ax.loglog(Ns, 1.96 * df["Std error"], "o-", label="95% CI half-width")
    ax.loglog(Ns, 1.96 * df["Std error"].iloc[-1] * np.sqrt(Ns[-1] / Ns), ":", color="gray", label=r"$\propto N^{-1/2}$")
    ax.axhline(CENT, color="red", lw=1, label="Half a cent")
    ax.set_xlabel("Number of paths N")
    ax.set_ylabel("95% CI half-width ($)")
    ax.set_title("Convergence of Monte Carlo Asian call price")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "asian_convergence_error.png"), dpi=150)

    # Price estimates with 95% CIs against N, around the largest-N estimate
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(Ns, df["MC price"], yerr=1.96 * df["Std error"], fmt="o", capsize=3, label="MC estimate")
    ax.axhline(df["MC price"].iloc[-1], color="black", lw=1,
               label=f"Largest-N estimate {df['MC price'].iloc[-1]:.4f}")
    ax.set_xscale("log")
    ax.set_xlabel("Number of paths N")
    ax.set_ylabel("Asian call price ($)")
    ax.set_title("Monte Carlo Asian call estimates with 95% confidence intervals")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "asian_convergence_price.png"), dpi=150)
    plt.close("all")


if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)
    geo_exact = geometric_asian_call(S0, K, T, r, q, sigma, m)

    # Pilot run estimates the payoff standard deviation, which gives the number
    # of paths needed for a 95% CI half-width of half a cent. The 2% margin covers
    # sampling noise in the pilot's own standard deviation estimate.
    pilot = 10**6
    _, se_pilot, _, _ = mc_asian(pilot, rng)
    payoff_sd = se_pilot * np.sqrt(pilot)
    N_needed = int(np.ceil(1.02 * (1.96 * payoff_sd / CENT) ** 2))

    rows, geo_rows = [], []
    for N in sorted([10**k for k in range(2, 8)] + [N_needed]):
        (price, se, geo_price, geo_se), elapsed = timed(N, rng)
        rows.append({
            "N": N,
            "MC price": price,
            "Std error": se,
            "CI lower": price - 1.96 * se,
            "CI upper": price + 1.96 * se,
            "Time (s)": elapsed,
            "Cent accurate": 1.96 * se < CENT,
        })
        geo_rows.append({
            "N": N,
            "MC geo price": geo_price,
            "Std error": geo_se,
            "Abs error": abs(geo_price - geo_exact),
            "Exact in CI": abs(geo_price - geo_exact) <= 1.96 * geo_se,
        })

    results, geo_results = pd.DataFrame(rows), pd.DataFrame(geo_rows)
    pd.set_option("display.width", 120)
    print("Arithmetic Asian call\n" + results.to_string(index=False, formatters={
        "N": "{:,}".format,
        "MC price": "{:.4f}".format,
        "Std error": "{:.5f}".format,
        "CI lower": "{:.4f}".format,
        "CI upper": "{:.4f}".format,
        "Time (s)": "{:.2e}".format,
    }))
    print(f"\nPayoff std dev ~ {payoff_sd:.3f}; N needed for cent accuracy ~ {N_needed:,}")

    print(f"\nValidation: geometric Asian call, exact price {geo_exact:.6f}\n" + geo_results.to_string(
        index=False, formatters={
            "N": "{:,}".format,
            "MC geo price": "{:.4f}".format,
            "Std error": "{:.5f}".format,
            "Abs error": "{:.5f}".format,
        }))

    plot_results(results)
    print(f"\nPlots saved to {PLOT_DIR}")
