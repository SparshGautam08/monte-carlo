# Monte Carlo pricing of a European put under Black-Scholes-Merton,
# with a convergence study against the exact closed-form price,
# an antithetic-variates comparison (part 1a) and a timing benchmark.
import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters from the problem
S0, K, T, r, q, sigma = 100.0, 100.0, 0.5, 0.04, 0.02, 0.2
CENT = 0.005  # target 95% CI half-width: half a cent, so the price rounds to the right cent
PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")


# Exact Black-Scholes-Merton price of a European put
def bs_put(S0, K, T, r, q, sigma):
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp(-q * T) * norm.cdf(-d1)


# Discounted put payoffs for an array of standard normals. The constants are
# computed once so the per-sample work is one exp, one subtract and one max.
def disc_payoffs(Z, S0, K, T, r, q, sigma):
    drift = (r - q - 0.5 * sigma**2) * T
    vol = sigma * np.sqrt(T)
    return np.exp(-r * T) * np.maximum(K - S0 * np.exp(drift + vol * Z), 0.0)


# Standard Monte Carlo estimate of the put price and its standard error
def mc_put(S0, K, T, r, q, sigma, N, rng):
    Y = disc_payoffs(rng.standard_normal(N), S0, K, T, r, q, sigma)
    return Y.mean(), Y.std(ddof=1) / np.sqrt(N)


# Antithetic Monte Carlo: N total payoffs from N/2 pairs (Z, -Z). The pair
# averages are i.i.d., so the standard error is computed over the pairs.
def mc_put_antithetic(S0, K, T, r, q, sigma, N, rng):
    Z = rng.standard_normal(N // 2)
    pairs = 0.5 * (disc_payoffs(Z, S0, K, T, r, q, sigma) + disc_payoffs(-Z, S0, K, T, r, q, sigma))
    return pairs.mean(), pairs.std(ddof=1) / np.sqrt(N // 2)


# Plain Python loop, used only to measure the speedup from vectorizing
def mc_put_loop(S0, K, T, r, q, sigma, N, rng):
    drift = (r - q - 0.5 * sigma**2) * T
    vol = sigma * np.sqrt(T)
    disc = np.exp(-r * T)
    total = total_sq = 0.0
    for z in rng.standard_normal(N).tolist():
        y = disc * max(K - S0 * np.exp(drift + vol * z), 0.0)
        total += y
        total_sq += y * y
    mean = total / N
    return mean, np.sqrt((total_sq - N * mean**2) / (N - 1) / N)


# Best-of-`repeats` wall time, so small-N timings are not dominated by noise
def timed(fn, N, rng, repeats=3):
    best = np.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        price, se = fn(S0, K, T, r, q, sigma, N, rng)
        best = min(best, time.perf_counter() - t0)
    return price, se, best


def convergence_table(fn, Ns, exact, rng):
    rows = []
    for N in Ns:
        price, se, elapsed = timed(fn, N, rng)
        rows.append({
            "N": N,
            "MC price": price,
            "Std error": se,
            "CI lower": price - 1.96 * se,
            "CI upper": price + 1.96 * se,
            "Abs error": abs(price - exact),
            "Time (s)": elapsed,
            "Cent accurate": 1.96 * se < CENT,
        })
    return pd.DataFrame(rows)


def print_table(title, df):
    print(f"{title}\n" + df.to_string(index=False, formatters={
        "N": "{:,}".format,
        "MC price": "{:.4f}".format,
        "Std error": "{:.5f}".format,
        "CI lower": "{:.4f}".format,
        "CI upper": "{:.4f}".format,
        "Abs error": "{:.5f}".format,
        "Time (s)": "{:.2e}".format,
    }) + "\n")


def plot_results(std, anti, exact):
    os.makedirs(PLOT_DIR, exist_ok=True)

    # Error and CI half-width against N on log-log axes, with a N^(-1/2) guide
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for df, label, color in [(std, "Standard", "C0"), (anti, "Antithetic", "C1")]:
        ax.loglog(df["N"], 1.96 * df["Std error"], "o-", color=color, label=f"{label}: 95% CI half-width")
        ax.loglog(df["N"], df["Abs error"], "x--", color=color, alpha=0.6, label=f"{label}: abs error")
    Ns = std["N"].to_numpy(dtype=float)
    ax.loglog(Ns, 1.96 * std["Std error"].iloc[0] * np.sqrt(Ns[0] / Ns), ":", color="gray", label=r"$\propto N^{-1/2}$")
    ax.axhline(CENT, color="red", lw=1, label="Half a cent")
    ax.set_xlabel("Sample size N")
    ax.set_ylabel("Error ($)")
    ax.set_title("Convergence of Monte Carlo put price")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "convergence_error.png"), dpi=150)

    # Price estimates with 95% CIs against N, around the exact price
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for df, label, color, shift in [(std, "Standard", "C0", 0.95), (anti, "Antithetic", "C1", 1.05)]:
        ax.errorbar(df["N"] * shift, df["MC price"], yerr=1.96 * df["Std error"],
                    fmt="o", color=color, capsize=3, label=label)
    ax.axhline(exact, color="black", lw=1, label=f"Black-Scholes {exact:.4f}")
    ax.set_xscale("log")
    ax.set_xlabel("Sample size N")
    ax.set_ylabel("Put price ($)")
    ax.set_title("Monte Carlo price estimates with 95% confidence intervals")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "convergence_price.png"), dpi=150)
    plt.close("all")


if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)
    exact = bs_put(S0, K, T, r, q, sigma)
    print(f"Black-Scholes exact put price: {exact:.6f}\n")

    # Pilot runs estimate the payoff standard deviation, which gives the
    # sample size needed for a 95% CI half-width of half a cent
    pilot = 10**5
    _, se_pilot = mc_put(S0, K, T, r, q, sigma, pilot, rng)
    _, se_pilot_anti = mc_put_antithetic(S0, K, T, r, q, sigma, pilot, rng)
    payoff_sd = se_pilot * np.sqrt(pilot)
    pair_sd = se_pilot_anti * np.sqrt(pilot // 2)
    N_needed = int(np.ceil((1.96 * payoff_sd / CENT) ** 2))
    N_needed_anti = 2 * int(np.ceil((1.96 * pair_sd / CENT) ** 2))  # 2 payoffs per pair

    Ns = sorted([10**k for k in range(2, 8)] + [N_needed_anti, N_needed, 2 * 10**7])
    std = convergence_table(mc_put, Ns, exact, rng)
    anti = convergence_table(mc_put_antithetic, Ns, exact, rng)
    pd.set_option("display.width", 120)
    print_table("Standard Monte Carlo", std)
    print_table("Antithetic variates (N = total payoffs, N/2 pairs)", anti)

    # Compare at equal total samples: variance per payoff evaluated
    var_ratio = payoff_sd**2 / (2 * pair_sd**2)
    print(f"Payoff std dev ~ {payoff_sd:.3f}; N needed for cent accuracy ~ {N_needed:,}")
    print(f"Antithetic pair std dev ~ {pair_sd:.3f}; N needed for cent accuracy ~ {N_needed_anti:,}")
    print(f"Antithetic variance reduction at equal N: {var_ratio:.2f}x")
    t_std = std.loc[std["N"] == N_needed, "Time (s)"].iloc[0]
    t_anti = anti.loc[anti["N"] == N_needed_anti, "Time (s)"].iloc[0]
    print(f"Time to cent accuracy: standard {t_std:.3f}s, antithetic {t_anti:.3f}s\n")

    # Vectorization benchmark: same estimator, Python loop vs NumPy arrays
    N_bench = 10**5
    _, _, t_loop = timed(mc_put_loop, N_bench, rng)
    _, _, t_vec = timed(mc_put, N_bench, rng)
    print(f"Vectorization at N = {N_bench:,}: loop {t_loop:.4f}s, NumPy {t_vec:.4f}s "
          f"({t_loop / t_vec:.0f}x faster)")

    plot_results(std, anti, exact)
    print(f"\nPlots saved to {PLOT_DIR}")
