# Monte Carlo pricing of an arithmetic-average Asian call under Black-Scholes-Merton,
# with a convergence study. There is no closed form for the arithmetic Asian, so the
# simulator is validated on the geometric Asian call, which does have one.
# Two variance reduction methods are compared with standard Monte Carlo:
#   2a. control variate, using the geometric Asian call as the control
#   2b. moment matching, rescaling simulated prices to their known means
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
MM_BATCH = 10**4  # paths per moment-matching batch; its bias shrinks like 1/batch size
DISC = np.exp(-r * T)
PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")


# Exact price of the discretely monitored geometric Asian call. The log of the
# geometric average of S_t1..S_tm is normal with the mean and variance below.
def geometric_asian_call(S0, K, T, r, q, sigma, m):
    mu = np.log(S0) + (r - q - 0.5 * sigma**2) * T * (m + 1) / (2 * m)
    v = sigma**2 * T * (m + 1) * (2 * m + 1) / (6 * m**2)
    d2 = (mu - np.log(K)) / np.sqrt(v)
    d1 = d2 + np.sqrt(v)
    return np.exp(-r * T) * (np.exp(mu + 0.5 * v) * norm.cdf(d1) - K * norm.cdf(d2))


GEO_EXACT = geometric_asian_call(S0, K, T, r, q, sigma, m)


# Log prices log S_t1..log S_tm for n paths, built in place in one (n, m) array
def log_paths(n, rng):
    dt = T / m
    X = rng.standard_normal((n, m))
    X *= sigma * np.sqrt(dt)
    X += (r - q - 0.5 * sigma**2) * dt
    np.cumsum(X, axis=1, out=X)
    X += np.log(S0)                     # X[:, i] = log S_{t_(i+1)}
    return X


# Discounted arithmetic and geometric Asian call payoffs for n simulated paths
def chunk_payoffs(n, rng):
    X = log_paths(n, rng)
    geo_avg = np.exp(X.mean(axis=1))
    np.exp(X, out=X)
    arith_avg = X.mean(axis=1)          # average over t_1..t_m, excluding S0
    return DISC * np.maximum(arith_avg - K, 0.0), DISC * np.maximum(geo_avg - K, 0.0)


def mean_se(s, s2, N):
    mean = s / N
    return mean, np.sqrt((s2 - N * mean**2) / (N - 1) / N)


# Standard Monte Carlo for N paths, simulated in chunks. Running sums of Y and
# Y^2 give the same mean and standard error as one array of all N payoffs.
# Also returns the geometric Asian estimate from the same paths for validation.
def mc_asian(N, rng, chunk=CHUNK):
    sums = np.zeros(4)  # sum Y_arith, sum Y_arith^2, sum Y_geo, sum Y_geo^2
    done = 0
    while done < N:
        n = min(chunk, N - done)
        Ya, Yg = chunk_payoffs(n, rng)
        sums += [Ya.sum(), Ya @ Ya, Yg.sum(), Yg @ Yg]
        done += n
    return mean_se(sums[0], sums[1], N) + mean_se(sums[2], sums[3], N)


# Optimal control variate coefficient b = Cov(Y_arith, Y_geo) / Var(Y_geo),
# estimated on a separate pilot so the main estimator stays unbiased
def control_variate_coef(n, rng):
    Ya, Yg = chunk_payoffs(n, rng)
    return np.cov(Ya, Yg)[0, 1] / Yg.var(ddof=1)


# 2a. Control variate: Y_cv = Y_arith - b (Y_geo - E[Y_geo]). The correction has
# mean zero, so Y_cv is unbiased, and its variance is (1 - rho^2) Var(Y_arith).
def mc_control_variate(N, rng, b, chunk=CHUNK):
    s = s2 = 0.0
    done = 0
    while done < N:
        n = min(chunk, N - done)
        Ya, Yg = chunk_payoffs(n, rng)
        Y = Ya - b * (Yg - GEO_EXACT)
        s, s2 = s + Y.sum(), s2 + Y @ Y
        done += n
    return mean_se(s, s2, N)


# 2b. Moment matching: within a batch of n paths, rescale the prices at each date
# so their sample mean equals the true mean E[S_t] = S0 exp((r - q) t). This makes
# the paths in a batch dependent, so the usual i.i.d. standard error no longer
# applies; it is computed instead from independent batch means. Dividing by a
# sample mean also biases the estimator by O(1/n), so batches must be large.
MM_TARGETS = S0 * np.exp((r - q) * T / m * np.arange(1, m + 1))


def mc_moment_matching(N, rng, n=MM_BATCH):
    batches = N // n
    per_chunk = max(1, CHUNK // n)  # small batches are matched several at a time
    means = np.empty(batches)
    for i in range(0, batches, per_chunk):
        k = min(per_chunk, batches - i)
        S = np.exp(log_paths(k * n, rng)).reshape(k, n, m)
        S *= MM_TARGETS / S.mean(axis=1, keepdims=True)
        means[i:i + k] = DISC * np.maximum(S.mean(axis=2) - K, 0.0).mean(axis=1)
    return means.mean(), means.std(ddof=1) / np.sqrt(batches)


# Paths needed for a 95% CI half-width of half a cent, given a per-path standard
# deviation from a pilot run. The margin covers sampling noise in the pilot's
# own estimate: 2% suits i.i.d. pilots. Moment matching gets 30%: its pilot
# standard deviation comes from 400 batch means (~3.5% noise, ~7% in N), and the
# final run's own standard error comes from only ~200 batch means (~5% noise).
# N is rounded up to a whole number of batches.
def paths_for_cent(sd, margin=1.02, multiple=1):
    N = int(np.ceil(margin * (1.96 * sd / CENT) ** 2))
    return -(-N // multiple) * multiple


# Best-of-`repeats` wall time; large runs are timed once since they take seconds
def timed(fn, N, rng):
    repeats = 3 if N <= 10**6 else 1
    best = np.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn(N, rng)
        best = min(best, time.perf_counter() - t0)
    return result, best


def convergence_table(fn, Ns, rng):
    rows = []
    for N in Ns:
        (price, se), elapsed = timed(fn, N, rng)
        rows.append({
            "N": N,
            "MC price": price,
            "Std error": se,
            "CI lower": price - 1.96 * se,
            "CI upper": price + 1.96 * se,
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
        "Time (s)": "{:.2e}".format,
    }) + "\n")


def plot_results(tables):
    os.makedirs(PLOT_DIR, exist_ok=True)
    colors = {"Standard": "C0", "Control variate": "C1", "Moment matching": "C2"}

    # CI half-width against N on log-log axes, with N^(-1/2) guides
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, df in tables.items():
        Ns = df["N"].to_numpy(dtype=float)
        ax.loglog(Ns, 1.96 * df["Std error"], "o-", color=colors[label], label=label)
        ax.loglog(Ns, 1.96 * df["Std error"].iloc[-1] * np.sqrt(Ns[-1] / Ns), ":", color=colors[label], alpha=0.6)
    ax.axhline(CENT, color="red", lw=1, label="Half a cent")
    ax.plot([], [], ":", color="gray", label=r"$\propto N^{-1/2}$")
    ax.set_xlabel("Number of paths N")
    ax.set_ylabel("95% CI half-width ($)")
    ax.set_title("Convergence of Monte Carlo Asian call price")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "asian_convergence_error.png"), dpi=150)

    # Price estimates with 95% CIs against N, around the most accurate estimate
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (label, df), shift in zip(tables.items(), (0.92, 1.0, 1.08)):
        ax.errorbar(df["N"] * shift, df["MC price"], yerr=1.96 * df["Std error"],
                    fmt="o", color=colors[label], capsize=3, label=label)
    best = min(tables.values(), key=lambda df: df["Std error"].iloc[-1])
    ax.axhline(best["MC price"].iloc[-1], color="black", lw=1,
               label=f"Most accurate estimate {best['MC price'].iloc[-1]:.4f}")
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

    # Pilot runs estimate each method's per-path standard deviation, which gives
    # the number of paths it needs for cent accuracy
    pilot = 10**6
    _, se_pilot, _, _ = mc_asian(pilot, rng)
    b = control_variate_coef(10**5, rng)
    cv = lambda N, rng: mc_control_variate(N, rng, b)
    sd = {
        "Standard": se_pilot * np.sqrt(pilot),
        "Control variate": cv(pilot, rng)[1] * np.sqrt(pilot),
        "Moment matching": mc_moment_matching(4 * pilot, rng)[1] * np.sqrt(4 * pilot),
    }
    N_needed = {label: paths_for_cent(sd[label]) for label in ("Standard", "Control variate")}
    N_needed["Moment matching"] = paths_for_cent(sd["Moment matching"], 1.30, MM_BATCH)

    # Standard Monte Carlo, with the geometric Asian check on the same paths
    rows, geo_rows = [], []
    for N in sorted([10**k for k in range(2, 8)] + [N_needed["Standard"]]):
        (price, se, geo_price, geo_se), elapsed = timed(mc_asian, N, rng)
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
            "Abs error": abs(geo_price - GEO_EXACT),
            "Exact in CI": abs(geo_price - GEO_EXACT) <= 1.96 * geo_se,
        })
    tables = {"Standard": pd.DataFrame(rows)}
    geo_results = pd.DataFrame(geo_rows)

    tables["Control variate"] = convergence_table(
        cv, sorted([10**k for k in range(2, 7)] + [N_needed["Control variate"]]), rng)
    tables["Moment matching"] = convergence_table(
        mc_moment_matching, sorted([10**5, 10**6, 10**7, N_needed["Moment matching"]]), rng)

    pd.set_option("display.width", 120)
    print_table("Arithmetic Asian call: standard Monte Carlo", tables["Standard"])
    print_table(f"2a. Control variate (geometric Asian call, b = {b:.4f})", tables["Control variate"])
    print_table(f"2b. Moment matching (batches of {MM_BATCH:,} paths; SE from batch means)",
                tables["Moment matching"])

    # Head-to-head: variance per path, paths and time needed for cent accuracy
    t_plain = tables["Standard"].set_index("N").loc[N_needed["Standard"], "Time (s)"]
    summary = pd.DataFrame([{
        "Method": label,
        "Std dev per path": s,
        "Variance reduction": sd["Standard"] ** 2 / s**2,
        "N for cent": N_needed[label],
        "Time for cent (s)": tables[label].set_index("N").loc[N_needed[label], "Time (s)"],
    } for label, s in sd.items()])
    summary["Speedup"] = t_plain / summary["Time for cent (s)"]
    print("Comparison at cent accuracy\n" + summary.to_string(index=False, formatters={
        "Std dev per path": "{:.4f}".format,
        "Variance reduction": "{:.1f}x".format,
        "N for cent": "{:,}".format,
        "Time for cent (s)": "{:.4f}".format,
        "Speedup": "{:.0f}x".format,
    }))
    rho = np.sqrt(1 - sd["Control variate"] ** 2 / sd["Standard"] ** 2)
    print(f"Implied correlation between arithmetic and geometric payoffs: {rho:.5f}")

    # Moment-matching bias against batch size, measured against a precise
    # control-variate price; 4 million paths per row
    ref, ref_se = cv(4 * 10**6, rng)
    bias_rows = []
    for n in (10, 100, 1000, 10**4):
        price, se = mc_moment_matching(4 * 10**6, rng, n)
        bias_rows.append({"Batch size": n, "MM price": price, "Std error": se, "Bias": price - ref,
                          "Bias / SE": (price - ref) / np.hypot(se, ref_se)})
    print(f"\nMoment-matching bias by batch size (reference: control variate {ref:.5f} +/- {ref_se:.5f})\n"
          + pd.DataFrame(bias_rows).to_string(index=False, formatters={
              "Batch size": "{:,}".format, "MM price": "{:.5f}".format,
              "Std error": "{:.5f}".format, "Bias": "{:+.5f}".format, "Bias / SE": "{:+.1f}".format}))

    print(f"\nValidation: geometric Asian call, exact price {GEO_EXACT:.6f}\n" + geo_results.to_string(
        index=False, formatters={
            "N": "{:,}".format,
            "MC geo price": "{:.4f}".format,
            "Std error": "{:.5f}".format,
            "Abs error": "{:.5f}".format,
        }))

    plot_results(tables)
    print(f"\nPlots saved to {PLOT_DIR}")
