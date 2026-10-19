# Exact pricing of an arithmetic-average Asian put on a recombining trinomial tree
# by grouping paths into equivalence classes, following Torres, Sreenivas & Sowers,
# "On the Enumeration of all Unique Paths of Recombining Trinomial Trees"
# (arXiv:2510.02727). Two paths that end at the same node and visit each price
# level the same number of times (the same "cardinality tuple") have the same
# path sum, so the Asian payoff only has to be evaluated once per class.
#
# The paper counts paths per class, which gives an expectation only when all
# paths ending at a node are equally likely. Under risk-neutral trinomial
# probabilities they usually are not, so this script
#   1. carries probability mass per class instead of path counts (always exact),
#   2. shows the paper's count-weighted average is exact if and only if p_m^2 = p_u * p_d,
#      and picks the tree stretch lambda that makes this hold,
#   3. checks everything against brute-force enumeration and Monte Carlo.
import os
import time
import itertools
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Same market as the club Asian call (project part 2), with a put payoff.
# The tree has one step per monitoring date, so D steps <-> m = D dates.
S0, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.10, 0.0, 0.20
NU = r - q - 0.5 * sigma**2
DISC = np.exp(-r * T)
PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")


# Kamrad-Ritchken trinomial tree: log-price moves by -dx, 0, +dx with
# dx = lam * sigma * sqrt(dt); probabilities match the mean and variance of
# the log-price increment. Returns dx and (p_d, p_m, p_u) for steps (-1, 0, +1).
def tree_params(D, lam):
    dt = T / D
    dx = lam * sigma * np.sqrt(dt)
    drift = NU * np.sqrt(dt) / (2 * lam * sigma)
    pu, pd = 1 / (2 * lam**2) + drift, 1 / (2 * lam**2) - drift
    return dx, (pd, 1 - 1 / lam**2, pu)


# A path ending at k* with j_- down-steps has probability
#   p_u^{k*} p_m^{D-k*} (p_u p_d / p_m^2)^{j_-},
# so all paths to a node are equally likely if and only if p_m^2 = p_u p_d. With
# a = 1/lam^2 and b = NU^2 dt / sigma^2 this is 3a^2 - (8 - b)a + 4 = 0;
# the smaller root keeps p_m >= 0. With zero drift it is the classic lam^2 = 3/2.
def equiprobable_lambda(D):
    b = NU**2 * (T / D) / sigma**2
    a = ((8 - b) - np.sqrt((8 - b) ** 2 - 48)) / 6
    return 1 / np.sqrt(a)


# Forward pass over the tree that merges paths as soon as they share
# (current position, cardinality tuple). Each state carries the total
# probability of its paths and how many paths it holds. The tuple is stored
# with offset D so index k + D counts visits to position k (including d = 0).
def path_classes(D, probs):
    start = [0] * (2 * D + 1)
    start[D] = 1
    states = {(0, tuple(start)): (1.0, 1)}
    for _ in range(D):
        nxt = defaultdict(lambda: [0.0, 0])
        for (k, c), (w, n) in states.items():
            for s, p in zip((-1, 0, 1), probs):
                i = k + s + D
                entry = nxt[(k + s, c[:i] + (c[i] + 1,) + c[i + 1:])]
                entry[0] += w * p
                entry[1] += n
        states = nxt
    return states


# Prices from the classes: the exact probability-weighted price, and the
# paper's count-weighted average  sum_p p N_p / sum_p N_p  taken per terminal
# node and mixed with the terminal-node probabilities.
def price_from_classes(states, D, dx):
    ends = np.array([k for k, _ in states])
    C = np.array([c for _, c in states], dtype=float)
    W, N = np.array(list(states.values())).T
    levels = S0 * np.exp(dx * np.arange(-D, D + 1))
    avg = (C @ levels - S0) / D          # average over t_1..t_D, excluding S0
    payoff = np.maximum(K - avg, 0.0)

    exact = DISC * W @ payoff
    P_end = np.bincount(ends + D, weights=W)
    N_end = np.bincount(ends + D, weights=N)
    count_weighted = DISC * np.sum(P_end[ends + D] * N / N_end[ends + D] * payoff)
    return exact, count_weighted


# All 3^D paths at once, for small D. Returns the price and, per terminal node,
# the largest ratio between the most and least likely path into it.
def brute_force(D, probs, dx):
    steps = np.array(list(itertools.product((-1, 0, 1), repeat=D)), dtype=np.int8)
    k = np.cumsum(steps, axis=1)
    avg = (S0 * np.exp(dx * k)).mean(axis=1)
    log_prob = np.log(np.array(probs))[steps + 1].sum(axis=1)
    price = DISC * np.exp(log_prob) @ np.maximum(K - avg, 0.0)
    spread = max(np.ptp(log_prob[k[:, -1] == e]) for e in range(-D, D + 1))
    return price, np.exp(spread)


# Monte Carlo on the tree itself, to check the class-based price independently
def mc_tree(D, probs, dx, N, rng):
    steps = rng.choice(np.array([-1, 0, 1], dtype=np.int8), size=(N, D), p=probs)
    avg = (S0 * np.exp(dx * np.cumsum(steps, axis=1))).mean(axis=1)
    Y = DISC * np.maximum(K - avg, 0.0)
    return Y.mean(), Y.std(ddof=1) / np.sqrt(N)


# The club's Monte Carlo pricer under continuous GBM with m = D monitoring dates
def mc_gbm(D, N, rng, chunk=2 * 10**5):
    dt = T / D
    s = s2 = 0.0
    for start in range(0, N, chunk):
        n = min(chunk, N - start)
        X = rng.standard_normal((n, D)) * (sigma * np.sqrt(dt)) + NU * dt
        avg = (S0 * np.exp(np.cumsum(X, axis=1))).mean(axis=1)
        Y = DISC * np.maximum(K - avg, 0.0)
        s, s2 = s + Y.sum(), s2 + Y @ Y
    mean = s / N
    return mean, np.sqrt((s2 - N * mean**2) / (N - 1) / N)


def sanity_checks():
    # Figure 1 of the paper: D = 4, node values 20 + 2k. Paths ending at 0 that
    # visit 0 four times and 1 once all sum to 102, and there are 3 of them.
    (c, (_, n)), = [(c, v) for (k, c), v in path_classes(4, (1, 1, 1)).items()
                    if k == 0 and c[4] == 4 and c[5] == 1]
    fig1_sum = sum(cnt * (20 + 2 * (i - 4)) for i, cnt in enumerate(c))
    print(f"Paper Figure 1: {n} paths with sum {fig1_sum} (paper: 3 paths, sum 102)")

    # Distinct (endpoint, tuple) pairs read off all 3^D paths must match the
    # number of classes the forward pass builds
    for D in (5, 7, 9):
        steps = np.array(list(itertools.product((-1, 0, 1), repeat=D)))
        k = np.hstack([np.zeros((len(steps), 1), int), np.cumsum(steps, axis=1)])
        distinct = {(row[-1], tuple(np.bincount(row + D, minlength=2 * D + 1))) for row in k}
        print(f"D = {D}: {len(distinct):,} distinct classes by brute force, "
              f"{len(path_classes(D, (1, 1, 1))):,} from the forward pass")

    # Same tuple, different probability: (0,1,0,1) and (0,0,1,1) both visit
    # 0 twice and 1 twice, but use steps (u,d,u) and (m,u,m)
    D, lam = 3, np.sqrt(2)
    _, (pd, pm, pu) = tree_params(D, lam)
    print(f"Paths (0,1,0,1) and (0,0,1,1) share a tuple; with lambda = sqrt(2), D = 3 their "
          f"probabilities are {pu * pu * pd:.4f} and {pm * pm * pu:.4f}\n")


if __name__ == "__main__":
    os.makedirs(PLOT_DIR, exist_ok=True)
    sanity_checks()

    # 1. Class-based pricing matches brute-force enumeration exactly, and the
    #    count-weighted average is exact only at the equiprobable lambda
    rows = []
    for D in range(2, 13):
        for label, lam in [("lambda*", equiprobable_lambda(D)), ("sqrt(2)", np.sqrt(2))]:
            dx, probs = tree_params(D, lam)
            t0 = time.perf_counter()
            bf, spread = brute_force(D, probs, dx)
            t_bf = time.perf_counter() - t0
            t0 = time.perf_counter()
            states = path_classes(D, probs)
            exact, counted = price_from_classes(states, D, dx)
            t_cls = time.perf_counter() - t0
            rows.append({"D": D, "lambda": label, "Paths 3^D": 3**D, "Classes": len(states),
                         "Brute force": bf, "Classes exact": exact, "|diff|": abs(bf - exact),
                         "Count-weighted": counted, "Count bias": counted - exact,
                         "Max prob ratio": spread, "t brute (s)": t_bf, "t classes (s)": t_cls})
    check = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("Brute force vs equivalence classes\n" + check.to_string(index=False, formatters={
        "Paths 3^D": "{:,}".format, "Classes": "{:,}".format,
        "Brute force": "{:.6f}".format, "Classes exact": "{:.6f}".format, "|diff|": "{:.1e}".format,
        "Count-weighted": "{:.6f}".format, "Count bias": "{:+.2e}".format,
        "Max prob ratio": "{:.3f}".format, "t brute (s)": "{:.3f}".format,
        "t classes (s)": "{:.3f}".format}) + "\n")

    # 2. Bias of the count-weighted average as a function of the tree stretch
    D_bias = 10
    lam_star = equiprobable_lambda(D_bias)
    lams = np.unique(np.concatenate([np.linspace(1.0, 2.0, 41), [lam_star]]))
    bias = []
    for lam in lams:
        dx, probs = tree_params(D_bias, lam)
        exact, counted = price_from_classes(path_classes(D_bias, probs), D_bias, dx)
        bias.append((counted - exact) * 100)  # cents
    print(f"Equiprobable stretch at D = {D_bias}: lambda* = {lam_star:.5f} "
          f"(zero-drift value sqrt(3/2) = {np.sqrt(1.5):.5f})\n")

    # 3. Deeper trees with classes only, against Monte Carlo on the tree and
    #    the club's GBM Monte Carlo with the same number of monitoring dates
    rng = np.random.default_rng(seed=42)
    N_mc = 10**6
    conv = []
    for D in range(2, 19, 2):
        dx, probs = tree_params(D, equiprobable_lambda(D))
        t0 = time.perf_counter()
        states = path_classes(D, probs)
        exact, _ = price_from_classes(states, D, dx)
        t_cls = time.perf_counter() - t0
        tree_mc, tree_se = mc_tree(D, probs, dx, N_mc, rng)
        gbm, gbm_se = mc_gbm(D, N_mc, rng)
        per_node = np.bincount([k + D for k, _ in states])
        conv.append({"D = m": D, "Paths 3^D": float(3**D), "Classes": len(states),
                     "Max per node": per_node.max(), "at |k*|": abs(per_node.argmax() - D),
                     "Tree exact": exact, "t classes (s)": t_cls,
                     "Tree MC": tree_mc, "Tree MC CI": 1.96 * tree_se,
                     "Exact in tree CI": abs(tree_mc - exact) <= 1.96 * tree_se,
                     "GBM MC": gbm, "GBM MC CI": 1.96 * gbm_se, "Tree - GBM": exact - gbm})
        print(f"  D = {D}: {len(states):,} classes in {t_cls:.2f}s", flush=True)
    conv = pd.DataFrame(conv)
    print("\nTree (exact, by classes) vs Monte Carlo, N = 10^6 paths each\n" + conv.to_string(
        index=False, formatters={
            "Paths 3^D": "{:.2e}".format, "Classes": "{:,}".format,
            "Max per node": "{:,}".format, "Tree exact": "{:.5f}".format,
            "t classes (s)": "{:.2f}".format, "Tree MC": "{:.5f}".format,
            "Tree MC CI": "{:.5f}".format, "GBM MC": "{:.5f}".format,
            "GBM MC CI": "{:.5f}".format, "Tree - GBM": "{:+.5f}".format}))

    # Growth of the number of classes against 3^D and the paper's 1.612^D rate
    deep = conv[conv["D = m"] >= 8]
    growth = np.polyfit(deep["D = m"], np.log(deep["Classes"]), 1)[0]
    growth_node = np.polyfit(deep["D = m"], np.log(deep["Max per node"]), 1)[0]
    print(f"\nFitted growth, D >= 8: all classes ~ {np.exp(growth):.3f}^D, largest single node "
          f"~ {np.exp(growth_node):.3f}^D (paper: 1.612^D; raw paths: 3^D)")

    # Plots
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(conv["D = m"], conv["Paths 3^D"], "o-", label="All paths, $3^D$")
    ax.semilogy(conv["D = m"], conv["Classes"], "o-", label="Equivalence classes (position, tuple)")
    Ds = conv["D = m"].to_numpy(dtype=float)
    ax.semilogy(Ds, conv["Classes"].iloc[-1] * 1.612 ** (Ds - Ds[-1]), ":", color="gray",
                label=r"$\propto 1.612^D$ (paper)")
    ax.set_xlabel("Tree depth D (= monitoring dates)")
    ax.set_ylabel("Count")
    ax.set_title("Paths vs. path equivalence classes")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "asian_put_classes_vs_paths.png"), dpi=150)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(lams, bias, "o-", ms=3)
    ax.axhline(0, color="black", lw=1)
    ax.axvline(lam_star, color="red", lw=1, ls="--", label=rf"$\lambda^*$ = {lam_star:.4f} ($p_m^2 = p_u p_d$)")
    ax.set_xlabel(r"Tree stretch $\lambda$")
    ax.set_ylabel("Count-weighted minus exact price (cents)")
    ax.set_title(f"Error from weighting paths equally, D = {D_bias}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "asian_put_count_weighting_bias.png"), dpi=150)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(conv["D = m"], conv["Tree exact"], "o-", label="Trinomial tree, exact via classes")
    ax.errorbar(conv["D = m"], conv["GBM MC"], yerr=conv["GBM MC CI"], fmt="s", capsize=3,
                label="Monte Carlo under GBM, 95% CI")
    ax.set_xlabel("Monitoring dates m (= tree depth D)")
    ax.set_ylabel("Asian put price ($)")
    ax.set_title("Exact tree price vs. Monte Carlo")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "asian_put_tree_vs_mc.png"), dpi=150)
    plt.close("all")
    print(f"\nPlots saved to {PLOT_DIR}")
