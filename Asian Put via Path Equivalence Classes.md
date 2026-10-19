# Pricing an Asian put with path equivalence classes

An experiment applying the cardinality-tuple equivalence classes from Torres, Sreenivas & Sowers, [*On the Enumeration of all Unique Paths of Recombining Trinomial Trees*](https://arxiv.org/abs/2510.02727) (arXiv:2510.02727), to exact pricing of an arithmetic-average Asian put. The code is in [`Asian Put via Path Equivalence Classes.py`](Asian%20Put%20via%20Path%20Equivalence%20Classes.py) and runs in about 20 seconds.

## Setup

- Arithmetic-average Asian put, S₀ = K = 100, T = 1, r = 10%, q = 0, σ = 20%, the same market as the club's Asian call project.
- The average is taken over D monitoring dates, one per step of a Kamrad–Ritchken trinomial tree. The log-price moves by ±λσ√Δt or 0, and the probabilities match the mean and variance of each step.
- Node prices depend only on position, so the tree has the translational invariance the paper relies on. Two paths that end at the same node and visit each level the same number of times have the same path sum, and therefore the same Asian payoff.

## Method

The paper's contribution is a recursion-free mass-shifting enumerator that generates one representative per class. I did **not** reimplement that algorithm. Instead, a forward pass over the tree merges paths as soon as they share (current position, cardinality tuple), and each class carries its total probability and its path count. This builds exactly the paper's classes $\hat C_{D,k^{\ast}}$. I checked that by counting distinct tuples over all 3^D paths for D = 5, 7 and 9, and it reproduces the paper's Figure 1 example (3 paths with sum 102).

## Results

**1. The class-based price is exact.** It matches brute-force enumeration of all 3^D paths to about 1e-13 for D ≤ 12. Monte Carlo on the tree agrees within the 95% CI at 8 of 9 depths.

**2. The compression is large.** At D = 18 there are 1.97 million classes against 387 million paths, about 200× fewer, and the pass takes 8 s in pure Python.

| D | Paths 3^D | Classes | Exact tree price |
|---|---|---|---|
| 10 | 59,049 | 6,274 | 2.52871 |
| 14 | 4.8M | 115,069 | 2.48100 |
| 18 | 387M | 1,967,885 | 2.45451 |

**3. Paths in the same class can have different probabilities.** That is where the interesting part is. The paper averages path sums as Σ p·N_p / Σ N_p, which weights every path equally. Under risk-neutral trinomial probabilities, (0,1,0,1) and (0,0,1,1) have the same tuple but probabilities p_u²p_d and p_m²p_u. More generally, a path ending at k\* with j₋ down-steps has probability

$$p_u^{k^{\ast}} p_m^{D-k^{\ast}} \left(\frac{p_u p_d}{p_m^2}\right)^{j_-},$$

so **the count-weighted average is the risk-neutral price iff p_m² = p_u·p_d**. Matching mean and variance as well turns this into a quadratic in a = 1/λ²:

$$3a^2 - (8 - b) a + 4 = 0, \qquad b = \nu^2 \Delta t / \sigma^2,$$

whose admissible root λ\* → √(3/2) as the drift vanishes. That is the stretch commonly used for Kamrad–Ritchken trees. With λ\*, every path into a node is equally likely: the brute-force max/min probability ratio is exactly 1, and the paper's counting formula prices exactly. With any other λ it is biased:

| λ | Count-weighted minus exact (D = 10) |
|---|---|
| 1 (p_m = 0) | −13 cents (zero-probability paths are still counted) |
| λ\* ≈ 1.2223 | 0 |
| √2 | +13 cents (~5% of the price) |
| 2 | +68 cents |

The general fix is to carry probability mass per class, as in result 1, instead of path counts.

**4. Comparison with Monte Carlo under GBM.** With the same number of monitoring dates, the tree prices 1–3 cents above the club Monte Carlo pricer for D ≥ 10. That gap comes from using only one tree step per monitoring date.

Plots: [`asian_put_classes_vs_paths.png`](plots/asian_put_classes_vs_paths.png), [`asian_put_count_weighting_bias.png`](plots/asian_put_count_weighting_bias.png), [`asian_put_tree_vs_mc.png`](plots/asian_put_tree_vs_mc.png).

## Open questions

- **Growth rate.** I measure the number of classes growing about 2.06^D in total and about 1.97^D at the largest single terminal node, against the 1.612^D in the abstract. Is the 1.612^D figure about a different quantity, such as the positive-side classes $C^+_{D,k^{\ast}}$, or am I using a different equivalence relation than intended?
- **Where the class count peaks.** The largest $|\hat C_{D,k^{\ast}}|$ occurs at |k\*| ≈ D/3 (for example |k\*| = 7 at D = 18), not at k\* = 0 as in eq. (6.1). Does (6.1) refer to raw path counts, which do peak at 0, rather than to tuple counts?
- **General probabilities.** For p_m² ≠ p_u·p_d, the classes could be refined by (tuple, number of flat steps), since that pair fixes a path's probability. Does the weak-composition count extend to this refinement? That would give closed-form weights instead of a forward pass.
- **Deep trees.** Because the Asian payoff depends only on the path sum, classes with nearly equal sums could be merged or stratified, along the lines of the stratified-sampling application in Section 7.2(A).
