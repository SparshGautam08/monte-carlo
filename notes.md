# Project notes

Working notes for the final report. Numbers are from the latest runs (seed 42). Rerun the scripts before writing the report, in case anything changes.

Grading criteria from the PDF: (1) complete and correct results, (2) an organized report, (3) results explained with tables and plots, (4) efforts to reduce computation time, reported and explained.

"Accurate to the cent" throughout means the 95% CI half-width is below $0.005, so the estimate rounds to the right cent with 95% confidence.

## Part 1: European put

`Monte Carlo Pricing of a European Put.py`. S₀ = K = 100, T = 0.5, r = 4%, q = 2%, σ = 20%.

- Exact Black-Scholes price: **5.074637**.
- Standard MC: payoff std dev ≈ 7.2, so cent accuracy needs **≈ 8.0M samples** (about 0.1 s).
- Error falls like N^(−1/2), shown on the log-log plot; going 10× more accurate costs 100× the samples.
- **1a, antithetic:** pairs (Z, −Z), with SE computed over the pair averages, not over individual payoffs. Gives **2.0× less variance**, cent accuracy at **≈ 4.0M samples** (about 0.04 s).
- **Computation time:** NumPy vectorization is **≈ 55× faster** than a Python loop at N = 10⁵.
- Figures: `plots/convergence_error.png`, `plots/convergence_price.png`.

## Part 2: Asian call

`Monte Carlo Pricing of an Asian Call.py`. S₀ = K = 100, T = 1, r = 10%, q = 0, σ = 20%, m = 50 dates, averaging over t₁…t₅₀ (excluding S₀).

- **Validation:** there's no exact price for the arithmetic Asian, so the geometric Asian (exact price 6.893214) is priced on the same paths. Over 40 seeds the z-scores had mean −0.12 ± 0.16 and SD 0.98, consistent with no bias.
- **Standard MC:** price ≈ **7.164–7.170**, payoff std dev ≈ 8.69, cent accuracy at **≈ 11.8M paths, ≈ 5 s**.
- **Memory (criterion 4):** paths are simulated in chunks of 10⁵ (≈ 40 MB) with running sums of Y and Y², instead of one 10⁷ × 50 array (≈ 4 GB). In-place NumPy operations avoid temporary copies.
- **2a, control variate** (geometric Asian, b ≈ 1.037 from an independent pilot, so the estimate stays unbiased):
  - correlation between the arithmetic and geometric payoffs is **0.99956**, giving **≈ 1,140× less variance**;
  - cent accuracy at **≈ 10,400 paths in ≈ 4 ms**, about **1,200× faster** than standard MC;
  - best price estimate: **7.1644 ± 0.0005** at N = 10⁶.
- **2b, moment matching** (rescale each date's prices so their sample mean equals S₀e^((r−q)t)):
  - ≈ **8.4× less variance**, cent accuracy at **≈ 1.8M paths, ≈ 0.8 s** (≈ 6× faster);
  - **standard error:** rescaling makes paths within a batch dependent, so the i.i.d. formula doesn't apply. Use batch means of independent batches instead;
  - **bias:** the estimator is biased by roughly −2.2/n for batches of n paths (measured −19.6¢ at n = 10, −1.7¢ at n = 100, −0.3¢ at n = 1,000, about −0.1¢ at n = 10,000, which isn't significant). That's why batches are 10,000 paths;
  - **margin:** a standard error estimated from about 200 batches is itself ±5% noisy, so the N for cent accuracy carries a 30% margin.
- **Why the control variate wins:** the geometric average is almost a deterministic function of the arithmetic one for σ = 20%, so nearly all the noise cancels. Moment matching only fixes the first moment at each date.
- Figures: `plots/asian_convergence_error.png`, `plots/asian_convergence_price.png`.

## Part 3: American put (Longstaff–Schwartz)

Not started. S₀ = K = 100, T = 1/12, r = 4%, q = 2%, σ = 20%. Study how N (paths), M (time steps) and the number of regressors affect the price. Optional 3a is a BBSR benchmark (binomial Black-Scholes with Richardson extrapolation); optional 3b is a literature review of LSM improvements.

## Report ideas

- Use one table per part (N, price, SE, CI, error, time) and the two plots per part.
- Show a "computation time" summary across parts: vectorization → chunking → antithetic → control variate (the biggest win) → moment matching.
- Point out what went wrong and was fixed along the way. It shows rigor:
  - an undersized pilot that missed cent accuracy, fixed with a larger pilot and a safety margin;
  - the moment-matching bias for small batches.
- **Possible extra:** the Asian put experiment with path equivalence classes (`Asian Put via Path Equivalence Classes.md`) prices a discretely monitored Asian exactly on a trinomial tree. It compares the tree price with GBM Monte Carlo and could be an appendix.
