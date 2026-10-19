# Monte Carlo Pricing of a European Put

Prices a European put under Black-Scholes-Merton by Monte Carlo simulation and compares it with the exact closed-form price. Runs a convergence study over sample sizes from 100 to 20 million, reporting standard errors, 95% confidence intervals and timing, and estimates the sample size needed for cent-level accuracy.

## Run

```bash
pip install numpy pandas scipy matplotlib
python "Monte Carlo Pricing of a European Put.py"
```

The script also compares standard Monte Carlo with antithetic variates, benchmarks a Python loop against the vectorized NumPy version, and saves convergence plots to `plots/`.
