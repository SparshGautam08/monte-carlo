# Monte Carlo Methods for Option Pricing

## European put

Prices a European put under Black-Scholes-Merton by Monte Carlo simulation and compares it with the exact closed-form price. Runs a convergence study over sample sizes from 100 to 20 million, reporting standard errors, 95% confidence intervals and timing, and estimates the sample size needed for cent-level accuracy.

## Run

```bash
pip install numpy pandas scipy matplotlib
python "Monte Carlo Pricing of a European Put.py"
python "Monte Carlo Pricing of an Asian Call.py"
```

The script also compares standard Monte Carlo with antithetic variates, benchmarks a Python loop against the vectorized NumPy version, and saves convergence plots to `plots/`.

## Asian call

Prices an arithmetic-average Asian call (50 monitoring dates) by Monte Carlo, simulating paths in chunks to keep memory bounded. There is no closed form for the arithmetic Asian, so the same paths also price the geometric Asian call, which is checked against its exact closed-form price.
