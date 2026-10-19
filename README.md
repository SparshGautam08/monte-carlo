# Monte Carlo Methods for Option Pricing

## European put

Prices a European put under Black-Scholes-Merton by Monte Carlo simulation and compares it with the exact closed-form price. Runs a convergence study over sample sizes from 100 to 20 million, reporting standard errors, 95% confidence intervals and timing, and estimates the sample size needed for cent-level accuracy.

## Run

```bash
pip install numpy pandas scipy matplotlib
python "Monte Carlo Pricing of a European Put.py"
python "Monte Carlo Pricing of an Asian Call.py"
python "Asian Put via Path Equivalence Classes.py"
```

The script also compares standard Monte Carlo with antithetic variates, benchmarks a Python loop against the vectorized NumPy version, and saves convergence plots to `plots/`.

## Asian call

Prices an arithmetic-average Asian call (50 monitoring dates) by Monte Carlo, simulating paths in chunks to keep memory bounded. There is no closed form for the arithmetic Asian, so the same paths also price the geometric Asian call, which is checked against its exact closed-form price.

## Asian put via path equivalence classes

An experiment applying the path equivalence classes of Torres, Sreenivas & Sowers ([arXiv:2510.02727](https://arxiv.org/abs/2510.02727)) to exact pricing of an arithmetic Asian put on a trinomial tree. It compares the exact tree price against brute-force enumeration and Monte Carlo, and examines when weighting paths equally gives the risk-neutral price. See [the write-up](Asian%20Put%20via%20Path%20Equivalence%20Classes.md).
