# Monte Carlo Methods for Option Pricing

## European put

Prices a European put under Black-Scholes-Merton by Monte Carlo simulation and compares it with the exact closed-form price. Runs a convergence study over sample sizes from 100 to 20 million, reporting standard errors, 95% confidence intervals and timing, and estimates the sample size needed for cent-level accuracy. It also compares standard Monte Carlo with antithetic variates and benchmarks a Python loop against the vectorized NumPy version.

## Asian call

Prices an arithmetic-average Asian call (50 monitoring dates) by Monte Carlo, simulating paths in chunks to keep memory bounded. There is no closed form for the arithmetic Asian, so the same paths also price the geometric Asian call, which is checked against its exact closed-form price. It also compares two variance reduction methods with standard Monte Carlo: a control variate using the geometric Asian call (about 1,100× less variance) and moment matching (about 8× less variance, with its small-batch bias measured).

## Asian put via path equivalence classes

An experiment applying the path equivalence classes of Torres, Sreenivas & Sowers ([arXiv:2510.02727v2](https://arxiv.org/abs/2510.02727v2)) to exact pricing of an arithmetic Asian put on a trinomial tree. It compares the exact tree price against brute-force enumeration and Monte Carlo, and examines when weighting paths equally gives the risk-neutral price. See [the write-up](Asian%20Put%20via%20Path%20Equivalence%20Classes.md).

## Run

```bash
pip install numpy pandas scipy matplotlib
python "Monte Carlo Pricing of a European Put.py"
python "Monte Carlo Pricing of an Asian Call.py"
python "Asian Put via Path Equivalence Classes.py"
```

Each script prints its tables and saves its plots to `plots/`.
