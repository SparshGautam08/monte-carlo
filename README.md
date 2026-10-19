# Monte Carlo Pricing of a European Put

Prices a European put under Black-Scholes-Merton by Monte Carlo simulation and compares it with the exact closed-form price. Runs a convergence study over sample sizes from 100 to 20 million, reporting standard errors, 95% confidence intervals and timing, and estimates the sample size needed for cent-level accuracy.

## Run

```bash
pip install numpy pandas scipy
python "Monte Carlo Pricing of a European Put.py"
```
