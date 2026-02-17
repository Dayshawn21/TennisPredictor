from collections import OrderedDict
from math import erf, sqrt, exp, log, factorial
import math

def _cdf(x, mu, sigma):
    z = (x - mu) / (sigma + 1e-9)
    return 0.5 * (1 + erf(z / sqrt(2)))

def discrete_normal_pmf(mu, sigma, k_min, k_max):
    pmf = OrderedDict()
    for k in range(max(0, k_min), k_max + 1):
        p = _cdf(k + 0.5, mu, sigma) - _cdf(k - 0.5, mu, sigma)
        pmf[k] = max(0.0, p)
    s = sum(pmf.values()) or 1.0
    return OrderedDict((k, v / s) for k, v in pmf.items())

def ev_per_dollar(odds: int, p: float) -> float:
    b = (100/abs(odds)) if odds < 0 else (odds/100)
    return p*b - (1-p)

def kelly_fraction(odds: int, p: float) -> float:
    b = (100/abs(odds)) if odds < 0 else (odds/100)
    f = (b*p - (1-p)) / b
    return max(0.0, min(f, 0.02))

def poisson_pmf(lmbda: float, k_min: int, k_max: int):
    pmf = OrderedDict()
    for k in range(max(0, k_min), k_max + 1):
        pmf[k] = exp(-lmbda) * (lmbda ** k) / factorial(k)
    s = sum(pmf.values()) or 1.0
    return OrderedDict((k, v / s) for k, v in pmf.items())

def discretized_lognormal_pmf(mu_ln: float, sigma_ln: float, k_min: int, k_max: int):
    # mu_ln, sigma_ln are params in log space; discretize with continuity correction
    def cdf(x):
        if x <= 0: return 0.0
        z = (log(x) - mu_ln) / (sigma_ln + 1e-9)
        return 0.5 * (1 + erf(z / sqrt(2)))
    pmf = OrderedDict()
    for k in range(max(0, k_min), k_max + 1):
        p = cdf(k + 0.5) - cdf(max(1e-6, k - 0.5))
        pmf[k] = max(0.0, p)
    s = sum(pmf.values()) or 1.0
    return OrderedDict((k, v / s) for k, v in pmf.items())
