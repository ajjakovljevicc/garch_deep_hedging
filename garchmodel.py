import numpy as np
from scipy.optimize import minimize, brentq
from scipy.stats import norm

# The parameter values Duan estimated on S&P 100 data using to reproduce his data
PAPER = dict(alpha0=1.524e-5, alpha1=0.1883, beta1=0.7162, lam=7.452e-3)


# BLACK-SCHOLES: the constant volatility benchmark
def bs_call(S, K, T, r, sigma):
    """BS price and delta. sigma is per-day vol, T in days."""
    # total volatility over the option's life: daily vol * sqrt(days)
    srt = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / srt
    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d1 - srt)
    # return both the price and the Black-Scholes delta=N~(d1)
    return price, norm.cdf(d1)


def implied_vol(price, S, K, T, r):
    """Per-day implied vol from a call price."""
    # find the sigma that makes the BS price equal the given price.
    # brentq searches for the root of f(sigma) = BS(sigma) - price
    return brentq(lambda s: bs_call(S, K, T, r, s)[0] - price, 1e-6, 1.0)


# Estimation under real world measure P
def garch_m_filter(a0, a1, b1, lam, ret, r=0.0):
    """Recover eps_t and h_t from returns under P (eq 2.1-2.2)."""
    n = len(ret)
    h, eps = np.empty(n), np.empty(n)
    h[0] = ret.var()                       
    for t in range(n):
        if t > 0:
            # the GARCH(1,1) update: baseline + reaction to shock + persistence
            h[t] = a0 + a1 * eps[t - 1]**2 + b1 * h[t - 1]
        # the shock is the actual return minus the return the model expected today
        eps[t] = ret[t] - (r + lam * np.sqrt(h[t]) - 0.5 * h[t])
    # return the full history of shocks and variances implied by these parameters
    return eps, h


def _neg_loglik(theta, ret, r):
    a0, a1, b1, lam = theta[0] * 1e-5, theta[1], theta[2], theta[3]   # rescale a0
    # reject invalid parameters: variance must stay positive and stationary (a1 + b1 < 1)
    if a0 <= 0 or a1 < 0 or b1 < 0 or a1 + b1 >= 1:
        return 1e10
    # compute the shocks and variances these parameters imply for the data
    eps, h = garch_m_filter(a0, a1, b1, lam, ret, r)
    # each shock is assumed N(0, h_t). The log of the normal density is
    #   -0.5 * [ log(2*pi*h_t) + eps_t^2 / h_t ]
    # summing over days gives the log-likelihood; we return its NEGATIVE
    # because the optimizer minimizes, and we want to maximize likelihood
    return 0.5 * np.sum(np.log(2 * np.pi * h) + eps**2 / h)


def fit_garch_m(ret, r=0.0):
    """ret = daily log returns ln(X_t / X_{t-1})."""
    # search for the parameters that minimize the negative log-likelihood.
    # x0 is the starting guess: a0 = 1e-5, a1 = 0.1, b1 = 0.8, lam = 0.
    # Nelder-Mead needs no derivatives, which is robust for a small problem like this.
    res = minimize(_neg_loglik, x0=[1.0, 0.1, 0.8, 0.0], args=(ret, r),
                   method="Nelder-Mead",
                   # allow many iterations and require tight convergence
                   options=dict(maxiter=10_000, xatol=1e-8, fatol=1e-8))
    # the best parameters found (with a0 still in rescaled units)
    a0, a1, b1, lam = res.x
    # undo the rescaling of a0 and return the estimates by name
    return dict(alpha0=a0 * 1e-5, alpha1=a1, beta1=b1, lam=lam)


def simulate_P(p, n, r=0.0, seed=50):
    """Simulate returns under P -- used to sanity-check the MLE."""
    # random number generator with a fixed seed so results are reproducible
    rng = np.random.default_rng(seed)
    # pull the parameters out of the dictionary
    a0, a1, b1, lam = p["alpha0"], p["alpha1"], p["beta1"], p["lam"]
    # start at the long-run (stationary) variance a0 / (1 - a1 - b1); make an empty return array
    h, ret = a0 / (1 - a1 - b1), np.empty(n)
    # generate one day at a time
    for t in range(n):
        # today's shock: a standard normal draw scaled by today's volatility sqrt(h)
        eps = np.sqrt(h) * rng.standard_normal()
        # today's log return: risk-free + risk premium - convexity term + shock
        ret[t] = r + lam * np.sqrt(h) - 0.5 * h + eps
        # update the variance for tomorrow using today's shock
        h = a0 + a1 * eps**2 + b1 * h
    # a fake return history with known true parameters
    return ret


# ============================================================================
# PART 2: PRICING UNDER DUAN'S RISK-NEUTRAL MEASURE Q  (Thm 2.2, Cor 2.3-2.4)
# ============================================================================
def garch_call_mc(S, K, T, r, p, h_next, n_paths=50_000, seed=0):
    """
    Call price and delta under Duan's locally risk-neutral measure.
    h_next = conditional variance for the first day (known at time t).
    Under Q:  ln(X_s/X_{s-1}) = r - h_s/2 + xi_s,   xi_s ~ N(0, h_s)
              h_{s+1} = a0 + a1 (xi_s - lam sqrt(h_s))^2 + b1 h_s
    """
    # unpack the GARCH parameters
    a0, a1, b1, lam = p["alpha0"], p["alpha1"], p["beta1"], p["lam"]
    # seeded random number generator (same seed -> same paths -> reproducible prices)
    rng = np.random.default_rng(seed)
    # a table of standard normal draws: one row per simulated future, one column per day
    Z = rng.standard_normal((n_paths, T))
    # discount factor to bring the expiry payoff back to today
    disc = np.exp(-r * T)

    # every path starts with zero cumulative log return and the same starting variance
    logS, h = np.zeros(n_paths), np.full(n_paths, h_next)
    # step forward one day at a time; each line below acts on ALL paths at once
    for s in range(T):
        # today's shock on every path: standard normal times today's volatility
        xi = np.sqrt(h) * Z[:, s]
        # add today's log return. Under Q the expected return is the risk-free rate r,
        # and -0.5*h is the convexity correction so that E[price growth] = e^r exactly
        logS += r - 0.5 * h + xi
        # Duan's key result (Theorem 2.2): the variance update uses the shock SHIFTED
        # by lam*sqrt(h). For lam > 0, down moves raise tomorrow's variance more than up moves.
        h = a0 + a1 * (xi - lam * np.sqrt(h))**2 + b1 * h
    # convert cumulative log return into the stock price at expiry on every path
    ST = S * np.exp(logS)

    # ---- Control variate ----
    # Idea: price a Black-Scholes option on the SAME random draws. Its exact price is known,
    # so its simulation error tells us roughly how lucky or unlucky our draws were,
    # and we subtract that error from the GARCH estimate.
    # Black-Scholes daily vol = the GARCH long-run vol sqrt(a0 / (1 - a1 - b1))
    sig = np.sqrt(a0 / (1 - a1 - b1))
    # BS terminal price on each path: the sum of T standard normals ~ N(0, T)
    ST_bs = S * np.exp((r - 0.5 * sig**2) * T + sig * Z.sum(axis=1))
    # the exact Black-Scholes price and delta from the formula
    bs_price, bs_delta = bs_call(S, K, T, r, sig)

    def cv(x, y, y_true):
        # x: GARCH payoffs, y: BS payoffs on the same paths, y_true: exact BS value
        # covariance matrix of x and y
        c = np.cov(x, y)
        # the optimal weight b = Cov(x, y) / Var(y): how much x moves when y moves
        b = c[0, 1] / c[1, 1]
        # corrected payoffs: remove the part of x explained by y's simulation error
        adj = x - b * (y - y_true)
        # the estimate is the mean; the standard error is std / sqrt(number of paths)
        return adj.mean(), adj.std(ddof=1) / np.sqrt(len(x))

    # PRICE (Cor 2.3): discounted call payoff max(ST - K, 0) on each path, averaged with the control variate
    price, se = cv(disc * np.maximum(ST - K, 0), disc * np.maximum(ST_bs - K, 0), bs_price)
    # DELTA (Cor 2.4): discounted (ST / S) on paths that finish in the money, averaged the same way
    delta, _ = cv(disc * ST / S * (ST >= K), disc * ST_bs / S * (ST_bs >= K), bs_delta)
    # return everything in a dictionary
    return dict(price=price, se=se, delta=delta, bs_price=bs_price, bs_delta=bs_delta)


#HECKS AGAINST THE PAPER 
if __name__ == "__main__":
    p = PAPER
    sig = np.sqrt(p["alpha0"] / (1 - p["alpha1"] - p["beta1"]))
    print(f"Stationary vol: {sig*np.sqrt(365):.2%} annualised (paper: 24.13%)\n")

    # --- Table 4.1 reproduction (prices x 10,000, r = 0, K = 1)
    # stock price / strike ratios the paper uses (0.8 = deep out of the money, 1.2 = deep in)
    moneyness = [0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20]
    # the paper's GARCH prices to compare against, keyed by (days to expiry, starting vol / long-run vol)
    paper_41 = {  # (T, sqrt(h)/sigma) -> paper GARCH prices
        (30, 0.8): [0.6892, 16.434, 75.449, 251.52, 583.95, 1023.6, 2002.0],
        (30, 1.0): [0.9495, 20.930, 86.028, 266.75, 596.13, 1030.2, 2003.5],
        (30, 1.2): [1.6164, 26.905, 99.244, 284.34, 610.44, 1037.9, 2004.2],
        (90, 1.0): [15.759, 116.06, 251.02, 468.90, 772.35, 1149.4, 2040.5],
        (180, 1.0): [68.357, 257.09, 431.70, 668.50, 964.29, 1313.9, 2134.7],
    }
    # loop over each block of the table
    for (T, ratio), ref in paper_41.items():
        print(f"T={T:>3}, sqrt(h)/sigma={ratio}")
        print(f"  {'S/K':>5} {'BS':>9} {'ours':>9} {'±se':>6} {'paper':>9}")
        for m, pp in zip(moneyness, ref):
            out = garch_call_mc(m, 1.0, T, 0.0, p, (ratio * sig)**2)
            # print everything x 10,000 to match the paper's units
            print(f"  {m:>5.2f} {out['bs_price']*1e4:>9.3f} {out['price']*1e4:>9.3f}"
                  f" {out['se']*1e4:>6.3f} {pp:>9.3f}")
        print()

    # --- Fig 4.1: implied-vol smile from GARCH prices (low initial vol)
    print("Implied vol smile, T=30, sqrt(h)/sigma=0.8 (annualised):")
    # moneyness from 0.80 to 1.20 in steps of 0.05
    for m in np.arange(0.80, 1.21, 0.05):
        # GARCH price with starting vol 20% below the long-run level
        c = garch_call_mc(m, 1.0, 30, 0.0, p, (0.8 * sig)**2)["price"]
        # convert to the Black-Scholes vol that gives the same price, annualized
        print(f"  S/K={m:.2f}  IV={implied_vol(c, m, 1.0, 30, 0.0)*np.sqrt(365):.2%}")

    # --- MLE sanity check: simulate under P, re-fit, compare
    print("\nMLE recovery on 1,000 and 10,000 simulated days:")
    # try a short sample (like the paper's ~4 years) and a long one
    for n in (1_000, 10_000):
        # simulate fake data with known parameters, then estimate them back
        est = fit_garch_m(simulate_P(p, n, seed=1))
        # print the estimates
        print(f"  n={n:>6}: " + ", ".join(f"{k}={v:.4g}" for k, v in est.items()))
    # print the true values for comparison (watch lam: it is very hard to recover)
    print("  true   : " + ", ".join(f"{k}={v:.4g}" for k, v in p.items()))