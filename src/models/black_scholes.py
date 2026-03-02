import numpy as np
from scipy.stats import norm

class BlackScholes:

    def __init__(self, S0, r, sigma):
        self.S0 = S0      # initial price
        self.r = r        # risk-free rate
        self.sigma = sigma  # volatility

    def simulate(self, T, n_paths):
        Z = np.random.normal(size=n_paths)
        ST = self.S0 * np.exp(
            (self.r - 0.5 * self.sigma**2) * T
            + self.sigma * np.sqrt(T) * Z
        )
        return ST

    def call_price_closed_form(self, K, T):
        d1 = (np.log(self.S0 / K) +
              (self.r + 0.5 * self.sigma**2) * T) / (self.sigma * np.sqrt(T))
        d2 = d1 - self.sigma * np.sqrt(T)
        return self.S0 * norm.cdf(d1) - K * np.exp(-self.r * T) * norm.cdf(d2)

    def call_price_mc(self, K, T, n_paths=100000):
        ST = self.simulate(T, n_paths)
        payoff = np.maximum(ST - K, 0)
        return np.exp(-self.r * T) * np.mean(payoff)
