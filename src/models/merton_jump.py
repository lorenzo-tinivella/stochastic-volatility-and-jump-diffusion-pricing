import numpy as np


class MertonJumpDiffusion:

    def __init__(self, S0, r, sigma, lam, mu_j, sigma_j):
        """
        S0 : initial asset price
        r : risk-free rate
        sigma : diffusion volatility
        lam : jump intensity (Poisson)
        mu_j : mean jump size
        sigma_j : jump volatility
        """

        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.lam = lam
        self.mu_j = mu_j
        self.sigma_j = sigma_j


    def simulate_terminal_price(self, T, n_paths):
        """
        Simulate terminal prices under Merton jump diffusion
        """

        Z = np.random.normal(size=n_paths)

        drift = (
            self.r
            - 0.5 * self.sigma**2
            - self.lam * (np.exp(self.mu_j + 0.5 * self.sigma_j**2) - 1)
        )

        diffusion = drift * T + self.sigma * np.sqrt(T) * Z

        N = np.random.poisson(self.lam * T, n_paths)

        jumps = np.random.normal(self.mu_j, self.sigma_j, n_paths)

        jump_component = N * jumps

        ST = self.S0 * np.exp(diffusion + jump_component)

        return ST


    def call_price_mc(self, K, T, n_paths=100000):
        """
        Monte Carlo pricing of European call option
        """

        ST = self.simulate_terminal_price(T, n_paths)

        payoff = np.maximum(ST - K, 0)

        price = np.exp(-self.r * T) * np.mean(payoff)

        return price