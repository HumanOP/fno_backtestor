from scipy.stats import norm
import numpy as np


class BlackScholes:
    def __init__(self, S, K, T, r, sigma):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.d1 = (np.log(self.S/self.K) + ((self.r + 0.5 * self.sigma ** 2) * self.T)) / (self.sigma * np.sqrt(self.T))
        self.d2 = self.d1 - self.sigma*np.sqrt(self.T)
        
    def call_price(self):
        return self.S * norm.cdf(self.d1) - self.K * np.exp(-self.r*self.T) * norm.cdf(self.d2)
    
    def put_price(self):
        return self.K * np.exp(-self.r*self.T) * norm.cdf(-self.d2) - self.S * norm.cdf(-self.d1)
        