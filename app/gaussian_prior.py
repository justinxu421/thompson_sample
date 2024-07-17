import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

class ThompsonSamplingGaussianPrior(object):
    def __init__(self, arms):
        self.N = len(arms)
        self.arms = arms
        self.ks = np.zeros(self.N)
        self.mus = np.zeros(self.N)

    def initialize(self):
        self.ks = np.zeros(self.N)
        self.mus = np.zeros(self.N) + 15

    def sigmas(self):
        return np.array([math.sqrt(1.0 / (self.ks[i] + 1.0)) for i in range(self.N)])

    def select_arm(self):
        return np.argmax(self.estimate_mean())

    def estimate_mean(self):
        # For each arm i=1,...,N, sample random value from normal distribution
        theta = [np.random.normal(self.mus[i], self.sigmas()[i]) for i in range(self.N)]
        return theta

    def update_param(self, arm_id, reward):
        # update parameter of normal distribution
        self.mus[arm_id] = (self.mus[arm_id] * (self.ks[arm_id] + 1.0) + reward) / (
            self.ks[arm_id] + 2.0
        )
        self.ks[arm_id] += 1.0

    def plot_distributions(self):
        x = np.arange(-5, 50, 0.01)
        for i in range(self.N):
            #x-axis ranges from -5 and 5 with .001 steps
            mu = self.mus[i]
            sig = self.sigmas()[i]

            #define multiple normal distributions
            plt.plot(x, norm.pdf(x, mu, sig), label=self.arms[i].name)
            plt.legend()

        plt.show()
