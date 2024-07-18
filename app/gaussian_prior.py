import math
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.widgets import Button
from scipy.stats import norm


class Prior:
    def __init__(self, mu, k):
        self.mu = mu
        self.k = k


colors = ["red", "purple", "blue", "green", "magenta"]


class ThompsonSamplingGaussianPrior(object):
    def __init__(self, arms):
        self.N = len(arms)
        self.arms = arms
        self.ks = np.zeros(self.N)
        self.mus = np.zeros(self.N)
        self.buttons = []

    def initialize(self, priors: list[Prior]):
        if priors:
            self.ks = np.array([prior.k for prior in priors])
            self.mus = np.array([prior.mu for prior in priors])
        else:
            self.ks = np.zeros(self.N)
            self.mus = np.zeros(self.N) + 5.0

    def sigmas(self, i):
        return math.sqrt(10.0 / (self.ks[i] + 1.0))

    def select_arm(self):
        return self.arms[np.argmax(self.estimate_mean())]

    def estimate_mean(self):
        # For each arm i=1,...,N, sample random value from normal distribution
        theta = [np.random.normal(self.mus[i], self.sigmas(i)) for i in range(self.N)]
        return theta

    def update_param(self, arm_id, reward):
        # update parameter of normal distribution
        self.mus[arm_id] = (self.mus[arm_id] * (self.ks[arm_id] + 1.0) + reward) / (
            self.ks[arm_id] + 2.0
        )
        self.ks[arm_id] += 1.0

    def get_title(self, arm_id):
        arm = self.arms[arm_id]
        name = arm.name
        mu = self.mus[arm_id]
        std = self.sigmas(arm_id)
        return "mean: {:.2f}, std: {:.2f}".format(mu, std)

    def print_arms(self):
        for i in range(self.N):
            arm = self.arms[i]
            mu = self.mus[i]
            std = self.sigmas(i)
            print(f"Arm id: {i}, name: {arm.name} mu: {mu} std: {std}")
        print()
    
    def make_button(self, axs, i, on_click):
        button_plot = axs[i, 1]
        button = Button(
            button_plot,
            label=self.arms[i].name,
            hovercolor="tomato",
        )
        button.on_clicked(lambda _: on_click(i))
        self.buttons.append(button)

    def plot_distributions(self, on_click, reward_string: Optional[str] = None):
        plt.style.use("seaborn-v0_8")

        x = np.arange(-20, 50, 0.01)
        fig, axs = plt.subplots(self.N, 2)
        fig.set_size_inches((8, 8))
        if reward_string:
            fig.suptitle(reward_string)
        else:
            fig.suptitle("")
        plt.tight_layout()

        for i in range(self.N):
            mu = self.mus[i]
            sig = self.sigmas(i)

            subplot = axs[i, 0]
            subplot.plot(x, norm.pdf(x, mu, sig), color=colors[i % len(colors)])
            subplot.set_title(self.get_title(i))
            self.make_button(axs, i, on_click)

        plt.show()
        return self.buttons

