import numpy as np

class NormalArm:
    def __init__(self, name, mean, sigma):
        self.name = name
        self.mean = mean
        self.sigma = sigma

    def play(self):
        return np.random.normal(self.mean, self.sigma)
