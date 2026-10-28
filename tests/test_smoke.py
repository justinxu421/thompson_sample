import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from app.bandit import Bandit
from app.normal_arm import NormalArm


class DependencySmokeTest(unittest.TestCase):
    def test_posterior_update_and_plot(self):
        arms = [NormalArm("a", 10, 0), NormalArm("b", 20, 0)]
        bandit = Bandit(arms, 2)
        bandit.algorithm.initialize()
        with patch("matplotlib.pyplot.show"):
            bandit.algorithm.plot_distributions(bandit.on_click)
            bandit.on_click(1)
        self.assertEqual(bandit.total, 20)
        self.assertEqual(bandit.selected_arms["b"], 1)
        self.assertAlmostEqual(bandit.algorithm.mus[1], 40 / 3)
        self.assertAlmostEqual(bandit.algorithm.sigmas[1], np.sqrt(10 / 3))
        self.assertTrue(np.isfinite(bandit.algorithm.estimate_mean()).all())
        self.assertEqual(len(plt.gcf().axes), 4)
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
