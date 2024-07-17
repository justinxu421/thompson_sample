from collections import defaultdict

from app.gaussian_prior import ThompsonSamplingGaussianPrior


class Bandit:
    def __init__(self, arms, play_num):
        self.arms = arms
        self.algorithm = ThompsonSamplingGaussianPrior(arms)
        self.play_num = play_num

    def experiment(self):
        self.algorithm.initialize()

        # main loop
        t = 0
        selected_arms = defaultdict(int)
        while True:
            # select arm
            arm_id = self.algorithm.select_arm()
            selected_arms[arm_id] += 1
            # play arm and observe reward
            reward = self.arms[arm_id].play()
            # update parameter of bandit algorithm
            self.algorithm.update_param(arm_id, reward)
            # output
            print(
                f"iteration: {t} reward: {reward} selected arm: {arm_id}, arms: {selected_arms}"
            )
            self.algorithm.plot_distributions()
            t += 1
            if t > self.play_num:
                break
