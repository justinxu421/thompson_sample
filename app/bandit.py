from collections import defaultdict

from app.gaussian_prior import Prior, ThompsonSamplingGaussianPrior


class Bandit:
    def __init__(self, arms, play_num, manual=False):
        self.t = 0
        self.total = 0
        self.selected_arms = defaultdict(int)
        self.arms = arms
        self.play_num = play_num
        self.manual = manual
        self.algorithm = ThompsonSamplingGaussianPrior(arms)

    def on_click(self, arm_id):
        if self.t > self.play_num:
            return

        print("arm id", arm_id)
        # select arm
        arm = self.arms[arm_id]
        self.selected_arms[arm.name] += 1

        # play arm and observe reward
        reward = arm.play()
        arm_id = self.arms.index(arm)
        self.total += reward

        # update parameter of bandit algorithm
        self.algorithm.update_param(arm_id, reward)
        reward_string = f"\nSelected arm: {arm.name}, id: {arm_id} iteration: {self.t} reward: {reward} \n\n Total Reward: {self.total}\n"
        self.t += 1
        self.algorithm.plot_distributions(self.on_click, reward_string)

    def experiment(self):
        priors = [
            Prior(10.0, 2.0),
            Prior(11.0, 1.0),
            Prior(12.0, 1.0),
            Prior(5.0, 0.0),
            Prior(9.0, 2.0),
        ]
        self.algorithm.initialize(priors)
        self.algorithm.plot_distributions(self.on_click)
