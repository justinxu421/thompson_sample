import argparse

from app.bandit import Bandit
from app.normal_arm import NormalArm


def main():
    parser = argparse.ArgumentParser(
        description="Bandit Experiment of TS-Gaussian-Prior"
    )

    # setting of experiment
    parser.add_argument("--exp_num", type=int, default=1, help="Number of experiments")
    parser.add_argument("--manual", type=bool, default=False, help="Manually select experiments")

    args = parser.parse_args()

    # define arms
    arms = [
        NormalArm("Write RFC", 10.0, 3.0),
        NormalArm("Mentor Teammate", 15, 3.0),
        NormalArm("Squash Bug", 8.0, 3.0),
        NormalArm("Implement Bug", -5, 10),
        NormalArm("Chat with coworkers", 0, 3.0),
        NormalArm("Prepare Tech Talk", 20, 2),
    ]

    # define bandit algorithm
    bandit = Bandit(arms, 1000, args.manual)

    # run experiment
    print("Run Exp")
    bandit.experiment()
    print("Finish Exp")
    print("")


if __name__ == "__main__":
    main()
