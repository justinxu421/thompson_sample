import argparse

from app.bandit import Bandit
from app.normal_arm import NormalArm


def main():
    parser = argparse.ArgumentParser(
        description="Bandit Experiment of TS-Gaussian-Prior"
    )

    # setting of experiment
    parser.add_argument("--exp_num", type=int, default=1, help="Number of experiments")

    args = parser.parse_args()

    # define arms
    arms = [
        NormalArm("Implement RFC", 10.0, 3.0),
        NormalArm("Squash Bug", 8.0, 3.0),
        NormalArm("Implement Bug", -5, 10),
        NormalArm("Prepare Tech Talk", 20, 2),
    ]

    # define bandit algorithm
    bandit = Bandit(arms, 5000)

    # run experiment
    print("----------Run Exp----------")
    for i in range(args.exp_num):
        print("Run Exp" + str(i))
        # define bandit algorithm
        bandit.experiment()
        print("Finish Exp" + str(i))
        print("")


if __name__ == "__main__":
    main()
