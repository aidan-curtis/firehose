import argparse
import os

from cell2fire.firehose.config import set_training_enabled
from cell2fire.gym_env import FireEnv
from firehose.baselines import NaiveAlgorithm, RandomAlgorithm

# TODO: make this global variable better
set_training_enabled(True)


def main(args):
    outdir = (
        os.environ["TMPDIR"]
        if "TMPDIR" in os.environ.keys()
        else os.path.dirname(os.path.realpath(__file__))
    )

    env = FireEnv(action_type=args.action_space, fire_map=args.map, output_dir=outdir)
    if args.algo == "random":
        model = RandomAlgorithm(env)
    elif args.algo == "naive":
        model = NaiveAlgorithm(env)
    else:
        raise NotImplementedError

    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()
    env.close()

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-al",
        "--algo",
        default="naive",
        help="Specifies the RL algorithm to use",
        choices={"random", "naive"},
    )
    parser.add_argument(
        "-m",
        "--map",
        default="Sub40x40",
        help="Specifies the map to run the environment in",
    )
    parser.add_argument(
        "-as", "--action_space", default="flat", help="Action space type"
    )
    parser.add_argument(
        "-i",
        "--ignition_type",
        default="fixed",
        help="Specifies whether to use a random or fixed fire ignitinon point",
        choices={"fixed", "random"},
    )
    args = parser.parse_args()
    main(args)
