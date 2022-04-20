import argparse
import json
import os
from typing import Optional

from sb3_contrib import TRPO
from stable_baselines3 import A2C, DQN, PPO

from cell2fire.gym_env import FireEnv
from firehose.baselines import (
    HumanInputAlgorithm,
    NaiveAlgorithm,
    NoAlgorithm,
    RandomAlgorithm,
)
from firehose.models import IgnitionPoint, IgnitionPoints
from firehose.video_recorder import FirehoseVideoRecorder

# Map name to ignition point and steps before simulation and steps per action
MAP_TO_IGNITION_POINTS = {
    "Sub40x40": IgnitionPoints(points=[IgnitionPoint(idx=1503, year=1, x=22, y=37)])
}
MAP_TO_EXTRA_KWARGS = {"Sub40x40": {"steps_before_sim": 30, "steps_per_action": 3}}

# Algorithms we support
SB3_ALGO_TO_MODEL_CLASS = {
    "a2c": A2C,
    "ppo": PPO,
    "trpo": TRPO,
    "dqn": DQN,
}
NO_MODEL_ALGO_TO_CLASS = {
    "random": RandomAlgorithm,
    "naive": NaiveAlgorithm,
    "human": HumanInputAlgorithm,
    "none": NoAlgorithm,
}

SUPPORTED_ALGOS = list(SB3_ALGO_TO_MODEL_CLASS.keys()) + list(
    NO_MODEL_ALGO_TO_CLASS.keys()
)


def _get_model(algo: str, model_path: Optional[str], env: FireEnv):
    """Get the model for the given algorithm."""
    if algo in NO_MODEL_ALGO_TO_CLASS:
        return NO_MODEL_ALGO_TO_CLASS[algo](env)
    elif algo in SB3_ALGO_TO_MODEL_CLASS:
        if not os.path.exists(model_path):
            raise ValueError(f"Model path {model_path} does not exist")
        return SB3_ALGO_TO_MODEL_CLASS[algo].load(model_path)
    else:
        raise NotImplementedError(f"Algo {algo} not supported")


def main(args):
    # Supercloud has TMPDIR so use that if it exists
    outdir = os.environ["TMPDIR"] if "TMPDIR" in os.environ.keys() else args.output_dir

    # TODO: cleaner way of specifying max steps and ignition points
    # TODO: support random ignition points
    env = FireEnv(
        action_type=args.action_space,
        fire_map=args.map,
        output_dir=outdir,
        max_steps=500,
        ignition_points=MAP_TO_IGNITION_POINTS.get(args.map, None),
        # verbose=True,
        **MAP_TO_EXTRA_KWARGS.get(
            args.map, {"steps_before_sim": 50, "steps_per_action": 10}
        ),
    )

    # Get the model for the algorithm and setup video recorder
    model = _get_model(algo=args.algo, model_path=args.model_path, env=env)
    video_recorder = FirehoseVideoRecorder(
        env, algo=args.algo, disable_video=args.disable_video
    )

    # Run policy until the end of the episode
    obs = env.reset()
    env.render()

    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        video_recorder.capture_frame()

    env.close()
    video_recorder.close()
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-al",
        "--algo",
        default="naive",
        help="Specifies the RL algorithm to use",
        choices=set(SUPPORTED_ALGOS),
    )
    parser.add_argument(
        "-m",
        "--map",
        default="Sub40x40",
        help="Specifies the map to run the environment in",
    )
    parser.add_argument(
        "-p",
        "--model_path",
        default="vectorize_model_2022-04-20_15-53-19/a2c_final.zip",
        help="Specifies the path to the model to evaluate",
    )
    parser.add_argument(
        "-as", "--action_space", default="flat", help="Action space type"
    )
    parser.add_argument(
        "-d", "--disable-video", action="store_true", help="Disable video recording"
    )
    parser.add_argument(
        "-i",
        "--ignition_type",
        default="fixed",
        help="Specifies whether to use a random or fixed fire ignition point",
        choices={"fixed", "random"},
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default=os.path.dirname(os.path.realpath(__file__)),
        help="Specifies the output directory for the simulation",
    )
    print("Args:", json.dumps(vars(parser.parse_args()), indent=2))
    # Uncomment to print out lots of stuff
    # set_debug_mode(True)

    main(args=parser.parse_args())
