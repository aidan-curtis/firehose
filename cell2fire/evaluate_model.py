import argparse
import json
import os
from typing import Optional

import torch
from sb3_contrib import TRPO
from stable_baselines3 import A2C, DQN, PPO
from torch import softmax

from cell2fire.gym_env import FireEnv
from firehose.baselines import (
    HumanInputAlgorithm,
    NaiveAlgorithm,
    NoAlgorithm,
    RandomAlgorithm,
)
from firehose.helpers import IgnitionPoint, IgnitionPoints
from firehose.results import FirehoseResults
from firehose.video_recorder import FirehoseVideoRecorder

# Map name to ignition point and steps before simulation and steps per action
MAP_TO_IGNITION_POINTS = {
    "Sub40x40": IgnitionPoints(points=[IgnitionPoint(idx=1503, year=1, x=22, y=37)])
}
MAP_TO_EXTRA_KWARGS = {"Sub40x40": {"steps_before_sim": 50, "steps_per_action": 3}}

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
    assert args.num_iters >= 1, "Must have at least one evaluation iteration"
    if args.disable_render:
        assert args.disable_video, "Must disable video if rendering is disabled"

    # TODO: support these other args

    # Supercloud has TMPDIR so use that if it exists
    outdir = os.environ["TMPDIR"] if "TMPDIR" in os.environ.keys() else args.output_dir

    # TODO: cleaner way of specifying max steps and ignition points
    # TODO: support random ignition points
    env = FireEnv(
        action_type=args.action_space,
        fire_map=args.map,
        output_dir=outdir,
        max_steps=500,
        ignition_points=(
            MAP_TO_IGNITION_POINTS.get(args.map, None)
            if args.ignition_type == "fixed"
            else None
        ),
        action_radius=2,
        observation_type="forest_rgb",
        # verbose=True,
        **MAP_TO_EXTRA_KWARGS.get(
            args.map, {"steps_before_sim": 20, "steps_per_action": 3}
        ),
    )

    # Get the model for the algorithm and setup video recorder
    model = _get_model(algo=args.algo, model_path=args.model_path, env=env)
    video_recorder = FirehoseVideoRecorder(
        env, algo=args.algo, disable_video=args.disable_video
    )

    results = FirehoseResults.from_env(env, args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enable_action_masking = False

    # Run policy until the end of the episode
    for _ in range(args.num_iters):
        action_mask = torch.ones((env.action_space.n,)).to(device)

        obs = env.reset()
        if not args.disable_render:
            env.render()

        done = False
        reward = None
        while not done:
            action, _states = model.predict(obs, deterministic=False)

            # Masked softmax to get probabilities and take mode - use logits so its more stable
            if enable_action_masking:
                processed_obs, vectorized = model.policy.obs_to_tensor(obs)
                categorical_dist = model.policy.get_distribution(processed_obs)
                action_dist = categorical_dist.distribution.probs.squeeze()
                masked_action_dist = action_mask * action_dist

                # Renormalize the probabilities
                # new_action_dist = masked_action_dist / masked_action_dist.sum()
                # new_action_dist = softmax(masked_action_logits, dim=0)
                new_action = masked_action_dist.argmax().item()

                print("===", new_action, "===")
                obs, reward, done, info = env.step(new_action)
                action_mask[new_action] = 0
            else:
                obs, reward, done, info = env.step(action)


            if not args.disable_render:
                env.render()
            video_recorder.capture_frame()

        if reward is None:
            raise RuntimeError("Reward is None. This should not happen")

        results.append(
            reward=reward,
            cells_harvested=len(env.cells_harvested),
            cells_on_fire=len(env.cells_on_fire),
            cells_burned=len(env.cells_burned),
            sim_steps=env.iter,
            ignition_points=env.ignition_points,
        )

    env.close()
    video_recorder.close()

    # Write results to disk
    results.write_json()

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
        default="dogrib",
        help="Specifies the map to run the environment in",
    )
    parser.add_argument(
        "-p",
        "--model_path",
        default="cnn_a2c_final.zip",
        type=str,
        help="Specifies the path to the model to evaluate",
    )
    parser.add_argument(
        "-as",
        "--action_space",
        default="flat",
        help="Action space type",
        choices=FireEnv.ACTION_TYPES,
    )
    parser.add_argument(
        "--disable-video", action="store_true", help="Disable video recording"
    )
    parser.add_argument(
        "-d", "--disable-render", action="store_true", help="Disable cv2 rendering"
    )
    parser.add_argument(
        "-i",
        "--ignition_type",
        default="random",
        help="Specifies whether to use a random or fixed fire ignition point",
        choices={"fixed", "random"},
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default=os.path.dirname(os.path.realpath(__file__)),
        help="Specifies the output directory for the simulation",
    )
    parser.add_argument(
        "-n",
        "--num-iters",
        type=int,
        default=1,
        help="Number of iterations to evaluate",
    )
    print("Args:", json.dumps(vars(parser.parse_args()), indent=2))
    # Uncomment to print out lots of stuff
    # set_debug_mode(True)

    main(args=parser.parse_args())
