import argparse
import json
import os
from datetime import datetime

from gym.wrappers.monitoring.video_recorder import VideoRecorder
from sb3_contrib import TRPO
from stable_baselines3 import A2C, PPO

from cell2fire.gym_env import FireEnv
from firehose.baselines import HumanInputAlgorithm, NaiveAlgorithm, RandomAlgorithm
from firehose.config import set_debug_mode

_NO_MODEL_ALGOS = {"random", "naive", "human"}


def main(args):
    outdir = (
        os.environ["TMPDIR"]
        if "TMPDIR" in os.environ.keys()
        else os.path.dirname(os.path.realpath(__file__))
    )

    env = FireEnv(
        action_type=args.action_space,
        fire_map=args.map,
        output_dir=outdir,
        max_steps=500,
        # ignition_points=IgnitionPoints([IgnitionPoint(370, 1)]),
        steps_before_sim=50,
        steps_per_action=5,
        verbose=True,
    )

    if args.algo not in _NO_MODEL_ALGOS:
        assert args.model_path, f"Model path is required for alg={args.algo}"

    if args.algo == "ppo":
        model = PPO.load(args.model_path)
    elif args.algo == "a2c":
        model = A2C.load(args.model_path)
    elif args.algo == "trpo":
        model = TRPO.load(args.model_path)
    elif args.algo == "random":
        model = RandomAlgorithm(env)
    elif args.algo == "naive":
        model = NaiveAlgorithm(env)
    elif args.algo == "human":
        model = HumanInputAlgorithm(env)
    else:
        raise NotImplementedError

    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if not os.path.exists("videos"):
        os.mkdir("videos")

    video_fname = f"videos/{args.algo}-{date_str}.mp4"
    if not args.disable_video:
        video_recorder = VideoRecorder(env, video_fname, enabled=True)

    obs = env.reset()
    env.render()

    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if not args.disable_video:
            video_recorder.capture_frame()

    env.close()

    if not args.disable_video:
        video_recorder.close()
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-al",
        "--algo",
        default="naive",
        help="Specifies the RL algorithm to use",
        choices={"human", "random", "naive", "ppo", "a2c", "trpo"},
    )
    parser.add_argument(
        "-m",
        "--map",
        default="Sub20x20",
        help="Specifies the map to run the environment in",
    )
    parser.add_argument(
        "-p",
        "--model_path",
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
    print("Args:", json.dumps(vars(parser.parse_args()), indent=2))
    set_debug_mode(True)

    main(args=parser.parse_args())
