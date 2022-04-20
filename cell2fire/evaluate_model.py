import argparse
import json
import os
from datetime import datetime

from gym.wrappers.monitoring.video_recorder import VideoRecorder
from sb3_contrib import TRPO
from stable_baselines3 import A2C, PPO

from cell2fire.gym_env import FireEnv
from firehose.baselines import HumanInputAlgorithm, NaiveAlgorithm, RandomAlgorithm, NoAlgorithm
from firehose.config import set_debug_mode
from firehose.models import IgnitionPoints, IgnitionPoint

_NO_MODEL_ALGOS = {"random", "naive", "human", "none"}


MAP_TO_IGNITION_POINTS = {
    "Sub40x40": IgnitionPoints(points=[IgnitionPoint(idx=1503, year=1, x=22, y=37)])
}
MAP_TO_EXTRA_KWARGS = {
    "Sub40x40": {"steps_before_sim": 30, "steps_per_action": 3}
}


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
        ignition_points=MAP_TO_IGNITION_POINTS.get(args.map, None),
        verbose=True,
        **MAP_TO_EXTRA_KWARGS.get(args.map, {"steps_before_sim": 50, "steps_per_action": 10}),
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
    elif args.algo == "none":
        model = NoAlgorithm(env)
    else:
        raise NotImplementedError

    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if not os.path.exists("videos"):
        os.mkdir("videos")

    video_fname = f"videos/{date_str}-{args.algo}.mp4"
    if not args.disable_video:
        video_recorder = VideoRecorder(env, video_fname, enabled=True)

    obs = env.reset()
    env.render()

    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        print("Reward:", reward)
        env.render()
        if not args.disable_video:
            video_recorder.capture_frame()

    env.close()

    if not args.disable_video:
        video_recorder.close()
        os.remove(video_recorder.metadata_path)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-al",
        "--algo",
        default="a2c",
        help="Specifies the RL algorithm to use",
        choices={"human", "random", "naive", "none", "ppo", "a2c", "trpo"},
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
    print("Args:", json.dumps(vars(parser.parse_args()), indent=2))
    # set_debug_mode(True)

    main(args=parser.parse_args())
