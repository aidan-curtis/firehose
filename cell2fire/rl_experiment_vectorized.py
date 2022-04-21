from datetime import datetime

from stable_baselines3 import PPO, A2C, DQN
from sb3_contrib.trpo.trpo import TRPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv

from evaluate_model import MAP_TO_IGNITION_POINTS, MAP_TO_EXTRA_KWARGS, SUPPORTED_ALGOS
from firehose.config import set_training_enabled, set_debug_mode
from cell2fire.gym_env import FireEnv
import os
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from firehose.models import ExperimentHelper, IgnitionPoints, IgnitionPoint
import argparse
import torch as th
import gym
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
    create_mlp,
)
from stable_baselines3.common.preprocessing import (
    get_action_dim,
    is_image_space,
    maybe_transpose,
    preprocess_obs,
)
from torch import nn
from typing import Callable

# TODO: make this global variable better
set_training_enabled(True)
num_cpu = 16


class PaddedNatureCNN(BaseFeaturesExtractor):
    """
    CNN from DQN nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.
    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(PaddedNatureCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space, check_channels=False), (
            "You should use NatureCNN "
            f"only with images not with {observation_space}\n"
            "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
            "If you are using a custom environment,\n"
            "please check it using our env checker:\n"
            "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html"
        )
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


def main(
    args,
    total_timesteps=3_000_000,
    checkpoint_save_freq=int(2_000_000 / 100),
    should_eval=False,
):

    tf_logdir = args.logdir

    model_save_dir = f'./vectorize_model_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    print("Saving checkpoints to", model_save_dir)
    print("Total timesteps:", total_timesteps)
    print("Checkpoint freq:", checkpoint_save_freq)

    # Set the log directory to be a combination of the hyperparameters
    tf_logdir = "{}/{}_{}_{}_{}_{}".format(
        tf_logdir, args.algo, args.map, args.ignition_type, args.action_space, args.seed
    )
    outdir = (
        os.environ["TMPDIR"]
        if "TMPDIR" in os.environ.keys()
        else os.path.dirname(os.path.realpath(__file__))
    )

    if args.ignition_type == "fixed":
        ig_points = MAP_TO_IGNITION_POINTS[args.map]
        single_env = lambda: FireEnv(
            ignition_points=ig_points,
            action_type=args.action_space,
            action_radius=args.action_radius,
            fire_map=args.map,
            output_dir=outdir,
            **MAP_TO_EXTRA_KWARGS[args.map],
        )
    elif args.ignition_type == "random":
        single_env = lambda: FireEnv(
            action_type=args.action_space, 
            action_radius=args.action_radius,
            fire_map=args.map,
            output_dir=outdir
        )
    else:
        raise NotImplementedError

    # Need to use use SubprocVecEnv so its parallelized. DummyVecEnv is sequential on a single core
    env = make_vec_env(single_env, n_envs=num_cpu, vec_env_cls=SubprocVecEnv)

    # model = DDPG("MlpPolicy", env, verbose=1, tensorboard_log="./tmp/ddpg_static_7")
    tf_logdir = f'{tf_logdir}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

    print(args.architecture)
    if args.architecture == "CnnPolicy":
        model_kwargs = {"features_extractor_class": PaddedNatureCNN}
    else:
        model_kwargs = {}

    if args.algo == "ppo":
        # model = PPO(args.architecture, env, features_extractor_class = PaddedNatureCNN, verbose=1, tensorboard_log=tf_logdir, policy_kwargs=model_kwargs)
        model = PPO(
            args.architecture,
            env,
            verbose=1,
            tensorboard_log=tf_logdir,
            gamma=args.gamma,
            policy_kwargs=model_kwargs,
        )
    elif args.algo == "a2c":
        model = A2C(
            args.architecture,
            env,
            verbose=1,
            tensorboard_log=tf_logdir,
            gamma=args.gamma,
            policy_kwargs=model_kwargs,
        )
    elif args.algo == "trpo":
        model = TRPO(
            args.architecture,
            env,
            verbose=1,
            tensorboard_log=tf_logdir,
            gamma=args.gamma,
            policy_kwargs=model_kwargs,
        )
    elif args.algo == "random":
        model = RandomAlgorithm(
            args.architecture, env, verbose=1, tensorboard_log=tf_logdir
        )
    elif args.algo == "naive":
        model = NaiveAlgorithm(
            args.architecture, env, verbose=1, tensorboard_log=tf_logdir
        )
    elif args.algo == "dqn":
        model = DQN(args.architecture, env, verbose=1, tensorboard_log=tf_logdir)
    else:
        raise NotImplementedError

    # Example of how to load pre-trained model
    # model = A2C.load("vectorize_model_2022-04-20_13-30-27/a2c_final.zip", env, verbose=1, tensorboard_log=tf_logdir)
    print("Tensorboard logdir:", tf_logdir)

    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_save_freq, save_path=model_save_dir
    )

    ####
    try:
        model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback])
    except Exception as e:
        model.save(os.path.join(model_save_dir, f"{args.algo}_final.zip"))
        raise e

    model.save(os.path.join(model_save_dir, f"{args.algo}_final.zip"))
    #####
    env.close()

    # Create new env for evaluation that isn't vectorized
    if should_eval:

        if args.ignition_type == "fixed":
            eval_env = lambda: FireEnv(
                ignition_points=ig_points,
                action_type=args.action_space,
                fire_map=args.map,
            )
        elif args.ignition_type == "random":
            eval_env = lambda: FireEnv(action_type=args.action_space, fire_map=args.map)
        else:
            raise NotImplementedError(f"Unsupported ignition type {args.ignition_type}")

        obs = eval_env.reset()
        for i in range(1000):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            eval_env.render()
            if done:
                obs = eval_env.reset()
        eval_env.close()

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-al",
        "--algo",
        default="a2c",
        help="Specifies the RL algorithm to use",
        choices=SUPPORTED_ALGOS,
    )
    parser.add_argument(
        "-m",
        "--map",
        default="Sub40x40",
        help="Specifies the map to run the environment in",
    )
    parser.add_argument(
        "-ar",
        "--architecture",
        default="MlpPolicy",
        help="Specifies whether to use an MLP or CNN as the neural architecture for the agent",
    )
    parser.add_argument(
        "-i",
        "--ignition_type",
        default="fixed",
        help="Specifies whether to use a random or fixed fire ignitinon point",
        choices={"fixed", "random"},
    )
    parser.add_argument(
        "-as",
        "--action_space",
        default="flat",
        help="Action space type",
        choices=FireEnv.ACTION_TYPES,
    )
    parser.add_argument(
        "-g",
        "--gamma",
        default=0.99,
        type=float,
        help="Agent gamma"
    )
    parser.add_argument(
        "-ar",
        "--action_radius",
        default=1,
        type=int,
        help="Action radius"
    )
    parser.add_argument("-s", "--seed", default="0", help="RL seed")
    parser.add_argument(
        "-l", "--logdir", default="/home/gridsan/acurtis/firehosetmp", help="Logdir"
    )
    # parser.add_argument("-l", "--logdir", default="/tmp/firehose", help="RL seed")
    args = parser.parse_args()
    main(args)
