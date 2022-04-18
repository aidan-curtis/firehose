from datetime import datetime

from stable_baselines3 import PPO, A2C, DQN
from sb3_contrib.trpo.trpo import TRPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv

from cell2fire.firehose.config import set_training_enabled
from cell2fire.gym_env import FireEnv
import os
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from firehose.models import ExperimentHelper, IgnitionPoints, IgnitionPoint
import argparse

from typing import Callable

# TODO: make this global variable better
set_training_enabled(True)
num_cpu = 16


def main(
    args,
    total_timesteps=2_000_000,
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
        ig_points = IgnitionPoints([IgnitionPoint(idx=200, year=1, x=0, y=0)])
        single_env = lambda: FireEnv(
            ignition_points=ig_points,
            action_type=args.action_space,
            fire_map=args.map,
            output_dir=outdir,
        )
    elif args.ignition_type == "random":
        single_env = lambda: FireEnv(
            action_type=args.action_space, fire_map=args.map, output_dir=outdir
        )
    else:
        raise NotImplementedError

    # Need to use use SubprocVecEnv so its parallelized. DummyVecEnv is sequential on a single core
    env = make_vec_env(single_env, n_envs=num_cpu, vec_env_cls=SubprocVecEnv)

    # model = DDPG("MlpPolicy", env, verbose=1, tensorboard_log="./tmp/ddpg_static_7")
    tf_logdir = f'{tf_logdir}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

    print(args.architecture)
    if args.algo == "ppo":
        model = PPO(args.architecture, env, verbose=1, tensorboard_log=tf_logdir)
    elif args.algo == "a2c":
        model = A2C(args.architecture, env, verbose=1, tensorboard_log=tf_logdir)
    elif args.algo == "trpo":
        model = TRPO(args.architecture, env, verbose=1, tensorboard_log=tf_logdir)
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

    print("Tensorboard logdir:", tf_logdir)

    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_save_freq, save_path=model_save_dir
    )

    ####
    try:
        model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback])
    except Exception as e:
        model.save(os.path.join(model_save_dir, "ppo_final.zip"))
        raise e

    model.save(os.path.join(model_save_dir, "ppo_final.zip"))
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
        "-al", "--algo", default="a2c", help="Specifies the RL algorithm to use"
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
    # parser.add_argument("-p", "--preharvest", default="fixed", help="Specifies whether or not to harvest before fire ignition")
    parser.add_argument(
        "-as", "--action_space", default="flat", help="Action space type"
    )
    parser.add_argument("-s", "--seed", default="0", help="RL seed")
    parser.add_argument(
        "-l", "--logdir", default="/home/gridsan/acurtis/firehosetmp", help="RL seed"
    )
    args = parser.parse_args()
    main(args)
