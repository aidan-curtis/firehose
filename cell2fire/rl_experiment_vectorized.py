from datetime import datetime

from stable_baselines3 import PPO, A2C, TRPO
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
    tf_logdir="./tmp/ppo_static_vectorized",
):
    model_save_dir = f'./vectorize_model_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    print("Saving checkpoints to", model_save_dir)
    print("Total timesteps:", total_timesteps)
    print("Checkpoint freq:", checkpoint_save_freq)

    if(args.ignition_type == "fixed"):
        env_with_fixed_ignition = lambda: FireEnv(
            ignition_points=IgnitionPoints([IgnitionPoint(1100, 1)]),
            action_type = args.action_space,
            fire_map = args.map
        )
    elif(args.ignition_type == "random"):
        env_with_fixed_ignition = lambda: FireEnv(
            action_type = args.action_space, 
            fire_map = args.map
        )
    else:
        raise NotImplementedError

    # Need to use use SubprocVecEnv so its parallelized. DummyVecEnv is sequential on a single core
    env = make_vec_env(
        env_with_fixed_ignition, n_envs=num_cpu, vec_env_cls=SubprocVecEnv
    )

    # model = DDPG("MlpPolicy", env, verbose=1, tensorboard_log="./tmp/ddpg_static_7")
    tf_logdir = f'{tf_logdir}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

    if(args.algo == "ppo"):
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=tf_logdir)
    elif(args.algo == "a2c")
        model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=tf_logdir)
    elif(args.algo == "trpo"):
        model = TRPO("MlpPolicy", env, verbose=1, tensorboard_log=tf_logdir)
    else:
        raise NotImplementedError


    print("Tensorboard logdir:", tf_logdir)
    # model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="./tmp/dqn_static_7")
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
        eval_env = FireEnv(ignition_points=IgnitionPoints([IgnitionPoint(1100, 1)]))
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
    parser.add_argument("-a", "--algo", default="ppo", help="Specifies the RL algorithm to use")
    parser.add_argument("-a", "--map", default="20x20", help="Specifies the map to run the environment in")
    parser.add_argument("-i", "--ignition_type", default="fixed", help="Specifies whether to use a random or fixed fire ignitinon point")
    parser.add_argument("-p", "--preharvest", default="fixed", help="Specifies whether or not to harvest before fire ignition")
    parser.add_argument("-s", "--seed", default="0", help="RL seed")
    args = parser.parse_args()
    main(args)
