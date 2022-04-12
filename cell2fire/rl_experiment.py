from stable_baselines3 import PPO, DDPG, DQN
from cell2fire.gym_env import FireEnv
import os
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from firehose.models import ExperimentHelper, IgnitionPoints, IgnitionPoint


env = FireEnv(ignition_points=IgnitionPoints([IgnitionPoint(1100, 1)]))

# model = DDPG("MlpPolicy", env, verbose=1, tensorboard_log="./tmp/ddpg_static_7")
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tmp/ppo_static_7")
# model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="./tmp/dqn_static_7")

model.learn(total_timesteps=400000)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()
env.close()