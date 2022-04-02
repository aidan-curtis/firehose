from stable_baselines3 import PPO, DDPG, DQN
from cell2fire.gym_env import FireEnv
import os
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy


# TODO(Aidan): debug why RL isn't working, maybe on 3x3 grid
log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)

env = FireEnv()
env = Monitor(env, log_dir)

# model = PPO("MlpPolicy", env, verbose=1)
model = DQN("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=1000000)

# obs = env.reset()
# for i in range(1000):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     env.render()
#     if done:
#       obs = env.reset()
# env.close()