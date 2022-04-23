import logging
import time

from gym.wrappers.monitoring.video_recorder import VideoRecorder
from stable_baselines3 import PPO

from stable_baselines3 import PPO

from cell2fire.gym_env import FireEnv
from firehose.helpers import IgnitionPoints, IgnitionPoint

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    eval_env = FireEnv(ignition_points=IgnitionPoints([IgnitionPoint(1100, 1)]))
    obs = eval_env.reset()
    video_recorder = VideoRecorder(eval_env, "random.mp4", enabled=True)

    for i in range(1000):
        action = eval_env.action_space.sample()
        obs, reward, done, info = eval_env.step(action)

        eval_env.render()
        video_recorder.capture_frame()

        print("\n", action, reward)
        time.sleep(0.025)
        if done:
            obs = eval_env.reset()
            break

    video_recorder.close()
    video_recorder.enabled = False
    eval_env.close()

