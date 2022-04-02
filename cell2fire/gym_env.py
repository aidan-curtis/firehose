import os
import time
from typing import List, Optional

import cv2
import numpy as np
import pandas as pd
from gym import Env, spaces
from gym.spaces import Discrete, Box

from firehose.models import IgnitionPoint, ExperimentHelper, IgnitionPoints
from firehose.process import Cell2FireProcess

ENVS = []
_MODULE_DIR = os.path.dirname(os.path.realpath(__file__))


def fire_size_reward(state, forest, scale=10):
    idxs = np.where(state > 0)
    return -len(idxs[0]) / (forest.shape[0] * forest.shape[1]) * scale


class FireEnv(Env):
    def __init__(
        self,
        fire_map: str = "Harvest40x40",
        max_steps: int = 200,
        ignition_points: Optional[IgnitionPoints] = None,
        reward_func=fire_size_reward,
        num_ignition_points: int = 5,  # if ignition_points is specified this is ignored
    ):
        # TODO: Create the process with the input map
        self.iter = 0
        self.max_steps = max_steps

        # Helper code
        self.helper = ExperimentHelper(base_dir=_MODULE_DIR, map=fire_map)
        self.forest_image = self.helper.forest_image

        # Randomly generate ignition points if required
        if not ignition_points:
            self.ignition_points = self.helper.generate_random_ignition_points(
                num_points=num_ignition_points,
            )
        else:
            self.ignition_points = ignition_points

        self.action_space = Discrete(
            self.forest_image.shape[0] * self.forest_image.shape[1]
        )
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.forest_image.shape[0], self.forest_image.shape[1]),
            dtype=np.uint8,
        )
        self.state = np.zeros((self.forest_image.shape[0], self.forest_image.shape[1]))

        # Reward function
        self.reward_func = reward_func

        # NOTE: this should be the last thing that is done after we have set all the
        #  relevant properties in this instance
        # Cell2Fire Process
        self.fire_process = Cell2FireProcess(self)

    def step(self, action, debug: bool = True):
        # if debug:
        #     print(action, "step")

        result = ""
        q = 0
        while result != "Input action":
            result = self.fire_process.read_line()
            # print(result)
            # assert len(result)>0

        value = str(action + 1) + "\n"
        value = bytes(value, "UTF-8")
        self.fire_process.write_action(value)

        state_file = self.fire_process.read_line()
        # FIXME: is this necessary?
        time.sleep(0.005)
        df = pd.read_csv(state_file, sep=",", header=None)
        self.state = df.values

        done = self.iter >= self.max_steps
        self.iter += 1

        return self.state, self.reward_func(self.state, self.forest_image), done, {}

    def render(self, mode="human", **kwargs):
        """Render the geographic image and fire"""
        im = (self.forest_image * 255).astype("uint8")

        # Set fire cells
        idxs = np.where(self.state > 0)
        im[idxs] = [0, 0, 255]

        # Scale to be larger
        im = cv2.resize(
            im, (im.shape[1] * 4, im.shape[0] * 4), interpolation=cv2.INTER_AREA
        )
        cv2.imshow("Fire", im)
        cv2.waitKey(10)

    def reset(self, **kwargs):
        """Reset environment and restart process"""
        self.iter = 0
        self.state = np.zeros((self.forest_image.shape[0], self.forest_image.shape[1]))
        # Kill and respawn Cell2Fire process
        self.fire_process.reset()
        return self.state


def main(**env_kwargs):
    # TODO(willshen): allow environment to be parallelized
    env = FireEnv(**env_kwargs)
    state = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        env.render()
        # if done:
        #     state = env.reset()
    print("Finished!")


if __name__ == "__main__":
    main()
