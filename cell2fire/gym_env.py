import os
import time
from typing import List, Optional

import cv2
import numpy as np
import pandas as pd
from gym import Env
from gym.spaces import Discrete, Box

from firehose.models import IgnitionPoint, ExperimentHelper
from firehose.process import Cell2FireProcess

ENVS = []
_MODULE_DIR = os.path.dirname(os.path.realpath(__file__))


class FireEnv(Env):
    def __init__(
        self,
        fire_map: str = "dogrib",
        max_steps: int = 200,
        ignition_points: Optional[List[IgnitionPoint]] = None,
    ):
        if not ignition_points:
            ignition_points = [IgnitionPoint()]
        # TODO: Create the process with the input map
        self.action_space = Discrete(3)
        self.observation_space = Box(low=np.array([0]), high=np.array([100]))
        self.iter = 0
        self.state = [0]
        self.max_steps = max_steps

        # Helper code
        self.helper = ExperimentHelper(base_dir=_MODULE_DIR, map=fire_map)
        self.forest_image = self.helper.load_forest_image()

        # Cell2Fire Process
        self.fire_process = Cell2FireProcess(self.helper)

        # TODO: pass these into the binary
        self.ignition_points = ignition_points

    def step(self, action, debug: bool = True):
        # if debug:
        #     print(action, "step")

        result = ""
        q = 0
        while result != "Input action":
            result = self.fire_process.read_line()
            # print(result)
            # assert len(result)>0

        value = str(action) + "\n"
        value = bytes(value, "UTF-8")
        self.fire_process.write_action(value)

        state_file = self.fire_process.read_line()
        # FIXME: is this necessary?
        time.sleep(0.01)
        df = pd.read_csv(state_file, sep=",", header=None)
        self.state = df.values

        done = self.iter >= self.max_steps
        info = {}
        reward = 0
        self.iter += 1

        return self.state, reward, done, info

    def render(self, **kwargs):
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
        self.state = [0]
        # Kill and respawn Cell2Fire process
        self.fire_process.reset()
        return self.state


def main(**env_kwargs):
    env = FireEnv(**env_kwargs)
    state = env.reset()
    for _ in range(500):
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        env.render()
        if done:
            state = env.reset()
    print("Finished!")


if __name__ == "__main__":
    main()
