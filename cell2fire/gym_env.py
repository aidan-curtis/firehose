import os
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from gym import Env, spaces
from gym.spaces import Discrete, Box

from cell2fire.firehose.config import training_enabled
from firehose.models import ExperimentHelper, IgnitionPoints, IgnitionPoint
from firehose.process import Cell2FireProcess
from firehose.utils import wait_until_file_populated

ENVS = []
_MODULE_DIR = os.path.dirname(os.path.realpath(__file__))


# Note: use RGB here, render method will convert to BGR for gym
_FIRE_COLOR = [255, 0, 0]  # red
_HARVEST_COLOR = [165, 42, 42]  # brown


def fire_size_reward(state, forest, scale=10):
    idxs = np.where(state > 0)
    return -len(idxs[0]) / (forest.shape[0] * forest.shape[1]) * scale


class FireEnv(Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(
        self,
        fire_map: str = "Harvest40x40",
        action_type: str = "flat",
        observation_type: str = "forest",
        max_steps: int = 200,
        ignition_points: Optional[IgnitionPoints] = None,
        reward_func=fire_size_reward,
        num_ignition_points: int = 1,  # if ignition_points is specified this is ignored
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

        self.action_type = action_type
        # Flat discrete action space
        if self.action_type == "flat":
            self.action_space = Discrete(
                self.forest_image.shape[0] * self.forest_image.shape[1]
            )
        elif self.action_type == "xy":
            self.action_space = Box(low=0, high=1, shape=(2,))
        else:
            raise NotImplementedError

        self.observation_type = observation_type
        if self.observation_type == "forest":
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(self.forest_image.shape[0], self.forest_image.shape[1]),
                dtype=np.uint8,
            )
        elif self.observation_space == "time":
            # Blind model
            self.observation_space = spaces.Box(
                low=0, high=max_steps + 1, shape=(1,), dtype=np.uint8,
            )

        self.state = np.zeros((self.forest_image.shape[0], self.forest_image.shape[1]))

        # Reward function
        self.reward_func = reward_func

        # NOTE: this should be the last thing that is done after we have set all the
        #  relevant properties in this instance
        # Cell2Fire Process
        self.fire_process = Cell2FireProcess(self)

    def step(self, action, debug: bool = False):
        """
        Step in the environment

        :param action: index of cell to harvest, using 0 indexing
        :param debug:
        """
        if debug:
            print(f"=== Step {self.iter} ===")
            print(f"Action: {action}")

        if self.action_type == "xy":
            # Action is an x/y vector, so convert to integer
            # TODO: Check if this is the correct x/y order to flatten
            x = int(action[0] * self.forest_image.shape[0])
            y = int(action[1] * self.forest_image.shape[1])
            min_action = self.forest_image.shape[0] * self.forest_image.shape[1] - 1
            action = min(x * self.forest_image.shape[1] + y, min_action)
        elif self.action_type == "flat":
            action = action
        else:
            raise NotImplementedError

        # IMPORTANT! Actions must be indexed from 0. The Cell2FireProcess class will
        # handle the indexing when calling Cell2Fire
        self.fire_process.apply_actions(action, debug)

        state_file = self.fire_process.read_line()
        if debug:
            print("State file:", state_file)
        if not state_file.endswith(".csv"):
            print("Proc Error. Resetting state")
            return self.state, self.reward_func(self.state, self.forest_image), True, {}

        # Bad Hack
        wait_until_file_populated(state_file)
        df = pd.read_csv(state_file, sep=",", header=None)
        self.state = df.values

        # Progress fire process to next state
        self.fire_process.progress_to_next_state(debug)

        # Check if we've exceeded max steps or Cell2Fire finished simulating
        done = self.iter >= self.max_steps or self.fire_process.finished
        if not debug and not training_enabled():
            print(f"\rStep {self.iter}", end="")
            if done:
                print()

        self.iter += 1

        return_state = self.iter if self.observation_type == "time" else self.state
        return return_state, self.reward_func(return_state, self.forest_image), done, {}

    def render(self, mode="human", **kwargs):
        """Render the geographic image and fire"""
        if mode not in {"human", "rgb_array"}:
            raise NotImplementedError(f"Only human mode is supported. Not {mode}")

        # Scale to 255
        im = (self.forest_image * 255).astype("uint8")

        # Set fire cells
        fire_idxs = np.where(self.state > 0)
        im[fire_idxs] = _FIRE_COLOR

        # Set harvest cells
        harvest_idxs = np.where(self.state < 0)
        im[harvest_idxs] = _HARVEST_COLOR

        # Scale to be larger
        im = cv2.resize(
            im, (im.shape[1] * 4, im.shape[0] * 4), interpolation=cv2.INTER_AREA
        )

        if mode == "human":
            # Flip RGB to BGR as cv2 uses the latter
            im = im[:, :, ::-1]
            cv2.imshow("Fire", im)
            cv2.waitKey(10)
        else:
            return im

    def reset(self, **kwargs):
        """Reset environment and restart process"""
        self.iter = 0
        self.state = np.zeros((self.forest_image.shape[0], self.forest_image.shape[1]))
        # Kill and respawn Cell2Fire process
        self.fire_process.reset(kwargs.get("debug", False))
        # return self.state

        return self.iter if self.observation_type == "time" else self.state


def main(debug: bool, **env_kwargs):
    # TODO(willshen): allow environment to be parallelized
    env = FireEnv(**env_kwargs)
    env.render()

    _ = env.reset(debug=debug)

    done = False
    idx = 0
    while not done:
        action = env.action_space.sample()
        idx += 1
        try:
            state, reward, done, info = env.step(action, debug=debug)
        except Exception as e:
            print(e)
            env.fire_process.kill()
            return

        env.render()
        # if done:
        #     state = env.reset()

    # input("Press Enter to finish...")
    print("Finished!")


if __name__ == "__main__":
    # main(debug=True, max_steps=1000)
    main(debug=True, ignition_points=IgnitionPoints([IgnitionPoint(1459, 1)]))
