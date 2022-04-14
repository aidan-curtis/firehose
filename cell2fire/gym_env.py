import os
import random
import time
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from gym import Env, spaces
from gym.spaces import Box, Discrete

from cell2fire.firehose.config import training_enabled
from firehose.models import ExperimentHelper, IgnitionPoint, IgnitionPoints
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
        output_dir: str = _MODULE_DIR,
        ignition_points: Optional[IgnitionPoints] = None,
        reward_func=fire_size_reward,
        num_ignition_points: int = 1,  # if ignition_points is specified this is ignored
    ):
        self.iter = 0
        self.max_steps = max_steps

        # Helper code
        self.helper = ExperimentHelper(
            base_dir=_MODULE_DIR, map=fire_map, output_dir=output_dir
        )
        self.forest_image = self.helper.forest_image

        # TODO: fix this
        # Randomly generate ignition points if required
        if not ignition_points:
            # We can only have 1 ignition in a given year in Cell2Fire (supposedly)
            assert (
                num_ignition_points == 1
            ), "Only 1 ignition point supported at the moment"
            self.ignition_points = self.helper.generate_random_ignition_points(
                num_points=num_ignition_points,
            )
        else:
            self.ignition_points = ignition_points

        # Set action and observation spaces
        self.action_type = action_type
        self._set_action_space()

        self.observation_type = observation_type
        self._set_observation_space()

        # Set initial state to be empty
        self.state = np.zeros(
            (self.forest_image.shape[0], self.forest_image.shape[1]), dtype=np.uint8
        )

        # Reward function
        self.reward_func = reward_func

        # Note: Cell2Fire Process. Call this at the end of __init__!
        self.fire_process = Cell2FireProcess(self)

    def _set_action_space(self):
        if self.action_type == "flat":
            # Flat discrete action space
            self.action_space = Discrete(
                (self.forest_image.shape[0] * self.forest_image.shape[1]) - 1
            )
        elif self.action_type == "xy":
            # Continuous action space for x and y. We round at evaluation time
            self.action_space = Box(low=0, high=1, shape=(2,))
        else:
            raise NotImplementedError(f"Unsupported action type {self.action_type}")

    def _set_observation_space(self):
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
                low=0,
                high=self.max_steps + 1,
                shape=(1,),
                dtype=np.uint8,
            )

    def get_observation(self):
        return self.iter if self.observation_type == "time" else self.state

    def get_action(self, raw_action):
        if self.action_type == "xy":
            # Action is an x/y vector, so convert to integer
            # TODO: Check if this is the correct x/y order to flatten
            x = int(raw_action[0] * self.forest_image.shape[0])
            y = int(raw_action[1] * self.forest_image.shape[1])
            min_action = self.forest_image.shape[0] * self.forest_image.shape[1] - 1

            action = min(x * self.forest_image.shape[1] + y, min_action)
            return action
        elif self.action_type == "flat":
            return raw_action
        else:
            raise NotImplementedError(f"Unsupported action type {self.action_type}")

    def step(self, action, debug: bool = False):
        """
        Step in the environment

        :param action: index of cell to harvest, using 0 indexing
        :param debug:
        """
        # Get action for action_type
        raw_action = action
        action = self.get_action(raw_action=raw_action)

        if debug:
            print(f"=== Step {self.iter} ===")
            print(f"Action: {raw_action}")
            if action != raw_action:
                print(f"Converted action: {action}")

        # Code crashes for some reason when action == ignition point
        for ignition_point in self.ignition_points.points:
            if ignition_point.idx == action + 1:
                action = random.choice([action + 1, action - 1])

        # IMPORTANT! Actions must be indexed from 0. The Cell2FireProcess class will
        # handle the indexing when calling Cell2Fire
        self.fire_process.apply_actions(action, debug)

        state_file = self.fire_process.read_line()

        if not state_file.endswith(".csv"):
            print(action)
            print("State file:", state_file)
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

        return_state = self.get_observation()
        return return_state, self.reward_func(return_state, self.forest_image), done, {}

    def render(self, mode="human", scale_factor: int = 4, **kwargs):
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
            im,
            (im.shape[1] * scale_factor, im.shape[0] * scale_factor),
            interpolation=cv2.INTER_AREA,
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
        self.state = np.zeros(
            (self.forest_image.shape[0], self.forest_image.shape[1]), dtype=np.uint8
        )
        # Kill and respawn Cell2Fire process
        self.fire_process.reset(kwargs.get("debug", False))
        return self.get_observation()


def main(debug: bool, delay_time: float = 0.0, **env_kwargs):
    env = FireEnv(**env_kwargs)
    env.render()

    _ = env.reset(debug=debug)

    done = False
    num_steps = 0
    while not done:
        action = env.action_space.sample()
        try:
            state, reward, done, info = env.step(action, debug=debug)
            num_steps += 1
        except Exception as e:
            env.fire_process.kill()
            raise e

        env.render()
        if delay_time > 0.0:
            time.sleep(delay_time)
        # if done:
        #     state = env.reset()

    # input("Press Enter to finish...")
    print(f"Finished! Num steps = {num_steps}")


if __name__ == "__main__":
    # main(debug=True, max_steps=1000)
    # main(debug=True, ignition_points=IgnitionPoints([IgnitionPoint(1459, 1)]))
    for _ in range(100):
        main(debug=False, delay_time=0.00)
