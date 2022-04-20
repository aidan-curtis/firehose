import copy
import os
import random
import time
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from gym import Env, spaces
from gym.spaces import Box, Discrete

from firehose.config import training_enabled
from firehose.models import ExperimentHelper, IgnitionPoint, IgnitionPoints
from firehose.process import Cell2FireProcess
from firehose.utils import wait_until_file_populated

ENVS = []
_MODULE_DIR = os.path.dirname(os.path.realpath(__file__))


# Note: use RGB here, render method will convert to BGR for gym
_FIRE_COLOR = [255, 0, 0]  # red
_HARVEST_COLOR = [28, 163, 236]  # blue
_IGNITION_COLOR = [255, 0, 255]  # pink


def num_cells_on_fire(state):
    return np.sum(state > 0)


def fire_size_reward(state, forest, scale=10):
    assert state.shape == forest.shape[:2]
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
        steps_before_sim: int = 0,
        steps_per_action: int = 1,
        verbose: bool = False,
    ):
        """

        :param fire_map: name of Fire map to use, should be in the data/ folder
        :param action_type: flat or xy
        :param observation_type: time or forest
        :param max_steps: maximum number of steps
        :param output_dir: base output directory
        :param ignition_points: ignition points to use. If None, will generate random ones
        :param reward_func: reward function
        :param num_ignition_points: #ignition points to generate if not specified
        :param steps_before_sim: number of steps to run before allowing any actions
        :param steps_per_action: number of steps to run after each action
            (is not run after pre-run steps)
        :param verbose: verbose logging
        """
        self.iter = 0
        self.max_steps = max_steps

        # Helper code
        self.helper = ExperimentHelper(
            base_dir=_MODULE_DIR, map=fire_map, output_dir=output_dir
        )
        self.forest_image = self.helper.forest_image
        self.uforest_image = (self.forest_image * 255).astype("uint8")

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

        # Number of steps to progress simulation before applying any actions
        assert steps_before_sim >= 0
        self.steps_before_sim = steps_before_sim

        # Number of steps after each action to wait before taking another action
        assert steps_per_action >= 1, "Must have at least 1 step per action"
        self.steps_per_action = steps_per_action

        self.verbose = verbose

        height, width = self.forest_image.shape[:2]
        self.flatten_idx_to_yx = {
            x + y * width: (y, x) for x in range(width) for y in range(height)
        }
        self.flatten_yx_to_idx = {b: a for a, b, in self.flatten_idx_to_yx.items()}

        # Note: Cell2Fire Process. Call this at the end of __init__!
        self.fire_process = Cell2FireProcess(self, verbose)

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
                low=-1,
                high=1,
                shape=(self.forest_image.shape[0], self.forest_image.shape[1]),
                dtype=np.uint8,
            )
        elif self.observation_space == "time":
            # Blind model
            self.observation_space = spaces.Box(
                low=0, high=self.max_steps + 1, shape=(1,), dtype=np.uint8,
            )

    def get_observation(self):
        return (
            self.iter if self.observation_type == "time" else self.state # self.get_painted_image()
        )

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
            if raw_action == -1:
                # handle no-op separately
                return raw_action

            y, x = self.flatten_idx_to_yx[raw_action]
            yxs = [
                # (y + 1, x + 1),
                # (y + 1, x),
                # (y + 1, x - 1),
                # (y, x + 1),
                (y, x),
                # (y, x - 1),
                # (y - 1, x + 1),
                # (y - 1, x),
                # (y - 1, x - 1),
            ]
            actions = []
            for yx in yxs:
                if yx in self.flatten_yx_to_idx:
                    actions.append(self.flatten_yx_to_idx[yx])
            assert len(actions) >= 1
            return actions
        else:
            raise NotImplementedError(f"Unsupported action type {self.action_type}")

    def step(self, action):
        """
        Step in the environment

        :param action: index of cell to harvest, using 0 indexing
        """
        # Get action for action_type
        raw_action = action
        action = self.get_action(raw_action=raw_action)

        if self.verbose:
            print(f"=== Step {self.iter} ===")
            print(f"Action: {raw_action}")
            # if action != raw_action:
            print(f"Converted action: {action}")

        # Code crashes for some reason when action == ignition point
        # for ignition_point in self.ignition_points.points:
        #     if ignition_point.idx == action + 1:
        #         print("action selected is ignition point hmm")
        #         # action = random.choice([action + 1, action - 1])

        # IMPORTANT! Actions must be indexed from 0. The Cell2FireProcess class will
        # handle the indexing when calling Cell2Fire
        self.fire_process.apply_actions(action)

        # Progress fire process to next state
        csv_files = self.fire_process.progress_to_next_state()
        if not csv_files:
            assert (
                not self.fire_process.finished
            ), "Fire process finished but no csv files"
            # No state and Cell2Fire didn't finish, so something went wrong
            print(action)
            print("CSV files are empty")
            # print("State file:", state_file)
            print("Proc Error. Resetting state")
            raise NotImplementedError
            return_state = self.get_observation()
            return (
                return_state,
                self.reward_func(self.state, self.forest_image),
                True,
                {},
            )
        else:
            # Use last CSV as that is most recent forest
            state_file = csv_files[-1]

        # Bad Hack
        # FIXME: if multiple CSVs are returned for each step, then we only take the
        #  first one while we want to take the last one.
        wait_until_file_populated(state_file)
        df = pd.read_csv(state_file, sep=",", header=None)
        self.state = df.values

        reward = self.reward_func(self.state, self.forest_image)

        # Check if we've exceeded max steps or Cell2Fire finished simulating
        done = self.iter >= self.max_steps or self.fire_process.finished
        if not self.verbose and not training_enabled():
            print(
                f"\rStep {self.iter + 1}/{self.max_steps}. "
                f"#Fire {num_cells_on_fire(self.state)}, ",
                f"Reward: {reward}",
                end="",
            )
            if done:
                print()

        self.iter += 1
        return_state = self.get_observation()
        # Note: call reward_func with state not return state!!!
        return return_state, reward, done, {}

    def get_painted_image(self):
        im = copy.copy(self.uforest_image)

        # Set fire cells
        fire_idxs = np.where(self.state > 0)
        im[fire_idxs] = _FIRE_COLOR

        # Set harvest cells
        harvest_idxs = np.where(self.state < 0)
        im[harvest_idxs] = _HARVEST_COLOR

        # Set ignition point
        assert (
            len(self.ignition_points.points) == 1
        ), "Only one ignition point supported"
        ignition_point = self.ignition_points.points[0]
        im[ignition_point.y, ignition_point.x] = _IGNITION_COLOR

        return im

    def render(self, mode="human", scale_factor: int = 10, **kwargs):
        """Render the geographic image and fire"""
        if mode not in {"human", "rgb_array"}:
            raise NotImplementedError(f"Only human mode is supported. Not {mode}")

        im = self.get_painted_image()

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
            cv2.waitKey(20)
        else:
            return im

    def reset(self, **kwargs):
        """Reset environment and restart process"""
        self.iter = 0
        self.state = np.zeros(
            (self.forest_image.shape[0], self.forest_image.shape[1]), dtype=np.uint8
        )
        # Kill and respawn Cell2Fire process
        self.fire_process.reset()

        return self.get_observation()


def main(debug: bool, delay_time: float = 0.0, **env_kwargs):
    env = FireEnv(**env_kwargs, verbose=debug)
    env.render()

    _ = env.reset()

    done = False
    num_steps = 0
    while not done:
        action = env.action_space.sample()
        try:
            state, reward, done, info = env.step(action)
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
    for run_idx in range(1):
        print(f"=== Run {run_idx} ===")
        main(debug=True, delay_time=0.00, steps_before_sim=50, steps_per_action=5)
