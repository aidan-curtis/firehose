import os
import shutil
import time
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import pandas as pd
from gym import Env, spaces
from gym.spaces import Box, Discrete

from firehose.config import training_enabled
from firehose.helpers import ExperimentHelper, IgnitionPoints
from firehose.process import Cell2FireProcess
from firehose.rewards import FireSizeReward
from firehose.utils import wait_until_file_populated

_MODULE_DIR = os.path.dirname(os.path.realpath(__file__))


# Note: use RGB here, render method will convert to BGR for gym
_FIRE_COLOR = [255, 0, 0]  # red
_HARVEST_COLOR = [28, 163, 236]  # blue
_IGNITION_COLOR = [255, 0, 255]  # pink


def num_cells_on_fire(state):
    return np.sum(state > 0)


class FireEnv(Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    ACTION_TYPES: Set[str] = {"flat", "xy"}
    OBSERVATION_TYPES: Set[str] = {"forest_rgb", "forest", "time"}

    def __init__(
        self,
        fire_map: str = "Harvest40x40",
        action_type: str = "flat",
        observation_type: str = "forest",
        max_steps: int = 200,
        output_dir: str = _MODULE_DIR,
        ignition_points: Optional[IgnitionPoints] = None,
        reward_func_cls=FireSizeReward,
        num_ignition_points: int = 1,  # if ignition_points is specified this is ignored
        steps_before_sim: int = 0,
        steps_per_action: int = 1,
        action_diameter: int = 1,
        verbose: bool = False,
    ):
        """

        :param fire_map: name of Fire map to use, should be in the data/ folder
        :param action_type: flat or xy
        :param observation_type: time or forest
        :param max_steps: maximum number of steps
        :param output_dir: base output directory
        :param ignition_points: ignition points to use. If None, will generate random ones
        :param reward_func_cls: reward function class
        :param num_ignition_points: #ignition points to generate if not specified
        :param steps_before_sim: number of steps to run before allowing any actions
        :param steps_per_action: number of steps to run after each action
            (is not run after pre-run steps)
        :param action_diameter: diameter of action around its cell. Only 1,2,3 supported right now.
            Corresponds to 1x1, 2x2, 3x3 actions.
        :param verbose: verbose logging
        """
        # Current step idx
        self.iter = 0
        # Maximum number of steps, note simulation could end before this is reached
        self.max_steps = max_steps

        # Helper object
        self.helper = ExperimentHelper(
            base_dir=_MODULE_DIR, map=fire_map, output_dir=output_dir
        )

        # Image of forest which we overlay
        self.forest_image = self.helper.forest_image
        self.uforest_image = (self.forest_image * 255).astype("uint8")

        if self.helper.reward_data is None:
            self.reward_mask = None
        else:
            self.reward_mask = np.where(self.helper.reward_data > 0)

        self.height, self.width = self.forest_image.shape[:2]
        self.num_cells = self.height * self.width

        # Randomly generate ignition points if required
        if not ignition_points:
            # We can only have 1 ignition in a given year in Cell2Fire (supposedly)
            assert (
                num_ignition_points == 1
            ), "Only 1 ignition point supported at the moment"
            self.ignition_points = self.helper.generate_random_ignition_points(
                num_points=num_ignition_points,
            )
            self.generate_ignition_points = True
            self.num_ignition_points = num_ignition_points
        else:
            assert (
                len(ignition_points) == 1
            ), "We don't know what happens with multiple ignition points"
            self.ignition_points = ignition_points
            self.generate_ignition_points = False
            self.num_ignition_points = len(ignition_points)

        # Set action and observation spaces.
        self.action_type = action_type
        self._set_action_space()

        # Note that the observation space != underlying state which is used for rewards
        self.observation_type = observation_type
        self._set_observation_space()

        # Set initial state to be empty
        self.state = np.zeros((self.height, self.width), dtype=np.uint8)

        # Number of steps to progress simulation before applying any actions
        assert steps_before_sim >= 0
        self.steps_before_sim = steps_before_sim

        # Number of steps after each action to wait before taking another action
        assert steps_per_action >= 1, "Must have at least 1 step per action"
        self.steps_per_action = steps_per_action

        # Radius of action around cell. If 1 then just affects cell itself,
        # if 2, then it affects 3x3 area around cell, etc.
        assert action_diameter in {1, 2}, "Only 1 and 2 action radius supported"
        self.action_diameter = action_diameter

        # Verbose logging in gym env and subprocess
        self.verbose = verbose

        # Mapping of flattened index (0-indexed) to y,x coordinates (i.e. row and column)
        self.flatten_idx_to_yx: Dict[int, Tuple[int, int]] = {
            x + y * self.width: (y, x)
            for y in range(self.height)
            for x in range(self.width)
        }
        self.yx_to_flatten_idx: Dict[Tuple[int, int], int] = {
            yx: idx for idx, yx, in self.flatten_idx_to_yx.items()
        }
        min_yx, max_yx = np.array(min(self.yx_to_flatten_idx)), np.array(
            max(self.yx_to_flatten_idx)
        )
        self.max_dist = np.linalg.norm(max_yx - min_yx)

        # Note: Reward function. Call this at end of __init__ just so we're safe
        #  Reward function uses the env state, etc. to compute rewards.
        self.reward_func = reward_func_cls(self)

        # Counters - indexed by (y, x) coordinates for now
        self.cells_harvested: Set = set()  # cells harvested (i.e., actions)
        self.cells_burned: Set = set()  # cells burned or burning on fire)
        self.cells_on_fire: Set = set()  # cells currently on fire

        # Previous actions for action masking
        self.prev_actions: Set[int] = set()

        # Note: Cell2Fire Process. Call this at the end of __init__ once everything in
        #  env itself is setup!
        self.fire_process = Cell2FireProcess(env=self, verbose=verbose)

    def _set_action_space(self):
        if self.action_type == "flat":
            # Flat discrete action space from (0 to number of pixels - 1)
            # We use 0-indexing here. This can be very high dimensional for large maps
            # Discrete itself will use 0 to self.num_cells
            self.action_space = Discrete(self.num_cells)
        elif self.action_type == "xy":
            # Continuous action space for x and y. We round at evaluation time
            # Note: underlying it is y,x so it fits better with np array
            self.action_space = Box(low=0, high=1, shape=(2,))
        else:
            raise NotImplementedError(f"Unsupported action type {self.action_type}")

    def _set_observation_space(self):
        # Note: observation space is what is returned to the agent. We use the state to
        #  calculate rewards.
        if self.observation_type == "forest_rgb":
            # Forest as a RGB image
            # TODO: should we normalize the RGB image? At least divide by 255 so its [0, 1]
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(self.height, self.width, 3),
                dtype=np.uint8,
            )
        elif self.observation_type == "forest":
            # Forest as -1 (harvested), 0 (nothing), 1 (on fire)
            self.observation_space = spaces.Box(
                low=-1, high=1, shape=(self.height, self.width), dtype=np.uint8
            )
        elif self.observation_space == "time":
            # Blind model
            self.observation_space = spaces.Box(
                low=0,
                high=self.max_steps + 1,
                shape=(1,),
                dtype=np.uint8,
            )
        else:
            raise ValueError(f"Unsupported observation type {self.observation_type}")

    def get_observation(self):
        if self.observation_type == "forest_rgb":
            return self.get_painted_image()
        elif self.observation_type == "forest":
            return self.state
        elif self.observation_type == "time":
            return self.iter
        else:
            raise ValueError(f"Unsupported observation type {self.observation_type}")

    def _get_actions_in_radius(self, cell_idx: int) -> List[int]:
        """Collect the cells within the radius of the given cell."""
        y, x = self.flatten_idx_to_yx[cell_idx]

        if self.action_diameter == 2:
            # 2x2 where the given cell is the top-left entry
            yxs = [
                (y, x),
                (y, x + 1),
                (y + 1, x),
                (y + 1, x + 1),
            ]
        elif self.action_diameter == 3:
            # 3x3 with center at (y, x)
            yxs = [
                (y + 1, x + 1),
                (y + 1, x),
                (y + 1, x - 1),
                (y, x + 1),
                (y, x),
                (y, x - 1),
                (y - 1, x + 1),
                (y - 1, x),
                (y - 1, x - 1),
            ]
        else:
            raise ValueError(
                f"Unsupported action diameter {self.action_diameter}. Expected 2 or 3"
            )

        actions = [
            self.yx_to_flatten_idx[yx] for yx in yxs if yx in self.yx_to_flatten_idx
        ]
        assert len(actions) >= 1
        return actions

    def get_action(self, raw_action):
        if self.action_type == "xy":
            # Note: we call action type xy, but it is actually represented as a yx
            y, x = raw_action

            # Round continuous values to their closest integer
            # Subtract 1 as we use 0-indexing for actions in the gym env
            y = round(y * (self.height - 1))
            x = round(x * (self.width - 1))
            if (y, x) not in self.yx_to_flatten_idx:
                raise ValueError(
                    f"Invalid action {raw_action}. Not within bounds of forest image"
                )
            return self.yx_to_flatten_idx[(y, x)]
        elif self.action_type == "flat":
            # No-op doesn't have a radius
            if raw_action == -1 or self.action_diameter == 1:
                return int(raw_action)  # int is important as sometimes it is a np.int64
            else:
                # Need to consider cells in radius
                return self._get_actions_in_radius(raw_action)
        else:
            raise ValueError(f"Unsupported action type {self.action_type}")

    def action_masks(self):
        # https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html
        if self.action_type != "flat":
            raise ValueError("Only flat action type supported for action masks")

        # List of boolean masks for each action
        # True if an action hasn't been applied yet, False otherwise
        mask = [True] * self.action_space.n
        for action in self.prev_actions:
            if action == -1:
                continue
            mask[action] = False
        return mask

    def _update_counters(self):
        """Update the counters based on the current state of the forest"""
        harvested = set(zip(*np.where(self.state == -1)))
        on_fire = set(zip(*np.where(self.state == 1)))

        self.cells_harvested.update(harvested)
        self.cells_burned.update(on_fire)
        self.cells_on_fire = on_fire

        # Sanity check that the number of total cells burned/burning is geq to
        # number of cells on fire + number of cells we harvested
        if not len(self.cells_on_fire) + len(self.cells_harvested) >= len(
            self.cells_burned
        ):
            raise ValueError("#Cells burned < #Cells on fire + #Cells harvested")

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

        # IMPORTANT! Actions must be indexed from 0. The Cell2FireProcess class will
        # handle the indexing when calling Cell2Fire
        try:
            self.fire_process.apply_actions(action)
            if isinstance(action, int):
                self.prev_actions.add(action)
            elif isinstance(action, list):
                self.prev_actions.update(set(action))
            else:
                raise RuntimeError(f"Unknown action of type {type(action)}")
        except BrokenPipeError as e:
            print("Could not write actions to cell2fire process. Weird stuff going on!")
            print(e)
            self.fire_process.write_lines_to_log()
            csv_files = None
        else:
            # Progress fire process to next state
            csv_files = self.fire_process.progress_to_next_state()

        if not csv_files:
            # Haven't encountered this state yet so lmk if you do
            if self.fire_process.finished:
                print(
                    "WARNING! Fire process finished but no csv files. You broke the code!"
                )

            # No state and Cell2Fire didn't finish, so something went wrong
            print(action)
            print("CSV files are empty")
            # print("State file:", state_file)
            print("Proc Error. Resetting state")

            # FIXME: it seems like cell2fire process finishes when waiting for an
            #  input action on very rare occassions. We won't worry about this for now
            # raise NotImplementedError

            obs = self.get_observation()
            reward = self.reward_func(reward_mask=self.reward_mask, action=action)
            return obs, reward, True, {}
        else:
            # Use last CSV as that is most recent forest
            state_file = csv_files[-1]

        # Bad Hack
        # FIXME: if multiple CSVs are returned for each step, then we only take the
        #  first one while we want to take the last one.
        wait_until_file_populated(state_file)
        df = pd.read_csv(state_file, sep=",", header=None)
        self.state = df.values
        self._update_counters()

        reward = self.reward_func(reward_mask=self.reward_mask, action=action)

        # Check if we've exceeded max steps or Cell2Fire finished simulating
        done = self.iter >= self.max_steps or self.fire_process.finished
        if not self.verbose and not training_enabled():
            print(
                f"\rStep {self.iter + 1}/{self.max_steps}. "
                f"#Cells on Fire {num_cells_on_fire(self.state)}, ",
                f"Reward: {reward}",
                end="",
            )
            if done:
                print()

        self.iter += 1
        obs = self.get_observation()
        return obs, reward, done, {}

    def get_painted_image(self) -> np.ndarray:
        # Make copy of array as we modify it in-place
        im = np.copy(self.uforest_image)

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

        # Scale image to be larger for visualization purposes as it is tiny
        # (e.g. 20x20, 40x40)
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
            # This is for sb3 I think
            return im

    def reset(self, **kwargs):
        """Reset environment and restart process"""
        self.iter = 0
        self.state = np.zeros((self.height, self.width), dtype=np.uint8)

        # Reset ignition points - if random will set it to random
        old_ignition_points = self.ignition_points

        if self.generate_ignition_points:
            self.ignition_points = self.helper.generate_random_ignition_points(
                num_points=self.num_ignition_points,
            )

        # Overwrite ignition points CSV as that is how we communicate to cell2fire
        if self.ignition_points != old_ignition_points:
            self.helper.overwrite_ignition_points(self.ignition_points)

        # Reset counters
        self.cells_harvested = set()
        self.cells_burned = set()
        self.cells_on_fire = set()

        # Reset prev actions
        self.prev_actions = set()

        # Delete all files in the output directory so we don't run out of inodes
        # Don't do checks as we don't want to kill training just because this failed
        shutil.rmtree(self.helper.output_folder, ignore_errors=True)
        os.makedirs(self.helper.output_folder, exist_ok=True)

        # Kill and respawn Cell2Fire process
        self.fire_process.reset()

        return self.get_observation()

    def close(self):
        """Clean up after ourselves"""
        self.helper.teardown()
        super().close()


def main(debug: bool, delay_time: float = 0.0, **env_kwargs):
    env = FireEnv(**env_kwargs, verbose=debug)
    env.render()

    _ = env.reset()

    done = False
    num_steps = 0
    while not done:
        action = env.action_space.sample()
        # try:
        state, reward, done, info = env.step(action)
        num_steps += 1
        # except Exception as e:
        #     env.fire_process.kill()
        #     raise e

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
