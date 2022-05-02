from abc import ABC, abstractmethod
from typing import Any, Set, Tuple

import numpy as np
from gym_env import FireEnv


class Algorithm(ABC):
    """
    Just a wrapper around what stablebaselines approach
    support so we can reuse the structure without it being a hassle
    """

    def __init__(self, env: FireEnv):
        self.env = env
        self.policy = None

    def learn(self, **kwargs):
        print("Nothing to learn")

    def save(self, **kwargs):
        print("Nothing to save")

    @abstractmethod
    def predict(self, obs, **kwargs) -> Tuple[Any, Any]:
        pass


class RandomAlgorithm(Algorithm):
    """Randomly select actions with duplicates"""

    def predict(self, obs, **kwargs) -> Tuple[Any, Any]:
        return self.env.action_space.sample(), None


class FlatActionSpaceAlgorithm(Algorithm, ABC):
    def __init__(self, env: FireEnv):
        if env.action_type != "flat":
            raise ValueError(
                f"{self.__class__.__name__} only supports flat action space"
            )
        super().__init__(env)


class NoAlgorithm(FlatActionSpaceAlgorithm):
    """Algorithm that is basically no-op"""

    def predict(self, obs, **kwargs) -> Tuple[Any, Any]:
        return -1, None


class HumanInputAlgorithm(FlatActionSpaceAlgorithm):
    """Read in human input from stdin, indexed from 0! (not 1)"""

    def predict(self, obs, **kwargs) -> Tuple[Any, Any]:
        human_actions_str = input("Input actions:")
        human_actions = [int(act) for act in human_actions_str.split()]
        assert len(human_actions) == 1, "Only one action supported right now"
        return human_actions[0], None


class NaiveAlgorithm(FlatActionSpaceAlgorithm):
    """
    The Naive algorithm selects the cell that is on fire that is closest to the
    ignition point in terms of Euclidean distance if use_min is True.

    If use_min is False, then we select the point furthest from the ignition.
    This is our Frontier baseline essentially.

    If cells have already been put out, then it will not consider them.
    If there are no cells on fire, then it will return -1 (no-op).
    """

    def __init__(self, env: FireEnv):
        super().__init__(env)
        assert (
            len(env.ignition_points.points) == 1
        ), "Only 1 ignition point supported for naive algorithm"

        self.prev_actions: Set[int] = {-1}
        self.ignition_point = self.env.ignition_points.points[0]
        self.ignition_point_yx = self.env.flatten_idx_to_yx[self.ignition_point.idx - 1]
        self.use_min = True

    def _update_ignition_point(self):
        """Update ignition point if it has changed, indicating a reset in the environment"""
        current_ignition_point = self.env.ignition_points.points[0]

        if current_ignition_point != self.ignition_point:
            print(
                "NaiveAlgorithm: Ignition point changed from "
                f"{self.ignition_point} to {current_ignition_point}"
            )
            self.ignition_point = current_ignition_point
            # subtract 1 to convert to 0-indexed
            self.ignition_point_yx = self.env.flatten_idx_to_yx[
                current_ignition_point.idx - 1
            ]

            # Reset prev actions
            self.prev_actions = {-1}

    def predict(self, obs, **kwargs) -> Tuple[Any, Any]:
        # Important! Check if ignition point has changed, indicating reset in env
        self._update_ignition_point()

        # Find the cell closest on fire to the ignition point that has not
        # already had an action taken
        cells_on_fire = self.env.state == 1
        fire_yx = list(zip(*np.where(cells_on_fire)))
        dist = [
            np.linalg.norm(np.array(yx) - np.array(self.ignition_point_yx))
            for yx in fire_yx
        ]

        if not dist:
            # No cells on fire so no-op
            return -1, None

        if self.use_min:
            # Choose closest cell on fire
            closest_idx = np.argmin(dist)
        else:
            # Choose furthest cell on fire
            closest_idx = np.argmax(dist)

        chosen_fire_yx = fire_yx[closest_idx]
        chosen_fire_idx = self.env.yx_to_flatten_idx[chosen_fire_yx]

        # Check if this cell has already been chosen
        chosen_action_is_ignition = chosen_fire_idx == self.ignition_point.idx - 1
        action_is_no_op = chosen_fire_idx == -1
        if (
            chosen_fire_idx in self.prev_actions
            and not action_is_no_op
            and not chosen_action_is_ignition
        ):
            print("Chosen:", chosen_fire_idx)
            print("Prev Actions:", self.prev_actions)
            # This really shouldn't happen but if it happens just remove the exception
            # raise RuntimeError("Unexpected case where a fire put out has recaught fire")
            print("WARNING! Unexpected case where a fire put out has recaught fire")

        self.prev_actions.add(chosen_fire_idx)
        return chosen_fire_idx, None


class HumanExpertAlgorithm(NaiveAlgorithm):
    def __init__(self, env: FireEnv):
        super().__init__(env)
        self.use_min = False
