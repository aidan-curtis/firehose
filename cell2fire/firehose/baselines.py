from abc import ABC, abstractmethod
from typing import Any, Set, Tuple

import numpy as np
from gym.spaces import Discrete
from gym_env import FireEnv


class Algorithm(ABC):
    """
    Just a wrapper around what stablebaselines approach
    support so we can reuse the structure without it being a hassle
    """

    def __init__(self, env: FireEnv):
        self.env = env

    def learn(self, **kwargs):
        print("Nothing to learn")

    def save(self, **kwargs):
        print("Nothing to save")

    @abstractmethod
    def predict(self, obs, **kwargs) -> Tuple[Any, Any]:
        pass


class RandomAlgorithm(Algorithm):
    def predict(self, obs, **kwargs) -> Tuple[Any, Any]:
        return self.env.action_space.sample(), None


class NaiveAlgorithm(Algorithm):
    def __init__(self, env: FireEnv):
        if not isinstance(env.action_space, Discrete):
            raise ValueError("Naive algorithm only supports discrete action spaces")
        super().__init__(env)

        self.prev_actions: Set[int] = {-1}

        assert (
            len(env.ignition_points.points) == 1
        ), "Only 1 ignition point supported for naive algorithm"
        self.ignition_point = env.ignition_points.points[0]

        height, width = env.forest_image.shape[:2]
        self.flatten_idx_to_yx = {
            x + y * width: (y, x) for x in range(width) for y in range(height)
        }
        self.flatten_yx_to_idx = {b: a for a, b, in self.flatten_idx_to_yx.items()}

        # subtract 1 to convert to 0-indexed
        self.ignition_point_yx = self.flatten_idx_to_yx[self.ignition_point.idx - 1]

    def predict(self, obs, **kwargs) -> Tuple[Any, Any]:
        cells_on_fire = obs == 1

        # No cells on fire, just return a random action
        if np.sum(cells_on_fire) <= 10:
            return 0, None
        else:
            # Find the cell closest on fire to the ignition point that has not
            # already had an action taken
            fire_yx = list(zip(*np.where(cells_on_fire)))
            dist = [
                np.linalg.norm(np.array(yx) - np.array(self.ignition_point_yx))
                for yx in fire_yx
            ]

            chosen_fire_idx = -1
            while (
                chosen_fire_idx == -1 or chosen_fire_idx == self.ignition_point.idx - 1
            ):
                if not dist:
                    return 0, None

                closest_idx = np.argmin(dist)
                chosen_fire_yx = fire_yx[closest_idx]
                chosen_fire_idx = self.flatten_yx_to_idx[chosen_fire_yx]
                del fire_yx[closest_idx]
                del dist[closest_idx]

            if chosen_fire_idx in self.prev_actions and chosen_fire_idx != -1:
                raise NotImplementedError(
                    "very unexpected case where a fire put out has recaught fire"
                )

            self.prev_actions.add(chosen_fire_idx)
            print("\n", chosen_fire_idx)
            return chosen_fire_idx, None
            # raise NotImplementedError
