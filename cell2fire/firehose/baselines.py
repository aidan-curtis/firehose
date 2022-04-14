from abc import ABC, abstractmethod

from typing import Tuple, Any, Set

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

        self.prev_actions: Set[int] = set()

        assert (
            len(env.ignition_points.points) == 1
        ), "Only 1 ignition point supported for naive algorithm"
        self.ignition_point = env.ignition_points.points[0]

    def predict(self, obs, **kwargs) -> Tuple[Any, Any]:
        cells_on_fire = obs == 1

        # No cells on fire, just return a random action
        if np.sum(cells_on_fire) == 0:
            return self.env.action_space.sample(), None
        else:
            # Find the cell closest on fire to the ignition point that has not
            # already had an action taken
            raise NotImplementedError
