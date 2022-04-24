from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    # Workaround for circular imports as I can't be bothered refactoring
    from gym_env import FireEnv


class Reward(ABC):
    """A reward function for a FireEnv. The FireEnv updates its own state so we rely on that to compute rewards."""

    def __init__(self, env: "FireEnv"):
        self.env = env

    def reset(self):
        pass

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        raise NotImplementedError

    @abstractmethod
    def __call__(self, action: int) -> float:
        raise NotImplementedError


class FireSizeReward(Reward):
    """Number of cells on fire essentially"""

    def __init__(self, env: "FireEnv"):
        super().__init__(env)
        self.prev_actions = set()

    @classmethod
    def name(cls) -> str:
        return "FireSizeReward"

    def reset(self):
        self.prev_actions = set()

    def __call__(self, action: int, scale: float = 10) -> float:
        """-(num cells on fire) / (total num cells in forest) * scale"""
        assert self.env.state.shape == self.env.forest_image.shape[:2]

        # Penalize same actions - this doesn't really work as it gives the wrong
        # reward signal I think
        # if action in self.prev_actions:
        #    return -999

        fire_idxs = np.where(self.env.state > 0)
        return -len(fire_idxs[0]) / self.env.num_cells * scale
