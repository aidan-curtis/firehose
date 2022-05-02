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

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        raise NotImplementedError

    @abstractmethod
    def __call__(self) -> float:
        raise NotImplementedError


class FireSizeReward(Reward):
    """Number of cells on fire essentially"""

    @classmethod
    def name(cls) -> str:
        return "FireSizeReward"

    def __call__(self, scale: float = 10, reward_mask=None) -> float:
        """-(num cells on fire) / (total num cells in forest) * scale"""
        assert self.env.state.shape == self.env.forest_image.shape[:2]
        # print(reward_mask)
        if reward_mask is None:
            fire_idxs = np.where(self.env.state > 0)
        else:
            fire_idxs = np.where(self.env.state[reward_mask] > 0)

        # print(-len(fire_idxs[0]) / self.env.num_cells * scale)
        return -len(fire_idxs[0]) / self.env.num_cells * scale
