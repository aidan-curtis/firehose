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

    @abstractmethod
    def __call__(self) -> float:
        raise NotImplementedError


class FireSizeReward(Reward):
    """Number of cells on fire essentially"""

    def __call__(self, scale: float = 10) -> float:
        """-(num cells on fire) / (total num cells in forest) * scale"""
        assert self.env.state.shape == self.env.forest_image.shape[:2]
        fire_idxs = np.where(self.env.state > 0)
        return -len(fire_idxs[0]) / self.env.num_cells * scale
