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
    def __call__(self, **kwargs) -> float:
        raise NotImplementedError


class FireSizeReward(Reward):
    """Number of cells on fire essentially"""

    @classmethod
    def name(cls) -> str:
        return "FireSizeReward"

    def __call__(self, scale: float = 10, reward_mask=None, **kwargs) -> float:
        """-(num cells on fire) / (total num cells in forest) * scale"""
        assert self.env.state.shape == self.env.forest_image.shape[:2]
        # print(reward_mask)
        if reward_mask is None:
            fire_idxs = np.where(self.env.state > 0)
        else:
            fire_idxs = np.where(self.env.state[reward_mask] > 0)

        # print(-len(fire_idxs[0]) / self.env.num_cells * scale)
        return -len(fire_idxs[0]) / self.env.num_cells * scale


class WillShenReward(Reward):
    """Will Shen's reward function"""

    @classmethod
    def name(cls) -> str:
        return "WillShenReward"

    def __call__(self, scale: float = 10, action: int = -1, **kwargs) -> float:
        assert self.env.state.shape == self.env.forest_image.shape[:2]

        fire_idxs = np.array(np.where(self.env.state > 0)).T
        num_cells_on_fire = fire_idxs.shape[0] if fire_idxs.size != 0 else 0

        # Proportion of cells on fire
        fire_term = 1 - num_cells_on_fire / self.env.num_cells

        # Hack that will suffice for 2x2 and 3x3
        if isinstance(action, list):
            # Choose median biased to the left
            action = action[len(action) // 2]

        # Distance of action to closest cell on fire
        action_yx = np.array(self.env.flatten_idx_to_yx[action])

        if num_cells_on_fire > 0:
            dists_to_fire = np.linalg.norm(fire_idxs - action_yx, axis=1)
            min_dist_to_fire = np.min(dists_to_fire)
        else:
            # If no cells on fire, encourage choosing actions close to ignition point
            ignition_point = self.env.ignition_points.points[0]
            ignition_point_yx = np.array(
                self.env.flatten_idx_to_yx[ignition_point.idx - 1]
            )
            min_dist_to_fire = np.linalg.norm(ignition_point_yx - action_yx)

        # Penalize actions far away from fire
        action_dist_term = min_dist_to_fire / self.env.max_dist

        # % of cells on fire - min dist to fire / scale
        reward = fire_term - action_dist_term / scale
        return reward


REWARD_FUNCTIONS = {
    reward_cls.name(): reward_cls for reward_cls in (FireSizeReward, WillShenReward)
}
