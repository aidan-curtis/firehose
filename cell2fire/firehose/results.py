import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

from firehose.helpers import IgnitionPoint, IgnitionPoints
from firehose.utils import NumpyEncoder

if TYPE_CHECKING:
    # Can't be bothered fixing any circular imports we might get so put this here
    from gym_env import FireEnv


@dataclass
class FirehoseResults:
    # Algorithm, Map, Action space, etc.
    evaluation_args: Dict

    # Other parameters used for evaluation
    reward_function: str
    action_diameter: int
    steps_before_sim: int
    steps_per_action: int

    # Just for debug purposes
    evaluation_date: datetime = field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )

    # Each element in the lists below is the result of one iteration
    # Reward in the final state
    rewards: List[float] = field(default_factory=list)

    # Number of *unique* cells we harvested (i.e., applied an action to)
    num_cells_harvested: List[int] = field(default_factory=list)

    # Number of cells currently on fire at end of simulation
    num_cells_on_fire: List[int] = field(default_factory=list)

    # Number of cells burned or burning (i.e., on fire). This may happen
    # when simulation finishes early
    num_cells_burned: List[int] = field(default_factory=list)

    # Number of steps the simulation had. Probably not useful but nice to have
    num_sim_steps: List[int] = field(default_factory=list)

    # Specific ignition points used. If we're using random ignition this can
    # evolve over each iteration, hence we use a list.
    ignition_points: List[IgnitionPoints] = field(default_factory=list)

    @classmethod
    def from_env(cls, env: "FireEnv", args) -> "FirehoseResults":
        return cls(
            evaluation_args=args.__dict__,
            reward_function=env.reward_func.name(),
            # TODO: move the below into the evaluation args?
            action_diameter=env.action_diameter,
            steps_before_sim=env.steps_before_sim,
            steps_per_action=env.steps_per_action,
        )

    def __post_init__(self):
        # Run validation checks
        self._validate()

    def _validate(self):
        assert (
            len(self.rewards)
            == len(self.num_cells_harvested)
            == len(self.num_cells_on_fire)
            == len(self.num_cells_burned)
            == len(self.num_sim_steps)
            == len(self.ignition_points)
        )

    def append(
        self,
        reward: float,
        cells_harvested: int,
        cells_on_fire: int,
        cells_burned: int,
        sim_steps: int,
        ignition_points: IgnitionPoints,
    ):
        # Append the result of one simulation
        self.rewards.append(reward)
        self.num_cells_harvested.append(cells_harvested)
        self.num_cells_on_fire.append(cells_on_fire)
        self.num_cells_burned.append(cells_burned)
        self.num_sim_steps.append(sim_steps)
        self.ignition_points.append(ignition_points)
        self._validate()

    @property
    def num_iters(self) -> int:
        return len(self.num_cells_harvested)

    def to_json(self) -> str:
        json_dict = asdict(self)
        json_str = json.dumps(json_dict, cls=NumpyEncoder, indent=2)
        return json_str

    def write_json(
        self, json_fname: Optional[str] = None, results_dir: str = "evaluation_results"
    ) -> None:
        # Create an informative filename
        if not json_fname:
            json_fname = os.path.join(
                results_dir,
                f"{self.evaluation_date}-{self.evaluation_args['map']}-"
                f"{self.evaluation_args['algo']}-iters{self.num_iters}.json",
            )

        # Create results dir if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)

        with open(json_fname, "w") as f:
            f.write(self.to_json())

        print(f"Wrote results to {json_fname}")

    @classmethod
    def from_json(cls, json_str: str) -> "FirehoseResults":
        json_dict = json.loads(json_str)
        return cls(**json_dict)

    @classmethod
    def read_json(cls, json_fname: str) -> "FirehoseResults":
        with open(json_fname, "r") as f:
            return cls.from_json(f.read())
