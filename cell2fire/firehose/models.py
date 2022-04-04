import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from functools import cached_property
from typing import List, ClassVar, Optional

import numpy as np

from utils.ReadDataPrometheus import Dictionary


_NO_FUEL_STR: str = "NFnfuel"


@dataclass(frozen=True)
class IgnitionPoint:
    idx: int
    year: int


@dataclass(frozen=True)
class IgnitionPoints:
    points: List[IgnitionPoint]

    # This is a class variable as it's fixed as input for
    # all ignition points. 0 is default used in parser.
    RADIUS: ClassVar[int] = 0
    CSV_NAME: ClassVar[str] = "Ignitions.csv"

    @property
    def year(self) -> int:
        year = [p.year for p in self.points]
        assert len(set(year)) == 1, "All ignition points must have the same year"
        return year[0]

    def get_csv(self) -> str:
        csv_list = ["Year,Ncell"]
        for point in self.points:
            csv_list.append(f"{point.year},{point.idx}")
        return "\n".join(csv_list)

    def write_to_csv(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.get_csv())


@dataclass(frozen=True)
class ExperimentHelper:
    base_dir: str
    map: str

    datetime_str: str = field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )

    @property
    def binary_path(self) -> str:
        return os.path.join(self.base_dir, "Cell2FireC/Cell2Fire")

    @property
    def data_folder(self) -> str:
        return "{}/../data/{}/".format(self.base_dir, self.map)

    @property
    def forest_datafile(self) -> str:
        return "{}/../data/{}/Forest.asc".format(self.base_dir, self.map)

    @property
    def output_folder(self) -> str:
        return "{}/../results/{}_{}/".format(self.base_dir, self.map, self.datetime_str)

    @property
    def tmp_input_folder(self):
        return "{}/../input/{}_{}/".format(self.base_dir, self.map, self.datetime_str)

    @cached_property
    def forest_data(self) -> np.ndarray:
        return np.loadtxt(self.forest_datafile, skiprows=6)

    @cached_property
    def fbp_lookup_dict(self) -> Dictionary:
        # Load the lookup table
        fbp_lookup = os.path.join(self.data_folder, "fbp_lookup_table.csv")
        fbp_dict = Dictionary(fbp_lookup)
        return fbp_dict

    @cached_property
    def forest_non_fuel(self) -> np.ndarray:
        """Array with 1 if non-fuel and 0 if fuel"""
        fuel_type_dict = self.fbp_lookup_dict[2]
        fuel_type_dict["-9999"] = _NO_FUEL_STR

        forest_non_fuels = np.zeros_like(self.forest_data)
        for x in range(forest_non_fuels.shape[0]):
            for y in range(forest_non_fuels.shape[1]):
                cell_id = str(int(self.forest_data[x, y]))
                forest_non_fuels[x, y] = fuel_type_dict[cell_id] == _NO_FUEL_STR
        return forest_non_fuels

    @cached_property
    def forest_image(self) -> np.ndarray:
        """Forest image in RGB (not BGR)"""
        # Load in the raw forest image
        forest_image_data = self.forest_data

        # Load color lookup dict
        color_dict = self.fbp_lookup_dict[1]
        color_dict["-9999"] = [0, 0, 0]

        # Apply lookup dict to raw forest data
        forest_image = np.zeros(
            (forest_image_data.shape[0], forest_image_data.shape[1], 3)
        )
        for x in range(forest_image_data.shape[0]):
            for y in range(forest_image_data.shape[1]):
                forest_image[x, y] = color_dict[str(int(forest_image_data[x, y]))][:3]

        return forest_image

    def manipulate_input_data_folder(
        self, ignition_points: Optional[IgnitionPoints] = None
    ):
        """
        Copy input data folder to somewhere new and write the ignition points we generated
        """
        # Copy input data folder to new one we generate
        tmp_dir = self.tmp_input_folder
        shutil.copytree(self.data_folder, tmp_dir)

        # Delete existing ignition points and write our ignition points
        if ignition_points:
            ignition_points_csv = os.path.join(tmp_dir, IgnitionPoints.CSV_NAME)
            # Only remove ignitions if it already exists
            if os.path.exists(ignition_points_csv):
                os.remove(ignition_points_csv)
            ignition_points.write_to_csv(ignition_points_csv)

        print(f"Copied modified input data folder to {tmp_dir}")

    def generate_random_ignition_points(
        self, num_points: int = 1, year: int = 1, radius: int = IgnitionPoints.RADIUS
    ) -> IgnitionPoints:
        """
        Generates random ignition points.

        :param num_points: number of ignition points to generate
        :param year: the year, we only support one year for now
        :param radius: the radius of the ignition point, default to 0
        :return: List of ignition points, which are tuples with
            (year, index of cell with the ignition point)
        """
        # Can only sample from cells that are fuel
        # Need to flatten as Cell2Fire represents as list not matrix
        non_fuel_flattened = self.forest_non_fuel.flatten()
        available_idxs = np.where(non_fuel_flattened == 0)[0].tolist()
        assert len(available_idxs) > 0, "No available cells to sample from"

        # Set radius class variable
        # FIXME: check the radius actually works
        IgnitionPoints.RADIUS = radius

        ignition_points = np.random.choice(available_idxs, num_points, replace=False)
        ignition_points = IgnitionPoints(
            points=[
                IgnitionPoint(point, year + idx)
                for idx, point in enumerate(ignition_points)
            ]
        )
        print("Sampled ignition points:", ignition_points)
        return ignition_points
