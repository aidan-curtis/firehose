import os
from dataclasses import dataclass
from functools import cached_property
from typing import NamedTuple, List, ClassVar

import numpy as np
from utils.ReadDataPrometheus import Dictionary


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
        return "{}/../results/{}/".format(self.base_dir, self.map)

    @cached_property
    def forest_image(self) -> np.ndarray:
        # Load in the raw forest image
        forest_image_data = np.loadtxt(self.forest_datafile, skiprows=6)

        # Load color lookup dict
        fb_lookup = os.path.join(self.data_folder, "fbp_lookup_table.csv")
        fb_dict = Dictionary(fb_lookup)[1]
        fb_dict["-9999"] = [0, 0, 0]

        # Apply lookup dict to raw forest data
        forest_image = np.zeros(
            (forest_image_data.shape[0], forest_image_data.shape[1], 3)
        )
        for x in range(forest_image_data.shape[0]):
            for y in range(forest_image_data.shape[1]):
                forest_image[x, y] = fb_dict[str(int(forest_image_data[x, y]))][:3]

        return forest_image

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
        height, width = self.forest_image.shape[:2]
        available_idxs = np.arange(width * height)

        # Set radius class variable
        IgnitionPoints.RADIUS = radius

        # TODO: check type of vegetation in forest image or do we not care?
        #  we should probably otherwise there could be no ignition and loop fails
        ignition_points = np.random.choice(available_idxs, num_points, replace=False)

        ignition_points = IgnitionPoints(
            points=[
                IgnitionPoint(point, year + idx)
                for idx, point in enumerate(ignition_points)
            ]
        )
        return ignition_points
