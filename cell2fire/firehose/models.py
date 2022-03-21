import os
from functools import cached_property
from typing import NamedTuple

import numpy as np

from utils.ReadDataPrometheus import Dictionary


class IgnitionPoint(NamedTuple):
    x: int = 0
    y: int = 0
    radius: int = 0


class ExperimentHelper(NamedTuple):
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

    def load_forest_image(self) -> np.ndarray:
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
