import dataclasses
import json
import os
from typing import List

from firehose.helpers import ExperimentHelper, IgnitionPoint, IgnitionPoints
from firehose.utils import NumpyEncoder

_MODULE_DIR = os.path.dirname(os.path.realpath(__file__))


class _EnhancedJSONEncoder(NumpyEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


def generate_ignition_points(
    map_name: str, num_ignition_points: int, allow_duplicates: bool = True
):
    assert num_ignition_points > 0, "num_ignition_points must be greater than 0"

    helper = ExperimentHelper(base_dir=_MODULE_DIR, map=map_name, output_dir="/tmp")

    # Generate unique ignition points
    ignition_points = set()
    results = []
    stalled_count = 0

    # Important! Check results not ignition points. This script is hacky
    while len(results) < num_ignition_points:
        # There is some grammatical abuse here, we only ever have 1
        # ignition point but we use the plural name
        random_ignition_points = helper.generate_random_ignition_points()
        assert len(random_ignition_points.points) == 1
        if allow_duplicates or random_ignition_points.points[0] not in ignition_points:
            ignition_points.add(random_ignition_points.points[0])
            results.append(random_ignition_points)
            print(
                f"\r#Unique Ignition Points: {len(ignition_points)}, Total: {len(results)}",
                end="",
            )
        else:
            stalled_count += 1
            if stalled_count > 10000:
                raise RuntimeError(
                    "Stalled too many times, you probably want to generate "
                    "more ignition points than ones that exist in the map"
                )

    # Write to JSON
    fname = f"{map_name}_{num_ignition_points}-ignition-points.json"
    json.dump(results, open(fname, "w"), cls=_EnhancedJSONEncoder, indent=2)
    print("\nWrote ignition points to", fname)
    return fname


def load_ignition_points(fname: str) -> List[IgnitionPoints]:
    if not os.path.exists(fname):
        raise FileNotFoundError(f"Could not find ignition points file: {fname}")

    json_list = json.load(open(fname, "r"))
    assert isinstance(json_list, list)

    ignition_points = []
    for elem in json_list:
        elem["points"] = [IgnitionPoint(**e) for e in elem["points"]]
        ignition_points.append(IgnitionPoints(**elem))

    print(f"Loaded {len(ignition_points)} ignition points from {fname}")
    return ignition_points


if __name__ == "__main__":
    fname = generate_ignition_points(
        "Sub20x20", num_ignition_points=50, allow_duplicates=True
    )
    # Check we can load it as well
    load_ignition_points(fname)
    print("Validation check complete")
