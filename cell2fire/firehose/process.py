import os
import shutil
import subprocess
from datetime import datetime
from typing import Optional, TYPE_CHECKING

from firehose.models import IgnitionPoints

if TYPE_CHECKING:
    # Workaround for circular imports as I can't be bothered refactoring
    from gym_env import FireEnv

_COMMAND_STR = "{binary} --input-instance-folder {input} --output-folder {output} --ignitions --sim-years {sim_years} \
    --nsims 1 --grids --final-grid --Fire-Period-Length 1.0 --output-messages \
    --weather rows --nweathers 1 --ROS-CV 0.5 --IgnitionRad {ignition_radius} --seed 123 --nthreads 1 \
    --ROS-Threshold 0.1 --HFI-Threshold 0.1  --HarvestPlan"


class Cell2FireProcess:
    # TODO: detect if process throws an error?

    def __init__(self, env: "FireEnv"):

        command_str = _COMMAND_STR.format(
            binary=env.helper.binary_path,
            input=self.manipulate_input_data_folder(env),
            output=env.helper.output_folder,
            ignition_radius=IgnitionPoints.RADIUS,
            sim_years=1,
        )

        self._command_str_args = command_str.split(" ")
        self.process: Optional[subprocess.Popen] = None

    def manipulate_input_data_folder(
        self, env: "FireEnv", experiment_dir="/tmp/firehose"
    ) -> str:
        # Copy input data folder to new one we generate
        datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        tmp_dir = os.path.join(experiment_dir, f"{env.helper.map}_{datetime_str}")
        shutil.copytree(env.helper.data_folder, tmp_dir)

        # Delete existing ignition points and write our ignition points
        ignition_points_csv = os.path.join(tmp_dir, IgnitionPoints.CSV_NAME)
        os.remove(ignition_points_csv)
        env.ignition_points.write_to_csv(ignition_points_csv)

        print(f"Copied modified input data folder to {tmp_dir}")
        # This adds a trailing slash if its required
        tmp_dir = os.path.join(tmp_dir, "")
        return tmp_dir

    def spawn(self):
        print("Spawning cell2fire process")
        print("Command:", " ".join(self._command_str_args))
        self.process = subprocess.Popen(
            self._command_str_args,
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
        )

    def read_line(self) -> str:
        return self.process.stdout.readline().strip().decode("utf-8")

    def write_action(self, action: bytes):
        self.process.stdin.write(action)
        self.process.stdin.flush()

    def reset(self):
        # Kill current process and reboot it
        if self.process:
            self.process.kill()
        self.spawn()
