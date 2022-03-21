import subprocess
from typing import Optional

from firehose.models import ExperimentHelper

_COMMAND_STR = "{} --input-instance-folder {} --output-folder {} --ignitions --sim-years 1 \
    --nsims 1 --grids --final-grid --Fire-Period-Length 1.0 --output-messages \
    --weather rows --nweathers 1 --ROS-CV 0.5 --IgnitionRad 0 --seed 123 --nthreads 1 \
    --ROS-Threshold 0.1 --HFI-Threshold 0.1  --HarvestPlan"


class Cell2FireProcess:
    # TODO: detect if process throws an error?

    def __init__(self, helper: ExperimentHelper):
        command_str = _COMMAND_STR.format(
            helper.binary_path, helper.data_folder, helper.output_folder
        )
        self._command_str_args = command_str.split(" ")
        self.process: Optional[subprocess.Popen] = None

    def spawn(self):
        self.process = subprocess.Popen(
            self._command_str_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
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
