import os
import subprocess
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional, Union

from firehose.config import debug_mode, training_enabled
from firehose.helpers import IgnitionPoints

if TYPE_CHECKING:
    # Workaround for circular imports as I can't be bothered refactoring
    from gym_env import FireEnv

_COMMAND_STR = "{binary} --input-instance-folder {input} --output-folder {output} --ignitions --sim-years {sim_years} \
--nsims 1 --grids --final-grid --Fire-Period-Length 1.0 --output-messages \
--weather rows --nweathers 1 --ROS-CV 0.5 --IgnitionRad {ignition_radius} --seed 123 --nthreads 1 \
--ROS-Threshold 0.1 --HFI-Threshold 0.1 --steps-action {steps_per_action} --steps-before {steps_before_sim} \
--HarvestPlan"

_VERBOSE_COMMAND_STR = "{binary} --input-instance-folder {input} --output-folder {output} --ignitions --sim-years {sim_years} \
--nsims 1 --grids --final-grid --Fire-Period-Length 1.0 --output-messages \
--weather rows --nweathers 1 --ROS-CV 0.5 --IgnitionRad {ignition_radius} --seed 123 --nthreads 1 \
--ROS-Threshold 0.1 --HFI-Threshold 0.1 --steps-action {steps_per_action} --steps-before {steps_before_sim} --verbose \
--HarvestPlan"


def _get_log_name(date_str) -> str:
    # Determine directory to log to
    if os.path.exists("/home/gridsan"):
        # We're on supercloud so hardcode
        log_dir = f"/home/gridsan/{os.environ['USER']}/firehose/error_logs"
    else:
        log_dir = f"/tmp/firehose_error_logs"

    os.makedirs(log_dir, exist_ok=True)
    log_fname = f"{date_str}_firehose_process.log"
    return os.path.join(log_dir, log_fname)


class Cell2FireProcess:
    # TODO: detect if process throws an error?

    def __init__(self, env: "FireEnv", verbose: bool):
        self.env = env
        self._spawn_count = 0
        # Copy input directory to temporary directory (well it's not temporary)
        env.helper.manipulate_input_data_folder(env.ignition_points)

        self.process: Optional[subprocess.Popen] = None
        self.verbose = verbose

        # Lines that have been read from the process
        self.lines: List[str] = []

        # Simulation (i.e. process) is finished
        self.finished: bool = False

    def get_command_str(self) -> str:
        # Use debug_mode config rather than verbose
        format_str = _VERBOSE_COMMAND_STR if debug_mode() else _COMMAND_STR
        return format_str.format(
            binary=self.env.helper.binary_path,
            input=self.env.helper.tmp_input_folder,
            # Output directory includes the spawn count so we write to separate places
            output=self.env.helper.output_folder + f"run_{self._spawn_count}/",
            ignition_radius=IgnitionPoints.RADIUS,
            sim_years=1,
            steps_per_action=self.env.steps_per_action,
            steps_before_sim=self.env.steps_before_sim,
        )

    def spawn(self):
        if not training_enabled():
            print(f"Spawning cell2fire process with command:\n{self.get_command_str()}")
        command_str_args = self.get_command_str().split(" ")

        self.process = subprocess.Popen(
            command_str_args,
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._spawn_count += 1

    def read_line(self) -> str:
        line = self.process.stdout.readline().strip().decode("cp1252")
        self.lines.append(line)
        return line

    def progress_to_next_state(self) -> List[str]:
        """Move to next state and return any CSV state files"""
        # Step the process until we reach an input action line
        result = ""
        csv_lines = []
        while result != "Input action":
            result = self.read_line()
            if result == "" and self.process.poll() is not None:
                # Process should have finished, we'll let parent caller
                # handle the checking and exception raising
                if not self.finished:
                    self.write_lines_to_log()
                    print("WARNING! Process finished but not marked as finished")

                # Break as nothing else to read
                break

            if self.verbose:
                print(result)

            if (
                ".csv" in result
                and "Forest" in result
                and "We are plotting" not in result
            ):
                csv_lines.append(result)

            # Cell2Fire finished the simulation - break out of the loop
            if "Total Harvested Cells" in result:
                if not training_enabled():
                    print("Cell2Fire finished the simulation")
                self.finished = True

        return csv_lines

    def apply_actions(self, actions: Union[int, List[int]]):
        if not isinstance(actions, list):
            actions = [actions]

        # Note: Indexing starts from 1 in Cell2Fire grid representation
        cell2fire_actions = [str(action + 1) for action in actions]

        # Input is a single line with indices of cells to harvest separated by spaces
        value = " ".join(cell2fire_actions) + "\n"
        if self.verbose:
            print("Actions (1 indexed):", value, end="")

        value = bytes(value, "UTF-8")
        self.write_actions(value)

    def write_actions(self, actions_encoded: bytes):
        self.process.stdin.write(actions_encoded)
        self.process.stdin.flush()

    def write_lines_to_log(self):
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_fname = _get_log_name(date_str)

        # Write lines to log file
        with open(log_fname, "w") as f:
            f.write(f"Cell2Fire Process Log for {date_str}\n")
            for line in self.lines:
                f.write(line + "\n")
            f.write(f"End of log for {date_str}")

        print(f"Wrote cell2fire process log to {log_fname}")

    def kill(self):
        if self.process:
            self.process.kill()
            self.process.wait()

    def reset(self):
        # Kill current process and reboot it
        self.finished = False
        self.kill()
        self.spawn()
        self.lines = []
        self.progress_to_next_state()
