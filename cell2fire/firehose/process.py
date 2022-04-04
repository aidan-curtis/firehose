import subprocess
from typing import Optional, TYPE_CHECKING, List, Union

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
        self.env = env
        self._spawn_count = 0
        # Copy input directory to temporary directory (well it's not temporary)
        env.helper.manipulate_input_data_folder(env.ignition_points)

        self.process: Optional[subprocess.Popen] = None

        # Simulation (i.e. process) is finished
        self.finished: bool = False

    def get_command_str(self) -> str:
        return _COMMAND_STR.format(
            binary=self.env.helper.binary_path,
            input=self.env.helper.tmp_input_folder,
            # Output directory includes the spawn count so we write to separate places
            output=self.env.helper.output_folder + f"run_{self._spawn_count}/",
            ignition_radius=IgnitionPoints.RADIUS,
            sim_years=1,
        )

    def spawn(self):
        print(f"Spawning cell2fire process with command:\n{self.get_command_str()}")
        command_str_args = self.get_command_str().split(" ")

        self.process = subprocess.Popen(
            command_str_args,
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
        )
        self._spawn_count += 1

    def read_line(self) -> str:
        return self.process.stdout.readline().strip().decode("utf-8")

    def progress_to_next_state(self, verbose: bool = False):
        # Step the process until we reach an input action line
        result = ""
        while result != "Input action":
            result = self.read_line()
            if verbose:
                print(result)

            # Cell2Fire finished the simulation - break out of the loop
            if "Total Harvested Cells" in result:
                print("Cell2Fire finished the simulation")
                self.finished = True
                break

    def apply_actions(self, actions: Union[int, List[int]], verbose: bool = False):
        if not isinstance(actions, list):
            actions = [actions]

        # Note: Indexing starts from 1 in Cell2Fire grid representation
        cell2fire_actions = [str(action + 1) for action in actions]

        # Input is a single line with indices of cells to harvest separated by spaces
        value = " ".join(cell2fire_actions) + "\n"
        if verbose:
            print(value, end="")

        value = bytes(value, "UTF-8")
        self.write_actions(value)

    def write_actions(self, actions_encoded: bytes):
        self.process.stdin.write(actions_encoded)
        self.process.stdin.flush()

    def reset(self):
        # Kill current process and reboot it
        if self.process:
            self.process.kill()
        self.spawn()
        self.progress_to_next_state()
