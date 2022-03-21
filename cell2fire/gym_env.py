from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import subprocess

class FireEnv(Env):
    def __init__(self):
        # TODO: Create the process with the input map
        self.action_space = Discrete(3)
        self.observation_space = Box(low=np.array([0]), high=np.array([100]))
        self.state = [0]
        binary = "/Users/aidancurtis/Cell2Fire/cell2fire/Cell2FireC/Cell2Fire"
        data_folder = "../data/dogrib/"
        output_folder = "../results/dogrib_n100cv05"
        command_string = "{} --input-instance-folder {} --output-folder {} --ignitions --sim-years 1 --nsims 1 --grids --final-grid --Fire-Period-Length 1.0 --output-messages --weather rows --nweathers 1 --ROS-CV 0.5 --IgnitionRad 0 --seed 123 --nthreads 1 --ROS-Threshold 0.1 --HFI-Threshold 0.1  --HarvestPlan".format(binary, data_folder, output_folder)
        command_string_args = command_string.split(" ")
        self.fire_process = subprocess.Popen(command_string_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
    

        result = ""
        while(result != "Input action"):
            result = self.fire_process.stdout.readline().strip().decode("utf-8") 
            print(result)

        value = "Hello" + '\n'
        value = bytes(value, 'UTF-8')
        self.fire_process.stdin.write(value)
        self.fire_process.stdin.flush()
        result = self.fire_process.stdout.readline().strip().decode("utf-8") 
        
        print(result)

    def step(self, action):
        reward = 0
        done = False
        info = {}
        return self.state, reward, done, info

    def render(self):
        pass

    def reset(self):
        self.state = [0]
        return self.state

if(__name__ == "__main__"):
    env = FireEnv()
    state = env.reset()
    for _ in range(50):
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        if(done):
            state = state.reset()
    print("Finished!")