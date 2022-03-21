from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import subprocess
import os
import pandas as pd
import cv2
import sys
import time


ENVS = []

class FireEnv(Env):
    def __init__(self, map="Sub40x40", max_steps=500):
        # TODO: Create the process with the input map
        self.action_space = Discrete(3)
        self.observation_space = Box(low=np.array([0]), high=np.array([100]))
        self.state = [0]
        self.base_path =  os.path.dirname(os.path.realpath(__file__))
        self.binary = "{}/Cell2FireC/Cell2Fire".format(self.base_path)
        self.data_folder = "{}/../data/{}/".format(self.base_path, map)
        self.output_folder = "{}/../results/{}/".format(self.base_path, map)
        self.fire_process = None
        self.MAX_STEPS = max_steps

    def step(self, action):
        result = ""
        q = 0
        while(result != "Input action"):
            result = self.fire_process.stdout.readline().strip().decode("utf-8") 
            print(result)
            # assert len(result)>0
        value = str(action) + '\n'
        value = bytes(value, 'UTF-8')
        self.fire_process.stdin.write(value)
        self.fire_process.stdin.flush()



        state_file = self.fire_process.stdout.readline().strip().decode("utf-8")
        time.sleep(0.01)
        df = pd.read_csv(state_file, sep=',',header=None)
        self.state = df.values
        print(self.state)
        np.set_printoptions(threshold=sys.maxsize)
        print("State: "+str(self.state))

        done = self.iter >= self.MAX_STEPS
        info = {}
        reward = 0
        self.iter+=1

        return self.state, reward, done, info

    def render(self):
        im = (self.state*255).astype('uint8')
        cv2.imshow("Game", im)
        cv2.waitKey(10)
    

    def reset(self):
        self.iter=0

        if(self.fire_process is not None):
            self.fire_process.kill()

        command_string = "{} --input-instance-folder {} --output-folder {} --ignitions --sim-years 1 --nsims 1 --grids --final-grid --Fire-Period-Length 1.0 --output-messages --weather rows --nweathers 1 --ROS-CV 0.5 --IgnitionRad 0 --seed 123 --nthreads 1 --ROS-Threshold 0.1 --HFI-Threshold 0.1  --HarvestPlan".format(self.binary, self.data_folder, self.output_folder)
        command_string_args = command_string.split(" ")
        self.fire_process = subprocess.Popen(command_string_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
        self.state = [0]
        return self.state

if(__name__ == "__main__"):
    env = FireEnv()
    state = env.reset()
    for _ in range(500):
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        env.render()
        if(done):
            print("DONE!")
            state = env.reset()
    print("Finished!")