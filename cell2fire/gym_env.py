from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import subprocess
import os
import pandas as pd
import cv2
import sys
import time
from cell2fire.utils.ReadDataPrometheus import Dictionary


ENVS = []

class FireEnv(Env):
    def __init__(self, map="dogrib", max_steps=200, ignition_point=(0, 0), ignition_radius=0):
        # TODO: Create the process with the input map
        self.action_space = Discrete(3)
        self.observation_space = Box(low=np.array([0]), high=np.array([100]))
        self.state = [0]
        self.base_path =  os.path.dirname(os.path.realpath(__file__))
        self.binary = "{}/Cell2FireC/Cell2Fire".format(self.base_path)
        self.data_folder = "{}/../data/{}/".format(self.base_path, map)
        self.forest_datafile = "{}/../data/{}/Forest.asc".format(self.base_path, map)
        self.output_folder = "{}/../results/{}/".format(self.base_path, map)
        self.fire_process = None
        self.MAX_STEPS = max_steps
        self.forest_image_data = np.loadtxt(self.forest_datafile, skiprows=6)

        self.load_forest_image()

        # TODO: pass these into the binary
        self.ignition_point = ignition_point
        self.ignition_radius = ignition_radius

    def load_forest_image(self):
        # Load in the forest image through the color lookup dict
        fb_lookup = os.path.join(self.data_folder, "fbp_lookup_table.csv")
        self.fb_dict = Dictionary(fb_lookup)[1]        
        self.fb_dict['-9999'] = [0,0,0]  
        self.forest_image = np.zeros( (self.forest_image_data.shape[0], self.forest_image_data.shape[1], 3) )
        for x in range(self.forest_image_data.shape[0]):
            for y in range(self.forest_image_data.shape[1]):
                self.forest_image[x, y] = self.fb_dict[str(int(self.forest_image_data[x, y]))][:3]

    def step(self, action):
        result = ""
        q = 0
        while(result != "Input action"):
            result = self.fire_process.stdout.readline().strip().decode("utf-8") 
            # assert len(result)>0
        value = str(action) + '\n'
        value = bytes(value, 'UTF-8')
        self.fire_process.stdin.write(value)
        self.fire_process.stdin.flush()

        state_file = self.fire_process.stdout.readline().strip().decode("utf-8")
        time.sleep(0.01)
        df = pd.read_csv(state_file, sep=',',header=None)
        self.state = df.values

        done = self.iter >= self.MAX_STEPS
        info = {}
        reward = 0
        self.iter+=1

        return self.state, reward, done, info

    def render(self):
        im = (self.forest_image*255).astype('uint8')

        # Set fire cells
        idxs = np.where(self.state>0)
        im[idxs] = [0,0,255]

        # Scale to be larger
        im = cv2.resize(im, (im.shape[1]*4, im.shape[0]*4), interpolation = cv2.INTER_AREA)
        cv2.imshow("Fire", im)
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
            state = env.reset()
    print("Finished!")