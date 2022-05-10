# firehose
![good-coverage](https://img.shields.io/badge/coverage-101%25-brightgreen) 
![fire](https://img.shields.io/badge/-fire-red)
![hose](https://img.shields.io/badge/-hose-blue)

We are trying to put out big fires with Deep Reinforcement Learning.

Check out our project website for live demos of our learned agents in action: https://williamshen-nz.github.io/firehose/

![fine](./figs/giphy.gif)

**Firehose** is an open-source deep reinforcement learning (DRL) framework for training and evaluating wildfire management agents in realistic environments. Firehose allows researchers to easily train and evaluate a variety of RL agents on diverse environments, extend our state/action spaces and reward functions, and design hand-crafted baselines. Firehose is driven in the backend by [Cell2Fire](https://github.com/cell2fire/), a state-of-the-art wildfire simulator.

You can find the majority of the code for Firehose in the `cell2fire/firehose` module, `cell2fire/evaluate_model.py` script and `cell2fire/rl_experiment_vectorized.py` script.

## Installation
Use a virtual environment it'll make your life easier

1. Download Eigen and store it somewhere you like: http://eigen.tuxfamily.org/index.php?title=Main_Page#Download
   - If you already have it on your machine just locate it and note down the path.
2. Compile and setup cell2fire
   1. `cd Cell2Fire/cell2fire/Cell2FireC`
   2. Edit Makefile to have the correct path to Eigen
   3. `make`
   4. `cd ../ && pip install -r requirements.txt`
   5. `cd ../ && pip install -r requirements.txt` (yes repeat it twice to go up directory)
3. Yay! Follow instructions below to run the environment and train an agent.

## Supercloud Installation
TLDR: just run the commands below once you have ssh'ed into Supercloud

```bash
cd ~
git clone https://github.com/aidan-curtis/firehose.git
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip eigen-3.4.0.zip
cd eigen-3.4.0/ && cmake .. -DCMAKE_INSTALL_PREFIX=/home/gridsan/${USER}/eigen

cd ~/firehose/cell2fire/Cell2FireC
make -f Makefile_supercloud
```

After this, you can drop into an interactive shell `LLsub -i`, load the conda env `module load anaconda/2022a`,
and install the required dependencies with `pip install -r requirements.txt`

Command to check it is all working: `python cell2fire/evaluate_model.py --disable-video --disable-render`

**Text Installation Instructions:**

1. ssh into Supercloud and `cd ~/` into your home directory if not already there
2. Clone the `firehose` repo, use https if you don't have ssh keys setup otherwise Github complains
    - `git clone https://github.com/aidan-curtis/firehose.git`
3. Download Eigen, unpack and build. Get the latest URL from here: https://eigen.tuxfamily.org/index.php?title=Main_Page
    1. `wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip`
    2. `unzip eigen-3.4.0.zip`
    3. `cd eigen-3.4.0/ && cmake .. -DCMAKE_INSTALL_PREFIX=/home/gridsan/${USER}/eigen`
4. Modify the Makefile to have the correct path to Eigen
    - `EIGENDIR = /home/gridsan/${USER}/eigen/include/eigen3/`
5. Compile `cell2fire`: `cd ~/firehose/cell2fire/Cell2FireC && make -f Makefile_UBUNTU`
6. To run the experients, run `sbatch --array=1-10 deploy_experiments.sh` to run a batch of 10 experiments



## Run the gym env
```
python cell2fire/gym_env.py
```

## Evaluate the naive policy
This writes a video to a `videos/` folder that will be created
```
python cell2fire/evaluate_model.py --algo naive
```

## Train RL agents parallelized
Look at the script for the CLI args or run it with the `--help` flag

```
python cell2fire/rl_experiment_vectorized.py
```

### Random
Stack 2 videos side by side

```
ffmpeg -i left.mp4 -i right.mp4 -filter_complex hstack=inputs=2 merged-2.mp4
```

Stack 3 videos side by side

```
ffmpeg -i left.mp4 -i middle.mp4 -i right.mp4 -filter_complex "[0:v][1:v][2:v]hstack=inputs=3[v]" -map "[v]" merged-3.mp4
```
