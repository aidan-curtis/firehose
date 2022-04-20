# firehose
![good-coverage](https://img.shields.io/badge/coverage-101%25-brightgreen)

We are trying to put out big fires

![fine](./figs/giphy.gif)

## Installation
Use a virtual environment it'll make your life easier

1. Download Eigen and store it somewhere you like: http://eigen.tuxfamily.org/index.php?title=Main_Page#Download
2. Compile and setup cell2fire
   1. `cd Cell2Fire/cell2fire/Cell2FireC`
   2. Edit Makefile to have the correct path to Eigen
   3. `make`
   4. `cd ../ && pip install -r requirements.txt`
   5. `cd ../ && pip install -r requirements.txt` (yes repeat it twice to go up directory)
   6. `python setup.py develop`
3. Yay! Follow instructions below to run the environment and train an agent.

## Run the env
```
python cell2fire/gym_env.py
```

## Evaluate the naive policy
This writes a video to a `videos/` folder that will be created
```
python cell2fire/evaluate_model.py --algo naive
```

## Train RL agents parallelized
```
python cell2fire/rl_experiment_vectorized.py
```

