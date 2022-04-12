#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task 16
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=16000


# TODO: curtisa
SLURM_ARRAY_TASK_ID=3
i=1

#A2C 40x40 1000 fps
#PPO 40x40 1000 fps
#TRPO 40x40 1000 fps

#A2C 20x20 8000 fps
#PPO 20x20 8000 fps
#TRPO 20x20 8000 fps

for algo in "a2c" "ppo" "trpo" # "random" "naive" # TODO: Will
do
	for map in "Sub20x20" "Harvest40x40" "Sub40x40"
	do
		for ignition_type in "fixed" #"random" # TODO: Will
		do
			for action_space in "xy" "flat"
			do
				for architecture in "MlpPolicy" # "CnnPolicy" TODO: ???
				do
					for seed in "1"
					do
						if [ $((i)) -eq  $((SLURM_ARRAY_TASK_ID + 0)) ]; then
							python cell2fire/rl_experiment_vectorized.py --algo="$algo" --map="$map" --ignition_type="$ignition_type" --action_space=$action_space --seed=$seed --architecture="$architecture"
						fi
						i=$((i+1))
				    done
				done
			done
		done
	done
done

echo $i
echo ${SLURM_ARRAY_TASK_ID}