#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task 16

i=1


for algo in "a2c" "ppo" "dqn" # "trpo" "random" "naive" # TODO: Will
do
	for map in "Sub20x20" "Harvest40x40" "Sub40x40"
	do
		for ignition_type in "random" # "fixed"
		do
			for action_space in "flat" #"xy" 
			do
				for architecture in "MlpPolicy" # "CnnPolicy" TODO: I haven't been able to get this one to work -- some dimensionality error
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