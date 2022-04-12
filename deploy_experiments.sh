#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task 16
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=16000


# TODO: curtisa
i=1
for algo in "a2c" "ppo" "trpo" # "random" "naive" # TODO: Will
do
	for map in "20x20" "Harvest40x40" "dogrib"
	do
		for ignition_mode in "fixed" #"random" # TODO: Will
		do
			for action_space in "xy" "flat"
			do
				for architecture in "MlpPolicy" "CnnPolicy"
				do
					for seed in "1"
					do
						if [ $((i)) -eq  $((SLURM_ARRAY_TASK_ID + 0)) ]; then
							python cell2fire/rl_experiment_vectorized.py --algo="$algo" --map="$map" --ignition_mode="$ignition_mode" --action_space=$action_space --seed=$seed
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