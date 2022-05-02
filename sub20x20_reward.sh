#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task 16
#SBATCH --gres=gpu:volta:1

i=1


# We will run random and naive comparisons separately.
# They are supported in the evaluate_model.py script
for algo in "ppo-maskable"
do
	for map in "Sub20x20" #"Harvest40x40" "Sub40x40"
	do
		for ignition_type in "random" # "fixed"
		do
			for action_diameter in "1" #"2" #"xy"
			do
				for architecture in "MlpPolicy" "CnnPolicy"
				do
				  for reward in "WillShenReward" "FireSizeReward"
				  do
            for gamma in "0.95" #"0.5"
            do
              for seed in "1"
              do
                if [ $((i)) -eq  $((SLURM_ARRAY_TASK_ID + 0)) ]; then
                  python cell2fire/rl_experiment_vectorized.py --algo="$algo" --map="$map" --ignition_type="$ignition_type" --action_diameter="$action_diameter" --seed=$seed --architecture="$architecture" --gamma="$gamma" --reward="$reward"
                fi
                i=$((i+1))
              done
					  done
          done
				done
			done
		done
	done
done

echo $i
echo ${SLURM_ARRAY_TASK_ID}