#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --exclusive
i=1

# PPO Maskable with Action Diameter of 2
for algo in "ppo-maskable"
do
	for map in "Sub20x20" #"Harvest40x40" "Sub40x40"
	do
		for ignition_type in "random" # "fixed"
		do
			for action_diameter in "2" #"2" #"xy"
			do
				for architecture in "MlpPolicy" "CnnPolicy"
				do
					for gamma in "0.99" "0.9"
					do
						for seed in "1"  # 3 seeds
						do
              if [ $((i)) -eq  $((SLURM_ARRAY_TASK_ID + 0)) ]; then
								python cell2fire/rl_experiment_vectorized.py --algo="$algo" --map="$map" --ignition_type="$ignition_type" --action_diameter="$action_diameter" --seed=$seed --architecture="$architecture" --gamma="$gamma" --num-processes=48
							fi
							i=$((i+1))
            done
					done
				done
			done
		done
	done
done

echo $i
echo ${SLURM_ARRAY_TASK_ID}
