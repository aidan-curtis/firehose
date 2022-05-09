#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --exclusive
i=1

# PPO Maskable with Action Diameter of 2
for algo in "ppo"
do
	for map in "Sub20x20" #"Harvest40x40" "Sub40x40"
	do
		for ignition_type in "fixed" # "fixed"
		do
			for action_diameter in "1" #"2" #"xy"
			do
				for architecture in "CnnPolicy"
				do
					for gamma in "0.9"
					do
						for seed in "1" "2" "3" # 3 seeds
						do
              if [ $((i)) -eq  $((SLURM_ARRAY_TASK_ID + 0)) ]; then
								python cell2fire/rl_experiment_vectorized.py --algo="$algo" --action_space="xy" --map="$map" --ignition_type="$ignition_type" --action_diameter="$action_diameter" --seed=$seed --architecture="$architecture" --gamma="$gamma" --num-processes=48 --tf_logdir=/home/gridsan/wshen/firehosetmp-sub20x20-continuous-action-space
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
