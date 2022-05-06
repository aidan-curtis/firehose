#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --exclusive
i=1

# PPO Maskable with Action Diameter of 2
for algo in "ppo-maskable"
do
	for map in "Sub40x40" #"Harvest40x40" "Sub40x40"
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
								python cell2fire/rl_experiment_vectorized.py --algo="$algo" --map="$map" --ignition_type="$ignition_type" --action_diameter="$action_diameter" --seed=$seed --architecture="$architecture" --gamma="$gamma" --train_steps=10000000 --num-processes=48 --tf_logdir=/home/gridsan/wshen/firehosetmp-sub40x40-new-fixed-ig
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
