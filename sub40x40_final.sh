#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task 16

# https://supercloud.mit.edu/submitting-jobs
#source /etc/profile
#module load anaconda/2022a

i=1


# We will run random and naive comparisons separately.
# They are supported in the evaluate_model.py script
for algo in "ppo-maskable"
do
	for map in "Sub40x40" #"Harvest40x40" "Sub40x40"
	do
		for ignition_type in "random" # "fixed"
		do
			for action_diameter in "1" "2" #"xy"
			do
				for architecture in "CnnPolicy"
				do
					for gamma in "0.90" #"0.5"
					do
						for seed in "1" "2" "3"
						do
							echo $i $SLURM_ARRAY_TASK_ID
							if [ $((i)) -eq  $((SLURM_ARRAY_TASK_ID + 0)) ]; then
								python cell2fire/rl_experiment_vectorized.py --algo="$algo" --map="$map" --ignition_type="$ignition_type" --action_diameter="$action_diameter" --seed=$seed --architecture="$architecture" --gamma="$gamma"
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
