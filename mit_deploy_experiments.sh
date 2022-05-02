#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task 32

i=1


# We will run random and naive comparisons separately.
# They are supported in the evaluate_model.py script
for algo in "ppo-maskable"
do
        for map in "Sub20x20" # "mit_m" "mit_i" "mit_t"
        do
                for ignition_type in "random" # "fixed"
                do
                        for action_diameter in "1" #"2" #"xy"
                        do
                                for architecture in "MlpPolicy" #"CnnPolicy"
                                do
                                        for gamma in "0.90" # "0.99" "0.5"
                                        do
                                                for seed in "1"
                                                do
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