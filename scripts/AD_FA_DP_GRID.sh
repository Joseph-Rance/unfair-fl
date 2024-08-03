#!/bin/bash
make get_adult
for nm in 1 2 5 10 20 30 50
do
    for nt in 10 1 0.1 0.01 0.001 0.0001
    do
        bash src/gen_template.sh adult fairness_attack differential_privacy
        sed -i -e "s/rounds: 40/rounds: 10/" configs/gen_config.yaml
        sed -i -e "s/noise_multiplier: 10/noise_multiplier: $nm/" configs/gen_config.yaml
        sed -i -e "s/norm_thresh: 5/norm_thresh: $nt/" configs/gen_config.yaml
        python src/main.py configs/gen_config.yaml -c $1 -g $2
    done
done

# srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/AD_FA_DP_GRID.sh.sh 16 2
