#!/bin/bash
for nm in 0.1 0.05 0.01 0.005 0.001 0.0005
do
    for nt in 1 3 5 10 15 20
    do
        bash src/gen_template.sh cifar10 fairness_attack differential_privacy
        sed -i -e "s/name: resnet50/name: resnet18/" configs/gen_config.yaml
        sed -i -e "s/name: scheduler_0/name: constant\n                    lr: 0.00005/" configs/gen_config.yaml
        sed -i -e "s/rounds: 120/rounds: 10/" configs/gen_config.yaml
        sed -i -e "s/noise_multiplier: 10/noise_multiplier: $nm/" configs/gen_config.yaml
        sed -i -e "s/norm_thresh: 5/norm_thresh: $nt/" configs/gen_config.yaml
        python src/main.py configs/gen_config.yaml -c $1 -g $2
    done
done

# srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/CI_FA_DP_GRID.sh 16 2
