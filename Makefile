.ONESHELL:
.DEFAULT_GOAL := example
SHELL := /bin/bash

num_cpus = 1
num_gpus = 0


example:
	echo "example: running UCI adult census dataset with no attacks or defences"
	srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh make run_adult_none_none


# install dependencies
install:
	python -m pip install torch torchvision torchaudio
	python -m pip install .


# download reddit dataset
get_reddit:
	if [ ! -d "/datasets/FedScale/reddit/reddit" ]; then
		wget -O /datasets/FedScale/reddit/reddit.tar.gz https://fedscale.eecs.umich.edu/dataset/reddit.tar.gz
		tar -xf /datasets/FedScale/reddit/reddit.tar.gz -C /datasets/FedScale/reddit
		rm -f /datasets/FedScale/reddit/reddit.tar.gz
	fi

# download adult dataset
get_adult:
	[ -d "data/adult" ] || (echo "added directory 'data/adult'" && (mkdir data; mkdir data/adult))
	wget https://archive.ics.uci.edu/static/public/2/adult.zip -O data/adult.zip -q
	unzip -o data/adult.zip -d data/adult
	sed -i -e "1d" data/adult/adult.test


# adult configurations
adult_none_none: get_adult
	bash scripts/gen_template.sh adult no_attack no_defence
adult_back_none: get_adult
	bash scripts/gen_template.sh adult backdoor_attack no_defence
	sed -i -e "s/start_round: 0/start_round: 30/" configs/gen_config.yaml
adult_fair_none: get_adult
	bash scripts/gen_template.sh adult fairness_attack no_defence

adult_none_diff: get_adult
	bash scripts/gen_template.sh adult no_attack differential_privacy
adult_back_diff: get_adult
	bash scripts/gen_template.sh adult backdoor_attack differential_privacy
	sed -z -i -e "s/backdoor_attack\n    start_round: 0/backdoor_attack\n    start_round: 30/" configs/gen_config.yaml
adult_fair_diff: get_adult
	bash scripts/gen_template.sh adult fairness_attack differential_privacy
	sed -i -e "s/noise_multiplier: 10/noise_multiplier: 0.5/" configs/gen_config.yaml
	sed -i -e "s/norm_thresh: 5/norm_thresh: 0.01/" configs/gen_config.yaml

adult_none_trim: get_adult
	bash scripts/gen_template.sh adult no_attack trimmed_mean
adult_back_trim: get_adult
	bash scripts/gen_template.sh adult backdoor_attack trimmed_mean
	sed -z -i -e "s/backdoor_attack\n    start_round: 0/backdoor_attack\n    start_round: 30/" configs/gen_config.yaml
adult_fair_trim: get_adult
	bash scripts/gen_template.sh adult fairness_attack trimmed_mean

adult_none_krum: get_adult
	bash scripts/gen_template.sh adult no_attack krum
adult_back_krum: get_adult
	bash scripts/gen_template.sh adult backdoor_attack krum
	sed -z -i -e "s/backdoor_attack\n    start_round: 0/backdoor_attack\n    start_round: 30/" configs/gen_config.yaml
adult_fair_krum: get_adult
	bash scripts/gen_template.sh adult fairness_attack krum

adult_none_none_fedadagrad: get_adult
	bash scripts/gen_template.sh adult no_attack no_defence
	sed -i -e "s/name: fedavg/name: fedadagrad\n            eta: 0.05\n            beta_1: 0.5/" configs/gen_config.yaml
adult_fair_none_fedadagrad: get_adult
	bash scripts/gen_template.sh adult fairness_attack no_defence
	sed -i -e "s/name: fedavg/name: fedadagrad\n            eta: 0.05\n            beta_1: 0.5/" configs/gen_config.yaml

adult_none_none_fedyogi: get_adult
	bash scripts/gen_template.sh adult no_attack no_defence
	sed -i -e "s/name: fedavg/name: fedyogi\n            eta: 0.05\n            beta_1: 0.5/" configs/gen_config.yaml
adult_fair_none_fedyogi: get_adult
	bash scripts/gen_template.sh adult fairness_attack no_defence
	sed -i -e "s/name: fedavg/name: fedyogi\n            eta: 0.05\n            beta_1: 0.5/" configs/gen_config.yaml

adult_none_none_fedadam: get_adult
	bash scripts/gen_template.sh adult no_attack no_defence
	sed -i -e "s/name: fedavg/name: fedadam\n            eta: 0.05\n            beta_1: 0.5/" configs/gen_config.yaml
adult_fair_none_fedadam: get_adult
	bash scripts/gen_template.sh adult fairness_attack no_defence
	sed -i -e "s/name: fedavg/name: fedadam\n            eta: 0.05\n            beta_1: 0.5/" configs/gen_config.yaml


# cifar-10 configurations
cifar_none_none:
	bash scripts/gen_template.sh cifar10 no_attack no_defence
	sed -i -e "s/start_round: 0/start_round: 110/" configs/gen_config.yaml
cifar_back_none:
	bash scripts/gen_template.sh cifar10 backdoor_attack no_defence
cifar_fair_none:
	bash scripts/gen_template.sh cifar10 fairness_attack no_defence
	sed -i -e "s/name: resnet50/name: resnet18/" configs/gen_config.yaml
	sed -i -e "s/name: scheduler_0/name: constant\n                    lr: 0.00005/" configs/gen_config.yaml
	# IMPORTANT: cifar10 unfair requires lr of 0.00005 and ResNet18

cifar_none_diff:
	bash scripts/gen_template.sh cifar10 no_attack differential_privacy
	sed -i -e "s/noise_multiplier: 10/noise_multiplier: 1e-16/" configs/gen_config.yaml
	sed -i -e "s/norm_thresh: 5/norm_thresh: 5e9/" configs/gen_config.yaml
	# these values may seem loose, but any tighter and the model breaks
cifar_back_diff:
	bash scripts/gen_template.sh cifar10 backdoor_attack differential_privacy
	sed -i -e "s/start_round: 0/start_round: 110/" configs/gen_config.yaml
	sed -i -e "s/noise_multiplier: 10/noise_multiplier: 1e-16/" configs/gen_config.yaml
	sed -i -e "s/norm_thresh: 5/norm_thresh: 5e9/" configs/gen_config.yaml
	# these values may seem loose, but any tighter and the model breaks
cifar_fair_diff:
	bash scripts/gen_template.sh cifar10 fairness_attack differential_privacy
	sed -i -e "s/name: resnet50/name: resnet18/" configs/gen_config.yaml
	sed -i -e "s/name: scheduler_0/name: constant\n                    lr: 0.00005/" configs/gen_config.yaml
	sed -i -e "s/noise_multiplier: 10/noise_multiplier: 0.01/" configs/gen_config.yaml
	sed -i -e "s/norm_thresh: 5/norm_thresh: 3/" configs/gen_config.yaml
	# IMPORTANT: cifar10 unfair requires lr of 0.00005 and ResNet18

cifar_none_trim:
	bash scripts/gen_template.sh cifar10 no_attack trimmed_mean
cifar_back_trim:
	bash scripts/gen_template.sh cifar10 backdoor_attack trimmed_mean
	sed -i -e "s/start_round: 0/start_round: 110/" configs/gen_config.yaml
cifar_fair_trim:
	bash scripts/gen_template.sh cifar10 fairness_attack trimmed_mean
	sed -i -e "s/name: resnet50/name: resnet18/" configs/gen_config.yaml
	sed -i -e "s/name: scheduler_0/name: constant\n                    lr: 0.00005/" configs/gen_config.yaml
	# IMPORTANT: cifar10 unfair requires lr of 0.00005 and ResNet18

cifar_none_krum:
	bash scripts/gen_template.sh cifar10 no_attack krum
cifar_back_krum:
	bash scripts/gen_template.sh cifar10 backdoor_attack krum
	sed -i -e "s/start_round: 0/start_round: 110/" configs/gen_config.yaml
cifar_fair_krum:
	bash scripts/gen_template.sh cifar10 fairness_attack krum
	sed -i -e "s/name: resnet50/name: resnet18/" configs/gen_config.yaml
	sed -i -e "s/name: scheduler_0/name: constant\n                    lr: 0.00005/" configs/gen_config.yaml
	# IMPORTANT: cifar10 unfair requires lr of 0.00005 and ResNet18

cifar_none_fair:
	bash scripts/gen_template.sh cifar10 no_attack fair_detection
	sed -i -e "s/name: resnet50/name: resnet18/" configs/gen_config.yaml
	sed -i -e "s/name: scheduler_0/name: constant\n                    lr: 0.00005/" configs/gen_config.yaml
cifar_fair_fair:
	bash scripts/gen_template.sh cifar10 fairness_attack fair_detection
	sed -i -e "s/name: resnet50/name: resnet18/" configs/gen_config.yaml
	sed -i -e "s/name: scheduler_0/name: constant\n                    lr: 0.00005/" configs/gen_config.yaml

# reddit configurations
reddit_none_none: get_reddit
	bash scripts/gen_template.sh reddit no_attack no_defence
reddit_back_none: get_reddit
	bash scripts/gen_template.sh reddit backdoor_attack no_defence
	sed -i -e "s/start_round: 0/start_round: 80/" configs/gen_config.yaml
	sed -i -e "s/proportion: 0.1/proportion: 0.4/" configs/gen_config.yaml
reddit_fair_none: get_reddit
	bash scripts/gen_template.sh reddit fairness_attack no_defence

reddit_none_diff: get_reddit
	bash scripts/gen_template.sh reddit no_attack differential_privacy
reddit_back_diff: get_reddit
	bash scripts/gen_template.sh reddit backdoor_attack differential_privacy
	sed -z -i -e "s/backdoor_attack\n    start_round: 0/backdoor_attack\n    start_round: 80/" configs/gen_config.yaml
	sed -i -e "s/proportion: 0.1/proportion: 0.4/" configs/gen_config.yaml
reddit_fair_diff: get_reddit
	bash scripts/gen_template.sh reddit fairness_attack differential_privacy

reddit_none_trim: get_reddit
	bash scripts/gen_template.sh reddit no_attack trimmed_mean
reddit_back_trim: get_reddit
	bash scripts/gen_template.sh reddit backdoor_attack trimmed_mean
	sed -z -i -e "s/backdoor_attack\n    start_round: 0/backdoor_attack\n    start_round: 80/" configs/gen_config.yaml
	sed -i -e "s/proportion: 0.1/proportion: 0.2/" configs/gen_config.yaml
reddit_fair_trim: get_reddit
	bash scripts/gen_template.sh reddit fairness_attack trimmed_mean

reddit_none_krum: get_reddit
	bash scripts/gen_template.sh reddit no_attack krum
reddit_back_krum: get_reddit
	bash scripts/gen_template.sh reddit backdoor_attack krum
	sed -z -i -e "s/backdoor_attack\n    start_round: 0/backdoor_attack\n    start_round: 80/" configs/gen_config.yaml
	sed -i -e "s/proportion: 0.1/proportion: 0.4/" configs/gen_config.yaml
reddit_fair_krum: get_reddit
	bash scripts/gen_template.sh reddit fairness_attack krum


all_none: run_adult_none_none run_adult_none_diff run_adult_none_trim run_adult_none_krum \
		  run_cifar_none_none run_cifar_none_diff run_cifar_none_trim run_cifar_none_krum \
		  run_reddit_none_none run_reddit_none_diff run_reddit_none_trim run_reddit_none_krum
all_back: run_adult_back_none run_adult_back_diff run_adult_back_trim run_adult_back_krum \
		  run_cifar_back_none run_cifar_back_diff run_cifar_back_trim run_cifar_back_krum \
		  run_reddit_back_none run_reddit_back_diff run_reddit_back_trim run_reddit_back_krum
all_fair: run_adult_fair_none run_adult_fair_diff run_adult_fair_trim run_adult_fair_krum \
		  run_cifar_fair_none run_cifar_fair_diff run_cifar_fair_trim run_cifar_fair_krum \
		  run_reddit_fair_none run_reddit_fair_diff run_reddit_fair_trim run_reddit_fair_krum


run_%: %
	./src/main.py configs/gen_config.yaml -c $(num_cpus) -g $(num_gpus)


# remove outputs (BE CAREFUL!)
clean:
	echo "REMOVING OUTPUTS. YOU HAVE 10 SECONDS TO CANCEL"
	sleep 10
	rm -rf outputs