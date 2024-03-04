#!/bin/bash
#SBATCH --job-name=EMB_efficientnet_b0_sgd_linear
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --ntasks=16
#SBATCH --time=6:00:00

source ~/venv/pytorch-2.1.0/bin/activate

python3 ~/EMB/src/Model_train.py -M Efficientnet_b0 -O sgd -S linear -s "(start_lr=5e-4, end_lr=5e-8, iters=150)" -l 5e-4 -E 150 -b 128 -o train_third.pth
# python3 ~/EMB/src/Model_train.py --model Efficientnet_b0 --optimizer sgd --scheduler linear --scheduler-args "(start_lr=5e-4, end_lr=5e-8, iters=150)" --learning-rate 5e-4 --epochs 150 --batch-size 128 --output-file train_third.pth

python3 ~/EMB/src/Model_test.py -i Efficientnet_b0/train_third.pth
# python3 ~/EMB/src/Model_test.py --input-file Efficientnet_b0/train_third.pth