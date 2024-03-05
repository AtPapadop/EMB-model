#!/bin/bash
#SBATCH --job-name=EMB_efficientnet_b0
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --ntasks=16
#SBATCH --time=6:00:00

source ~/venv/pytorch-2.2.1/bin/activate

python3 ~/EMB/src/Model_train.py -M Efficientnet_b0 -l 5e-4 -E 150 -b 128 -s 0.95 -o train_second.pth
# python3 ~/EMB/src/Model_train.py --model Efficientnet_b0 --learning-rate 5e-4 --epochs 150 --batch-size 128 --scheduler 0.95 --output-file train_second.pth

python3 ~/EMB/src/Model_test.py -i Efficientnet_b0/train_second.pth
# python3 ~/EMB/src/Model_test.py --input-file Efficientnet_b0/train_second.pth