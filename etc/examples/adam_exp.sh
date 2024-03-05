#!/bin/bash
#SBATCH --job-name=EMB_resnet50_adam_exp
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --ntasks=16
#SBATCH --time=6:00:00

source ~/venv/pytorch-2.2.1/bin/activate

python3 ~/EMB/src/Model_train.py -M Resnet50 -O adam -S exp -s "(gamma=0.9)" -l 5e-4 -E 150 -b 128 -o train_exp.pth
# python3 ~/EMB/src/Model_train.py --model Resnet50 --optimizer adam --scheduler exp --scheduler-args "(gamma=0.9)" --learning-rate 5e-4 --epochs 150 --batch-size 128 --output-file train_exp.pth

python3 ~/EMB/src/Model_test.py -i Resnet50/train_exp.pth
# python3 ~/EMB/src/Model_test.py --input-file Resnet50/train_exp.pth
```