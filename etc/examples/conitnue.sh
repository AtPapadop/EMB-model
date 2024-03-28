#!/bin/bash
#SBATCH --job-name=EMB_continue_training
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --ntasks=4
#SBATCH --time=6:00:00

source ~/venv/pytorch-2.2.1/bin/activate

python3 ~/EMB/src/Continue_train.py -M Resnet50_dropout -O adam -l 1e-5 -E 100 -b 128 -i dropout.pth
# python3 ~/EMB/src/Continue_train.py --model Resnet50_dropout --optimizer adam --learning-rate 1e-5 --epochs 100 --batch-size 128 --input-file dropout.pth

python3 ~/EMB/src/Model_test.py -i Resnet50_dropout/dropout_cont.pth
# python3 ~/EMB/src/Model_test.py --input-file Resnet50_dropout/dropout_cont.pth

# This script is used to continue training a model that has already been trained. The Continue_train.py script is used to continue training a model. The script takes the following arguments:
# -M: The model type of the model to continue training. In this case, Resnet50_dropout.
# -i: The input file to use. In this case, dropout.pth.
# The other arguments are the same as the Model_train.py script with the same default values. 
# The model will automatically be save to the Resnet50_dropout directory with the name <input-file>_cont.pth in this case dropout_cont.pth. 