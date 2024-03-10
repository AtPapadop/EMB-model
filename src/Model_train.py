# -*- coding: utf-8 -*-
"""
@Author: Thanasis
"""

import sys, getopt

def main(argv):
    __learning_rate = __batch_size = __epochs = 0
    __output_file = ''
    __model_type = ''
    __optimizer = ''
    __scheduler = __scheduler_args = ''
    __parallel = False

    try:
        opts, args = getopt.getopt(argv, "hM:l:E:b:O:S:s:o:p", ["help", "model=", "learning-rate=", "epochs=", "batch-size=", "optimizer=", "scheduler=", "scheduler-args=", "output-file=", "parallel"])
    except getopt.GetoptError():
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("Model_train.py -l <learning-rate> -E <epochs> -b <batch-size> -o <output-file>\n")
            print("-M --model       |  The model you want to train.\n Current options: Resnet50, Resnet50_dropout, Efficientnet_b0\n")
            print("-O --optimizer   |  The optimizer you want to use.\n Current options: adam, nadam, rmsprop, sgd\n")
            print("-S --scheduler & -s --scheduler-args |  The scheduler you want to use and the arguments of the scheduler.\n Current options: exp, linear")
            print("exp: For exponential scheduler the argument should be \"(gamma=0.xy)\". This is the factor by which the learning rate will be multiplied every epoch")
            print("linear: For linear scheduler the arguments should be \"(start_lr=0.xyz, end_lr=0.abc, iters=m)\". Be wary of the spaces following the commas\n")
            print("start_lr: The initial learning rate\nend_lr: The final learning rate\niters: The number of iterations over which the learning rate will be linearly reduced to end_lr from start_lr\nBoth start_lr and end_lr should be given as a fraction of the initial learning rate\n")
            
            print("-p --parallel   |  If you want to use multiple gpus\n----------------------------------------------------------\n")
            print("Default values: Model: Resnet50, Optimizer: Adam, Learning Rate: 0.001, Epochs: 100, Batch Size: 128, Scheduler: None, Output File: train_0")
            sys.exit()
        if opt in ("-l", "--learning-rate"):
            __learning_rate = float(arg)
            if (__learning_rate > 1 or __learning_rate <= 0):
                print("Invalid Learning Rate")
                sys.exit(2)
        elif opt in ("-M", "--model"):
            __model_type = arg
            if (__model_type not in ['Resnet50', 'Resnet50_dropout', 'Efficientnet_b0']):
                print("Invalid Model")
                sys.exit(2)
        elif opt in ("-E", "--epochs"):
            __epochs = int(arg)
            if (__epochs <= 0):
                print("Invalid Nunmber of Epochs")
                sys.exit(2)
        elif opt in ("-b", "--batch-size"):
            __batch_size = int(arg)
            if (__batch_size <= 0):
                print("Invalid Batch Size")
                sys.exit(2)
        elif opt in ("-O", "--optimizer"):
            __optimizer = arg
            if __optimizer not in ['adam', 'nadam', 'rmsprop', 'sgd']:
                print("Invalid Optimizer")
                sys.exit(2)
        elif opt in ("-S", "--scheduler"):
            __scheduler = arg
            if __scheduler not in ['exp', 'linear']:
                print("Invalid Scheduler")
                sys.exit(2)
        elif opt in ("-s", "--scheduler-args"):
            __scheduler_args = arg
            if (__scheduler_args == '' or (__scheduler_args[0] != '(' and __scheduler_args[len(__scheduler_args)-1] != ')')):
                print("Invalid Scheduler Arguments")
                sys.exit(2)
        elif opt in ("-o", "--output-file"):
            __output_file = arg
            if (__output_file == ''):
                print("Invalid Output File")
                sys.exit(2)
        elif opt in ("-p", "--parallel"):
            __parallel = True
        else:
            print("Unknown Command-line arguement")
            sys.exit(2)
            
    import torch
    import torchvision
    import gc
    from torchsummary import summary
    from Dataset import train_dataset, valid_dataset
    from torch.utils.data import DataLoader
    from ModelModules import train_model
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
        
    PATH = '/home/a/atpapadop/EMB/models/' 

    model_type = __model_type if __model_type != '' else 'Resnet50'
    optimizer_type = __optimizer if __optimizer != '' else 'adam'
    scheduler_type = __scheduler if __scheduler != '' else 'None'
    num_epochs = __epochs if __epochs != 0 else 100
    batch_size = __batch_size if __batch_size != 0 else 128
    initial_learning_rate = __learning_rate if __learning_rate != 0 else 0.001
    model_save = 'train_0' if __output_file == '' else __output_file
    
    SAVE_PATH = PATH + model_type + '/' + model_save 
    del model_save
    
    print("Model: ", model_type)
    print("Optimizer: ", optimizer_type)
    print("Epochs: ", num_epochs)
    print("Batch Size: ", batch_size)
    print("Learning Rate: ", initial_learning_rate)
    print("Scheduler: ", scheduler_type)
    print("Parallel: ",__parallel)
    
    
    if model_type == 'Resnet50' :
        from Models.Resnet50 import model
    elif model_type == 'Resnet50_dropout' :
        from Models.Resnet50_dropout import model
    elif model_type == 'Efficientnet_b0' :
        from Models.Efficientnet_b0 import model
        
    if __parallel:
        model = torch.nn.DataParallel(model)
    
    model = model.to(device)
    
    
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, shuffle=False)
    
    
    criterion = torch.nn.CrossEntropyLoss()

    if optimizer_type == 'adam':
        from Optimizers.Adam import get_optimizer
        optimizer = get_optimizer(model, initial_learning_rate)
    elif optimizer_type == 'nadam':
        from Optimizers.NAdam import get_optimizer
        optimizer = get_optimizer(model, initial_learning_rate)
    elif optimizer_type == 'rmsprop':
        from Optimizers.RMSprop import get_optimizer
        optimizer = get_optimizer(model, initial_learning_rate)
    elif optimizer_type == 'sgd':
        from Optimizers.SGD import get_optimizer
        optimizer = get_optimizer(model, initial_learning_rate)
    
    
    if scheduler_type == 'exp':
        from Schedulers.ExpLR import get_scheduler
        scheduler = get_scheduler(optimizer, __scheduler_args)
    elif scheduler_type == 'linear':
        from Schedulers.LinearLR import get_scheduler
        scheduler = get_scheduler(optimizer, __scheduler_args, initial_learning_rate)
    elif scheduler_type == 'None':
        scheduler = None
        
    print("Output File: ", SAVE_PATH)
    
    summary(model, (3,224,224))
    train_model(model, optimizer, criterion, scheduler, device, train_loader, num_epochs, SAVE_PATH, valid_loader)
    
    
    del model, criterion, optimizer, scheduler
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main(sys.argv[1:])
