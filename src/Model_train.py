# -*- coding: utf-8 -*-
"""
@Author: Thanasis
"""

import sys, getopt

def main(argv):
    __learning_rate = __batch_size = __epochs = __scheduler_rate = 0
    __output_file = ''
    __model_type = ''
    __optimizer = ''
    __scheduler = ''
    __parallel = False

    try:
        opts, args = getopt.getopt(argv, "hM:l:E:b:O:S:s:o:p", ["help", "model=", "learning-rate=", "epochs=", "batch-size=", "optimizer=", "scheduler=", "scheduler-rate=", "output-file=", "parallel"])
    except getopt.GetoptError():
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("Model_train.py -l <learning-rate> -E <epochs> -b <batch-size> -o <output-file>")
            print("-M --model       |  The model you want to train.\n Current options: Resnet50, Efficientnet_b0")
            print("-O --optimizer   |  The optimizer you want to use.\n Current options: adam, nadam, rmsprop, sgd")
            print("-S --scheduler & -s --scheduler-param |  The scheduler you want to use and the decay rate of the scheduler.\n Current options: exp")
            print("exp: For exponential scheduler the decay should be a single value between 0 and 1. This is the factor by which the learning rate will be multiplied every epoch")
            print("-p --parallel   |  If you want to use multiple gpus")
            print("Default values: Model: Resnet50, Optimizer: Adam, Learning Rate: 0.001, Epochs: 100, Batch Size: 128, Scheduler: None, Output File: train_0")
            sys.exit()
        if opt in ("-l", "--learning-rate"):
            __learning_rate = float(arg)
            if (__learning_rate > 1 or __learning_rate <= 0):
                print("Invalid Learning Rate")
                sys.exit(2)
        elif opt in ("-M", "--model"):
            __model_type = arg
            if (__model_type != 'Resnet50' and __model_type != 'Efficientnet_b0'):
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
            if __scheduler not in ['exp']:
                print("Invalid Scheduler")
                sys.exit(2)
        elif opt in ("-s", "--scheduler-rate"):
            __scheduler_rate = float(arg)
            if (__scheduler_rate > 1 or __scheduler_rate <= 0):
                print("Invalid Scheduler Rate")
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
    scheduler_rate = __scheduler_rate if __scheduler_rate != 0 else 0.97
    model_save = 'train_0' if __output_file == '' else __output_file
    
    SAVE_PATH = PATH + model_type + '/' + model_save 
    del model_save
    
    print("Model: ", model_type)
    print("Optimizer: ", optimizer_type)
    print("Epochs: ", num_epochs)
    print("Batch Size: ", batch_size)
    print("Learning Rate: ", initial_learning_rate)
    if scheduler_type != 'None':
        print("Scheduler: ", scheduler_rate)
    print("Parallel: ",__parallel)
    print("Output File: ", SAVE_PATH)
    

    
    if model_type == 'Resnet50' :
        from Models.Resnet50 import model
    elif model_type == 'Efficientnet_b0' :
        from Models.Efficientnet_b0 import model
        
    if __parallel:
        model = torch.nn.DataParallel(model)
    
    model = model.to(device)
    summary(model, (3,224,224))
    
    
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
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, scheduler_rate)
    elif scheduler_type == 'None':
        scheduler = None
    
    train_model(model, optimizer, criterion, scheduler, device, train_loader, num_epochs, SAVE_PATH, valid_loader)
    
    
    del model, criterion, optimizer, scheduler
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main(sys.argv[1:])
