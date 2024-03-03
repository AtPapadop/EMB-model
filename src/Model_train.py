# -*- coding: utf-8 -*-
"""
@Author: Thanasis
"""

import sys, getopt

def main(argv):
    __learning_rate = __batch_size = __epochs = __scheduler = 0
    __output_file = ''
    __model_type = ''
    __parallel = False

    try:
        opts, args = getopt.getopt(argv, "hM:l:E:b:s:o:p", ["help", "model=", "learning-rate=", "epochs=", "batch-size=","scheduler=", "output-file=", "parallel"])
    except getopt.GetoptError():
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("Model_train.py -l <learning-rate> -E <epochs> -b <batch-size> -s <scheduler-rate> -o <output-file>")
            print("-M --model       |  The model you want to train.\n Current options: Resnet50, Efficientnet_b0")
            print("-p --parallel   |  If you want to use multiple gpus")
            print("Default values: Model: Resnet50, Learning Rate: 0.001, Epochs: 100, Batch Size: 128, Scheduler Rate: 0.97, Output File: train_0")
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
        elif opt in ("-s", "--scheduler"):
            __scheduler = float(arg)
            if (__scheduler > 1 or __scheduler <= 0):
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
    num_epochs = __epochs if __epochs != 0 else 100
    batch_size = __batch_size if __batch_size != 0 else 128
    initial_learning_rate = __learning_rate if __learning_rate != 0 else 0.001
    scheduler_rate = __scheduler if __scheduler != 0 else 0.97
    model_save = 'train_0' if __output_file == '' else __output_file
    
    SAVE_PATH = PATH + model_type + '/' + model_save 
    del model_save
    
    print("Model: ", model_type)
    print("Epochs: ", num_epochs)
    print("Batch Size: ", batch_size)
    print("Learning Rate: ", initial_learning_rate)
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
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, scheduler_rate)
    
    train_model(model, optimizer, criterion, scheduler, device, train_loader, num_epochs, SAVE_PATH, valid_loader)
    
    
    del model, criterion, optimizer, scheduler
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main(sys.argv[1:])
