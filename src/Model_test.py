# -*- coding: utf-8 -*-
"""
@Author: Thanasis
"""

import sys, getopt

def main(argv):
    
    __input_file = ''
    __validate = False
    try:
        opts, args = getopt.getopt(argv,"hi:V", ["help", "input-file=", "validation"])
    except getopt.GetoptError():
        sys.exit(2)
        
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("Model_test.py -i <input-file> | <input-file> should be of the form <model_type>/<model_name> e.g. Resnet50/train_0")
            print("-V (for validation mode)")
            sys.exit()
        elif opt in ("-i", "--input-file"):
            __input_file = arg
            if (__input_file == '' or __input_file.count('/') != 1 or __input_file.split('/')[0] not in ['Resnet50', 'Efficientnet_b0'] or __input_file.split('/')[1] == ''):
                print("Invalid Input File")
                sys.exit(2)
        elif opt in ("-V", "--validate"):
            print("Validation Mode")
            __validate = True
        else:
            print("Unknown Command-line arguement")
            sys.exit(2)
    
    import torch
    import torchvision
    import gc
    from Dataset import test_dataset, valid_dataset
    from torch.utils.data import DataLoader
    from ModelModules import test_model

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    
    PATH = '/home/a/atpapadop/EMB/models/'

    model_load = 'Resnet50/train_0' if __input_file == '' else __input_file
    
    if model_load == 'Resnet50/train_0':
        print("No input file given. Using default model\n If this is done by mistake press Ctrl+C and give the correct input file to avoid overwriting the model.")
    
    
    LOAD_PATH = PATH + model_load
    print("Input File: ", LOAD_PATH)
    
    model_type = model_load.split('/')[0]
    
    if model_type == 'Resnet50':
        from Models.Resnet50 import model
    elif model_type == 'Efficientnet_b0':
        from Models.Efficientnet_b0 import model
    else:
        print("Invalid Model Type")
        sys.exit(2)
    
    model = model.to(device)
    
    if __validate:
        test_loader = DataLoader(valid_dataset, shuffle=False)
    else:
        test_loader = DataLoader(test_dataset, shuffle=False)
    
    test_model(model, device, test_loader, LOAD_PATH)
    
    
    del model
    torch.cuda.empty_cache()
    gc.collect()
    

if __name__ == "__main__":
    main(sys.argv[1:])
