# -*- coding: utf-8 -*-
"""
@Author: Thanasis
"""

from torch.optim.lr_scheduler import ExponentialLR
import sys

def parse_args(opt_arg):
    if opt_arg[0] != '(' and opt_arg[len(opt_arg)-1] != ')':
        print("Parametres for EXP Scheduler should be of type: \"(gamma=0.xy)\"")
        sys.exit(2)
    opt_arg = opt_arg[1:(len(opt_arg)-1)]
    args = opt_arg.split(', ')
    args = list(arg.split('=') for arg in args)
    kwargs = {}
    for opt, arg in args:
        try:
            kwargs[opt] = float(arg)
        except ValueError:
            kwargs[opt] = arg
            
    print("Scheduler Arguments: ", kwargs)
    return kwargs

def get_scheduler(optimizer, scheduler_args):
    if scheduler_args == '':
        print("No Scheduler Arguments given thus no Scheduler will be used.")
        return None
    kwargs = parse_args(scheduler_args)
    
    for opt in kwargs.keys():
        if opt not in ['gamma']:
            raise ValueError(f'Invalid argument: {opt}')
            sys.exit(2)
    
    return ExponentialLR(optimizer, **kwargs)

