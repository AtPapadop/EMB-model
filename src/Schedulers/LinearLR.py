# -*- coding: utf-8 -*-
"""
@Author: Thanasis
"""

from torch.optim.lr_scheduler import LinearLR
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

def get_scheduler(optimizer, scheduler_args, lr):
    if scheduler_args == '':
        print("No Scheduler Arguments given thus no Scheduler will be used.")
        return None
    
    temp_kwargs = parse_args(scheduler_args)
    kwargs = {}
    for opt, arg in temp_kwargs.items():
        if opt not in ['start_lr', 'end_lr', 'iters']:
            raise ValueError(f'Invalid argument: {opt}')
            sys.exit(2)
        if opt == 'start_lr':
            kwargs['start_factor'] = arg/lr
        elif opt == 'end_lr':
            kwargs['end_factor'] = arg/lr
        else:
            kwargs['total_iters'] = arg
    
    return LinearLR(optimizer, **kwargs)
    