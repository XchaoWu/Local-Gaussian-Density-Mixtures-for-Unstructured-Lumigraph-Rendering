import torch
import torch.nn as nn 
import math 

# def warp_weight_warming_func(weight, step):
#     return weight * (1 - np.exp(-step / 2000))

def decay_func1(step, decay_step, decay_rate):
    return decay_rate ** ( (step / decay_step) ** 0.1)

def decay_func2(step, decay_step, decay_rate):
    return decay_rate ** ( (step / decay_step))


class Scheduler:
    def __init__(self, name, start_eta, end_eta, iterations, groups=[], decay_rate=0.1, 
                 decay_steps=None, start_itr=0, end_itr=100000000, decay_func=2):
        
        if decay_steps == None:
            self.decay_steps = iterations / math.log(end_eta/start_eta, decay_rate)
        else:
            self.decay_steps = decay_steps
            
        self.decay_rate = decay_rate
        
        self.start_eta = start_eta
        self.eta = start_eta
        self.groups = groups
        self.name = name 
        self.start_itr = start_itr 
        self.end_itr = end_itr

        if decay_func == 1:
            self.decay_func = decay_func1
        elif decay_func == 2:
            self.decay_func = decay_func2
    
    def step(self, global_step, optimizer):
        if global_step < self.start_itr or global_step >= self.end_itr:
            self.eta = 0
        else:
            # self.eta = self.start_eta * (self.decay_rate ** (global_step / self.decay_steps) )
            self.eta = self.start_eta * self.decay_func(global_step, self.decay_steps, self.decay_rate)
            # if self.name == "cam" and global_step >= 1000:
            #     self.eta = self.eta * 0.5

        if len(self.groups) == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.eta
        else:
            for idx in self.groups:
                optimizer.param_groups[idx]['lr'] = self.eta

class SchedulerManager:
    def __init__(self, scheduler_list):
        self.scheduler_list = scheduler_list
    
    def getEta(self):
        eta_list = []
        name_list = []
        for sche in self.scheduler_list:
            eta_list += [sche.eta]
            name_list += [sche.name]
        return eta_list, name_list
    
    def getInfo(self, writer=None, global_step=None):
        info = ""
        eta_list, name_list = self.getEta()
        for name, eta in zip(name_list, eta_list):
            info += "Eta %-10s\t%.8f\n" % (name, eta)
            if writer != None:
                writer.add_scalar(name, eta, global_step)
            # info += f"{name}:\t{eta:.8f}\n" 
        return info 

    def step(self, global_step, optimizer):
        for sche in self.scheduler_list:
            sche.step(global_step, optimizer)
