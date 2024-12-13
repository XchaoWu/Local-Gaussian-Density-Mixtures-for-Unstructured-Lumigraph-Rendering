import torch 
import numpy as np 
from easydict import EasyDict as edict



def MSE(pred, gt, **kwargs):
    return torch.mean((pred - gt) ** 2)


def weight_warming_func(global_step):
    return min(global_step / 15000, 1.0)
    # return 1 - np.exp(-step / 10000)


class Criterion:
    def __init__(self):
        
        
        self.loss_items = {}

    def append(self, loss_func, loss_name, weight, warm_up=False): 

        
        self.loss_items[loss_name] = edict({"func": loss_func, 
                                            "name": loss_name,
                                            "weight": weight,
                                            "warm_up": warm_up,
                                            "cur_weight": weight,
                                            "record": []})
        
        
    def compute_loss(self, loss_name, pred, gt, global_step, **kwargs):
        
        item = self.loss_items[loss_name]
        
        loss = item.func(pred, gt, **kwargs)
        item.record.append(float(loss))

        if item.warm_up:
            item.cur_weight = item.weight * weight_warming_func(global_step)
        else:
            item.cur_weight = item.weight

        return loss * item.cur_weight
    
    def getInfo(self):
        info = ""
        for k,v in self.loss_items.items():
            if len(v.record) == 0:
                continue 
            info += "%-20s\t%.8f\tweight: %.8f\n" % (k, np.mean(v.record), v.cur_weight)
            v.record = []
        return info
