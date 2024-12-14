import numpy as np
from tqdm import tqdm
import optuna
import torch
from data import BraggnnDataset
from data.BraggnnDataset import setup_data_loaders
from torch.optim.lr_scheduler import _LRScheduler
from functools import partial
import torch.nn as nn
import os
from datetime import datetime
# from models.train_utils import * # commented

from utils.processor import * # Added

from models.blocks import *
import torch.nn.utils.prune as prune

#Helper function for pruning
def get_parameters_to_prune(model, bias = False):
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            parameters_to_prune.append((module, 'weight'))
            if bias and module.bias != None:
                parameters_to_prune.append((module, 'bias'))
        
    return tuple(parameters_to_prune)

def get_sparsities(model):
    sparsities = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            layer_sparsity = torch.sum(module.weight_mask == 0).float() / module.weight_mask .numel()
            sparsities.append(layer_sparsity)
    return tuple(sparsities)

def main():
    #NAC Model
    b = 4 #Bit width 
    # Blocks = nn.Sequential(
    #     QAT_ConvBlock([32,4,32], [1,1], [nn.ReLU(), nn.LeakyReLU(negative_slope=0.01)], [None, 'batch'], img_size=9, bit_width=b),
    #     QAT_ConvBlock([32,4,32], [1,3], [nn.GELU(), nn.GELU()], ['batch', 'layer'], img_size=9, bit_width=b),
    #     QAT_ConvBlock([32,8,64], [3,3], [nn.GELU(), None], ['layer', None], img_size=7, bit_width=b),
    # ) 
    # mlp = QAT_MLP(widths=[576,8,4,4,2], acts=[nn.ReLU(), nn.GELU(), nn.GELU(), None], norms=['layer', None, 'layer', None], bit_width=b)
    # model = QAT_CandidateArchitecture(Blocks,mlp,32).to(device)

    Blocks = nn.Sequential(
        QAT_ConvBlock([8,64,32], [3,3], [None, None], ['batch', 'batch'], img_size=9, bit_width=b), #img_size = 9 -> 5
    ) 
    mlp = QAT_MLP(widths=[5*5*32,32,64,64,2], acts=[nn.LeakyReLU(negative_slope=1/128), nn.ReLU(), nn.LeakyReLU(negative_slope=1/128), None], norms=[None,None,'batch','batch'], bit_width=b)
    model = QAT_CandidateArchitecture(Blocks,mlp,8).to(device)

    prune.global_unstructured(get_parameters_to_prune(model), pruning_method=prune.L1Unstructured,amount=0)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=.0015, weight_decay=2.2e-9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
    
    for prune_iter in range(0,20):

        # validation_loss = train_model(model, optimizer, scheduler, criterion, train_loader, val_loader, device, num_epochs)
        # val_mean_dist = get_performance(model, val_loader, device, psz=11)
        # test_mean_dist = get_performance(model, test_loader, device, psz=11)

        validation_loss = train(model, optimizer, scheduler, criterion, train_loader, val_loader, device, num_epochs)
        # val_mean_dist = evaluate_BraggNN(model, val_loader, device, psz=11)
        # test_mean_dist = evaluate_BraggNN(model, test_loader, device, psz=11)
        val_mean_dist = get_mean_dist(model, val_loader, device, psz=11)
        test_mean_dist = get_mean_dist(model, test_loader, device, psz=11) # get_mean_dist is in utils.metrics
        
        sparsities = get_sparsities(model)
        with open("./NAC_Bragg_Compress.txt", "a") as file:
            file.write(f"Trial 608 {b}-Bit QAT Large Bragg Model Prune Iter: {prune_iter}, Test Mean Dist: {test_mean_dist}, Val Mean Dist: {val_mean_dist}, Val Loss: {validation_loss}, Sparsities: {sparsities}\n")
        
        prune.global_unstructured(get_parameters_to_prune(model), pruning_method=prune.L1Unstructured,amount=.2)

if __name__ == '__main__':
    batch_size=256
    IMG_SIZE = 11
    aug=1
    num_epochs= 3 # 300
    # device = torch.device('cuda:3') # commented out when running on cpu (9/8)
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    print(device)
    train_loader, val_loader, test_loader = setup_data_loaders(batch_size, IMG_SIZE = 11, aug=1, num_workers=4, pin_memory=False, prefetch_factor=2)
    print('Loaded Dataset...')
    main()