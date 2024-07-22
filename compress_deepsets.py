from data import DeepsetsDataset
import torch
import torch.nn as nn
from models.blocks import *
from utils.processor import evaluate_Deepsets, get_acc
import torch.nn.utils.prune as prune
import brevitas.nn as qnn


bit_width = 32

aggregator = lambda x: torch.mean(x,dim=2)

phi = QAT_ConvPhi(
    widths=[3,32,32,32], 
    acts=[nn.ReLU(), nn.ReLU(), nn.ReLU()], 
    norms=[None, None, None],
    bit_width = bit_width
    )

rho = QAT_Rho(
    widths=[32,16,5], 
    acts=[nn.ReLU(), None], 
    norms=[None, None],
    bit_width = bit_width
    )

deepsets_model = DeepSetsArchitecture(phi, rho, aggregator)


large_phi = QAT_ConvPhi(
    widths=[3,32,32], 
    acts=[nn.ReLU(), nn.ReLU()], 
    norms=['batch', 'batch'],
    bit_width = bit_width
    )

large_rho = QAT_Rho(
    widths=[32,32,64,5], 
    acts=[nn.ReLU(),nn.ReLU(),nn.LeakyReLU(negative_slope=0.01)], 
    norms=['batch', None, 'batch'],
    bit_width = bit_width
    )

large_model = DeepSetsArchitecture(large_phi, large_rho, aggregator)

medium_phi = QAT_ConvPhi(
    widths=[3,32,16], 
    acts=[nn.ReLU(),nn.ReLU()], 
    norms=['batch', 'batch'],
    bit_width = bit_width
    )

medium_rho = QAT_Rho(
    widths=[16,64,8,32,5], 
    acts=[nn.ReLU(),nn.LeakyReLU(negative_slope=0.01),nn.ReLU(),nn.ReLU()], 
    norms=['batch','batch','batch','batch'],
    bit_width = bit_width
    )

medium_model = DeepSetsArchitecture(medium_phi, medium_rho, aggregator)

small_phi = QAT_ConvPhi(
    widths=[3,8,8], 
    acts=[nn.LeakyReLU(negative_slope=0.01),nn.ReLU()], 
    norms=['batch', None],
    bit_width = bit_width
    )

small_rho = QAT_Rho(
    widths=[8,16,16,5], 
    acts=[nn.LeakyReLU(negative_slope=0.01),nn.ReLU(),nn.LeakyReLU(negative_slope=0.01)], 
    norms=['batch','batch',None],
    bit_width = bit_width
    )

small_model = DeepSetsArchitecture(small_phi, small_rho, aggregator)

tiny_phi = QAT_ConvPhi(
    widths=[3,16], 
    acts=[nn.ReLU()], 
    norms=['batch'],
    bit_width = bit_width
    )

tiny_rho = QAT_Rho(
    widths=[16,8,8,4,5], 
    acts=[nn.ReLU(),None,nn.ReLU(),nn.ReLU()], 
    norms=['batch',None,None,'batch'],
    bit_width = bit_width
    )

tiny_model = DeepSetsArchitecture(tiny_phi, tiny_rho, aggregator)

adjusted_tiny_rho = QAT_Rho(
    widths=[16,8,4,5], 
    acts=[nn.ReLU(),nn.ReLU(),nn.ReLU()], 
    norms=['batch',None,'batch'],
    bit_width = bit_width
    )

adjusted_tiny_model = DeepSetsArchitecture(tiny_phi, adjusted_tiny_rho, aggregator)

# def get_parameters_to_prune(model, bias = False):
#     parameters_to_prune = []
#     for name, module in model.named_modules():
#         if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
#             parameters_to_prune.append((module, 'weight'))
#             if bias and module.bias != None:
#                 parameters_to_prune.append((module, 'bias'))
        
#     return tuple(parameters_to_prune)

# def get_sparsities(model):
#     sparsities = []
#     for name, module in model.named_modules():
#         if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
#             layer_sparsity = torch.sum(module.weight_mask == 0).float() / module.weight_mask .numel()
#             sparsities.append(layer_sparsity)
#     return tuple(sparsities)

def get_sparsities(model):
    sparsities = {'phi': [], 'rho': []}
    
    if isinstance(model, DeepSetsArchitecture):
        for component, part in [('phi', model.phi), ('rho', model.rho)]:
            for name, module in part.named_modules():
                if isinstance(module, (qnn.QuantLinear, qnn.QuantConv1d, qnn.QuantConv2d)):
                    if hasattr(module.weight, 'mask'):
                        layer_sparsity = torch.sum(module.weight.mask == 0).float() / module.weight.mask.numel()
                        sparsities[component].append(layer_sparsity)
    
    return (tuple(sparsities['phi']), tuple(sparsities['rho']))

def get_parameters_to_prune(model):
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, (qnn.QuantLinear, qnn.QuantConv1d, qnn.QuantConv2d)):
            parameters_to_prune.append((module, 'weight'))
    return tuple(parameters_to_prune)


if __name__ == "__main__":
    # #TODO: Change to fit anyones device
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print('Device:', device)
    batch_size = 4096
    num_workers = 8

    train_loader, val_loader, test_loader = DeepsetsDataset.setup_data_loaders('jet_images_c8_minpt2_ptetaphi_robust_fast', batch_size, num_workers, prefetch_factor=True, pin_memory=True)
    print('Loaded Dataset...')

    for model, model_name in [(large_model, 'Large'), (medium_model, 'Medium'), (small_model, 'Small'), (tiny_model, 'Tiny')]:
        
        parameters_to_prune = get_parameters_to_prune(model)
        prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=0)
        
        for prune_iter in range(0, 20):
            val_accuracy, inference_time, validation_loss, param_count = evaluate_Deepsets(model, train_loader, val_loader, device, num_epochs=100)
            test_accuracy = get_acc(model, test_loader, device)
            
            phi_sparsities, rho_sparsities = get_sparsities(model)
            
            with open("./NAC_Compress.txt", "a") as file:
                file.write(f"Deepsets {model_name} Model {bit_width}-Bit QAT Model Prune Iter: {prune_iter}, "
                        f"Test Accuracy: {test_accuracy}, Val Accuracy: {val_accuracy}, Val Loss: {validation_loss}, "
                        f"Phi Sparsities: {phi_sparsities}, Rho Sparsities: {rho_sparsities}\n")
            
            if prune_iter < 19:  # Don't prune on the last iteration
                parameters_to_prune = get_parameters_to_prune(model)
                prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=0.2)