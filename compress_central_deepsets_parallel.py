from data import DeepsetsDataset
import torch
import torch.nn as nn
from models.blocks import *
from utils.processor import evaluate_Deepsets, get_acc
import torch.nn.utils.prune as prune
import brevitas.nn as qnn
import sys
import hashlib


bit_width = 32

aggregator = lambda x: torch.mean(x,dim=2)

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
                    if hasattr(module, 'weight'):
                        if hasattr(module.weight, 'mask'):
                            layer_sparsity = torch.sum(module.weight.mask == 0).float() / module.weight.mask.numel()
                        else:
                            layer_sparsity = torch.sum(module.weight == 0).float() / module.weight.numel()
                        sparsities[component].append(layer_sparsity)
                        print(f"{component} - {name}: Sparsity = {layer_sparsity}")
    
    return (tuple(sparsities['phi']), tuple(sparsities['rho']))

def get_parameters_to_prune(model):
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, (qnn.QuantLinear, qnn.QuantConv1d, qnn.QuantConv2d)):
            parameters_to_prune.append((module, 'weight'))
    return tuple(parameters_to_prune)


def create_model(model_size, bit_width):
    # Function to create models based on size and bit width
    if model_size == 'large':
        # phi = QAT_ConvPhi(
        phi = QAT_ConvPhi(
            widths=[3,64,16], 
            acts=[nn.ReLU(), None], 
            norms=['batch', 'batch'],
            bit_width=bit_width
        )
        rho = QAT_Rho(
            widths=[16, 128, 16, 64, 5], 
            acts=[nn.ReLU(),nn.LeakyReLU(negative_slope=1/128),nn.ReLU(), None], 
            norms=['batch', 'batch', 'batch', 'batch'],
            bit_width=bit_width
        )
    elif model_size == 'medium':
        phi = QAT_ConvPhi(
            widths=[3, 32,8], 
            acts=[nn.ReLU(), None], 
            norms=[None, None],
            bit_width=bit_width
        )
        rho = QAT_Rho(
            widths=[8, 32, 16, 64, 5], 
            acts=[nn.LeakyReLU(negative_slope=1/128),nn.LeakyReLU(negative_slope=1/128),nn.LeakyReLU(negative_slope=1/128),None], 
            norms=['batch',None, 'batch', None],
            bit_width=bit_width
        )
    elif model_size == 'small':
        phi = QAT_ConvPhi(
            widths=[3, 16, 8], 
            acts=[nn.LeakyReLU(negative_slope=1/128), None], 
            norms=[None, 'batch'],
            bit_width=bit_width
        )
        rho = QAT_Rho(
            widths=[8, 8, 16, 16, 5], 
            acts=[nn.ReLU(),nn.LeakyReLU(negative_slope=1/128), nn.LeakyReLU(negative_slope=1/128), None], 
            norms=['batch',None, None, 'batch'],
            bit_width=bit_width
        )
    elif model_size == 'tiny':
        phi = QAT_ConvPhi(
            widths=[3,8,8], 
            acts=[nn.ReLU(), None], 
            norms=['batch', None],
            bit_width=bit_width
        )
        rho = QAT_Rho(
            widths=[8, 32, 5], 
            acts=[nn.ReLU(),None], 
            norms=[ None, None],
            bit_width=bit_width
        )
    else:
        raise ValueError(f"Unknown model size: {model_size}")

    return DeepSetsArchitecture(phi, rho, aggregator)


if __name__ == "__main__":
    print("All command-line arguments:", sys.argv)
    model_index = int(sys.argv[1])
    print("model index: ", model_index)

    model_sizes = ['large', 'medium', 'small', 'tiny']
    bit_widths = [32, 16, 8, 4]  # Add more bit widths if needed

    model_size = model_sizes[model_index]
    print(f"Processing {model_size} model with index {model_index}")

    # Set up device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # Set up data loaders
    batch_size = 4096
    num_workers = 4
    train_loader, val_loader, test_loader = DeepsetsDataset.setup_data_loaders(
        'jet_images_c8_minpt2_ptetaphi_robust_fast',
        batch_size,
        num_workers,
        prefetch_factor=True,
        pin_memory=True
    )
    print('Loaded Dataset...')

    for bit_width in bit_widths:
        print(f"\nProcessing {model_size} model with {bit_width}-bit width")
        
        # Create model for current bit width
        model = create_model(model_size, bit_width)


        # Initial pruning
        parameters_to_prune = get_parameters_to_prune(model)
        prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=0)

        # Create a unique output file for this model and bit width
        output_file = f"./compression/deepsets_Compress_sep18_{model_size}_{bit_width}bit.txt"

        # Pruning loop
        for prune_iter in range(20):
            print(f"\nPrune iteration: {prune_iter}")

            # Evaluate the model
            val_accuracy, inference_time, validation_loss, param_count = evaluate_Deepsets(
                model, train_loader, val_loader, device, num_epochs=100
            )
            test_accuracy = get_acc(model, test_loader, device)

            # Get sparsities
            phi_sparsities, rho_sparsities = get_sparsities(model)

            # Log the results to the model-specific file
            with open(output_file, "a") as file:
                file.write(f"Deepsets {model_size.capitalize()} Model {bit_width}-Bit QAT Model Prune Iter: {prune_iter}, "
                           f"Test Accuracy: {test_accuracy}, Val Accuracy: {val_accuracy}, Val Loss: {validation_loss}, "
                           f"Phi Sparsities: {phi_sparsities}, Rho Sparsities: {rho_sparsities}\n")

            # Apply pruning for the next iteration
            if prune_iter < 19:  # Don't prune on the last iteration
                parameters_to_prune = get_parameters_to_prune(model)
                prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=0.2)
                print(f"Applied pruning with amount 0.2")

        print(f"Completed processing {model_size} model with {bit_width}-bit width. Results written to {output_file}")

    print(f"Completed processing all bit widths for {model_size} model.")