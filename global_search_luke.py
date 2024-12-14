import argparse
from data import BraggnnDataset, DeepsetsDataset
#from data.BraggnnDataset import setup_data_loaders
import torch
import torch.nn as nn
import optuna
from models.blocks import *
from utils.processor import evaluate_BraggNN, evaluate_Deepsets
from utils.bops import *
from examples.hyperparam_examples import OpenHLS_params, BraggNN_params, Example1_params, Example2_params, Example3_params
import sys

"""
Optuna Objective to evaluate a trial
1) Samples architecture from hierarchical search space
2) Trains Model
3) Evaluates Mean Distance, bops, param count, inference time, and val loss
Saves all information in global_search.txt
"""



def parse_args():
    parser = argparse.ArgumentParser(description="Run Global Search")
    parser.add_argument("--config_file", default="") #config file for training hyperparameters
    parser.add_argument("--search_space", type=str, default='BraggNN', 
                        choices=['BraggNN', 'Deepsets'])
    parser.add_argument("--dataset", type=str, default='BraggNN', 
                        choices=['BraggNN', 'Deepsets'])
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--n_trials", type=int, default=1000) #NSGA-II number of trials
    parser.add_argument("--population_size", type=int, default=20) #NSGA-II population size
    parser.add_argument("--device", type=str, default='cuda:0') #Device to run global search

    args = parser.parse_args()
    return args







# def sample_MLP(trial, in_dim, out_dim, prefix = 'MLP', num_layers = 4):
#     width_space = (4,8,16,32,64)
#     act_space = (nn.ReLU(), nn.LeakyReLU(negative_slope=0.01), None)
#     norm_space = (None, 'batch') #Note: Removed layer norm!
#     widths = [in_dim] + [width_space[trial.suggest_int(prefix + '_width_' + str(i), 0, len(width_space) - 1)] for i in range(num_layers-1)] + [out_dim]
#     acts = [act_space[trial.suggest_categorical(prefix + '_acts_' + str(i), (0,1,2))] for i in range(num_layers)]
#     norms = [trial.suggest_categorical(prefix + '_norms_' + str(i), norm_space) for i in range(num_layers)]
#     return widths, acts, norms


def get_conv1d_bops(layer, input_shape, bit_width=32):
    output_shape = (input_shape[0], layer.out_channels, input_shape[2] - layer.kernel_size[0] + 1)
    input_numel = torch.prod(torch.tensor(input_shape[1:]))
    output_numel = torch.prod(torch.tensor(output_shape[1:]))
    
    sparsity = get_sparsity(layer.weight.data)
    return output_numel * input_numel * layer.kernel_size[0] * ((1-sparsity) * bit_width**2 + 2*bit_width + math.log2(input_numel * layer.kernel_size[0]))

def BraggNN_objective(trial):
    #Build Model
    num_blocks = 3
    channel_space = (8,16,32,64)
    block_channels = [ channel_space[trial.suggest_int('Proj_outchannel', 0, len(channel_space) - 1) ] ] #sample the first channel dimension, save future dimensions here
    
    #Sample Block Types
    b = [trial.suggest_categorical('b' + str(i), ['Conv', 'ConvAttn', 'None']) for i in range(num_blocks)]

    Blocks = [] #Save list of blocks
    img_size = 9 #Size after first conv patch embedding
    bops = 0 #Record Estimated BOPs

    #Build Blocks
    for i, block_type in enumerate(b):
        if block_type == 'Conv':
            #Create block and add to Blocks
            channels, kernels, acts, norms = sample_ConvBlock(trial, 'b' + str(i) + '_Conv', block_channels[-1])
            reduce_img_size = 2*sum([1 if k == 3 else 0 for k in kernels]) #amount the image size will be reduced by kernel size, assuming no padding
            while img_size - reduce_img_size <= 0:
                kernels[kernels.index(3)] = 1
                reduce_img_size = 2*sum([1 if k == 3 else 0 for k in kernels])
            Blocks.append(ConvBlock(channels, kernels, acts, norms, img_size))

            #Calculate bops for this block
            bops += get_Conv_bops(Blocks[-1], input_shape = [batch_size, channels[0], img_size, img_size], bit_width=32)
            img_size -= reduce_img_size
            block_channels.append(channels[-1]) #save the final out dimension so next block knows what to expect

        elif block_type == 'ConvAttn':
            #Create block and add to Blocks
            hidden_channels, act = sample_ConvAttn(trial, 'b' + str(i) + '_ConvAttn')
            Blocks.append(ConvAttn(block_channels[-1], hidden_channels, act))

            #Calculate bops for this block
            bops += get_ConvAttn_bops(Blocks[-1], input_shape = [batch_size, block_channels[-1], img_size, img_size], bit_width=32)
            #Note: ConvAttn does not change the input shape because we use a skip connection
    
    #Build MLP
    in_dim = block_channels[-1] * img_size**2 #this assumes spatial dim stays same with padding trick
    widths, acts, norms = sample_MLP(trial, in_dim)
    mlp = MLP(widths, acts, norms)

    #Calculate bops for the mlp
    bops +=  get_MLP_bops(mlp, bit_width=32)
    
    #Initialize Model
    Blocks = nn.Sequential(*Blocks)
    model = CandidateArchitecture(Blocks, mlp, block_channels[0])
    bops += get_conv2d_bops(model.conv, input_shape = [batch_size, 1, 11, 11], bit_width=32) #Calculate bops for the patch embedding

    #Evaluate Model
    print(model)
    print('BOPs:', bops)
    print('Trial ', trial.number,' begins evaluation...')
    mean_distance, inference_time, validation_loss, param_count = evaluate_BraggNN(model, train_loader, val_loader, device)
    with open("./global_search.txt", "a") as file:
        file.write(f"Trial {trial.number}, Mean Distance: {mean_distance}, BOPs: {bops}, Inference time: {inference_time}, Validation Loss: {validation_loss}, Param Count: {param_count}, Hyperparams: {trial.params}\n")
    return mean_distance, bops

def Deepsets_objective(trial):

    # print("saving to file: ", f"./global_search_{model_name}.txt")

    bops = 0
    in_dim, out_dim = 3, 5

    bottleneck_dim = 2**trial.suggest_int('bottleneck_dim', 0, 6)

    # aggregator_space = [lambda x: torch.mean(x,dim=1), lambda x: torch.max(x,dim=1).values]
    aggregator_space = [lambda x: torch.mean(x, dim=2), lambda x: torch.max(x, dim=2).values]
    aggregator_type = trial.suggest_int('aggregator_type', 0,1)
    if aggregator_type == 0:
        bops += get_AvgPool_bops(input_shape=(8, bottleneck_dim), bit_width=8)
    else:
        bops += get_MaxPool_bops(input_shape=(8, bottleneck_dim), bit_width=8)
    aggregator = aggregator_space[aggregator_type]

    #Initialize Phi (first MLP)
    phi_len = trial.suggest_int('phi_len', 1, 4)
    widths, acts, norms = sample_MLP(trial, in_dim, bottleneck_dim, 'phi_MLP', num_layers=phi_len)
    # phi = Phi(widths, acts, norms) #QAT_Phi(widths, acts, norms)
    # phi = ConvPhi(widths, acts, norms)
    phi = ConvPhi(widths, acts, norms)
    # bops +=  get_MLP_bops(phi, bit_width=8)*8
    phi_input_shape = (batch_size, in_dim, 8)  # 8 is the sequence length from your input
    bops += get_Conv_bops(phi, input_shape=phi_input_shape, bit_width=8)
    print("Conv bp: ",get_Conv_bops(phi, input_shape=phi_input_shape, bit_width=8))
    # for i, layer in enumerate(phi.layers):
    #     if isinstance(layer, nn.Conv1d):
    #         bops += get_conv1d_bops(layer, phi_input_shape, bit_width=8)
    #         phi_input_shape = (phi_input_shape[0], layer.out_channels, phi_input_shape[2] - layer.kernel_size[0] + 1)


    #Initialize Rho (second MLP)
    rho_len = trial.suggest_int('rho_len', 1, 4)
    widths, acts, norms = sample_MLP(trial, bottleneck_dim, out_dim, 'rho_MLP', num_layers=rho_len)
    rho = Rho(widths, acts, norms) #QAT_Rho(widths, acts, norms)
    # rho = ConvRho(widths, acts, norms)
    bops +=  get_MLP_bops(rho, bit_width=8)


    # #Initialize Rho (second MLP)
    # rho_len = trial.suggest_int('rho_len', 1, 4)
    # widths, acts, norms = sample_MLP(trial, bottleneck_dim, out_dim, 'rho_MLP', num_layers=rho_len)
    # rho = Rho(widths, acts, norms) #QAT_Rho(widths, acts, norms)

    rho_bops = get_MLP_bops(rho, bit_width=8)
    bops += rho_bops
    
    model = DeepSetsArchitecture(phi, rho, aggregator)

    print(model)
    print('BOPs:', bops)
    print('Trial ', trial.number,' begins evaluation...')
    accuracy, inference_time, validation_loss, param_count = evaluate_Deepsets(model, train_loader, val_loader, device)
    with open(f"./global_search_results_central2/global_search_{model_name}.txt", "a") as file:
        file.write(f"Trial {trial.number}, Accuracy: {accuracy}, BOPs: {bops}, Inference time: {inference_time}, Validation Loss: {validation_loss}, Param Count: {param_count}, Hyperparams: {trial.params}\n")
    
    return accuracy, bops

if __name__ == "__main__":
    args = parse_args()

    #TODO: make config_parse whatever, parse config into python dictionary
    training_config = config_parse(args.config_file) #this should be a python dicitonary

    if args.dataset == 'BraggNN':
        train_loader, val_loader, test_loader = BraggNNDataset.setup_data_loaders(batch_size=training_configs['batch_size'], 
                                                                                IMG_SIZE = 11, 
                                                                                aug=1, 
                                                                                num_workers=4, 
                                                                                pin_memory=False, 
                                                                                prefetch_factor=2)
    elif args.dataset == 'Deepsets':
        train_loader, val_loader, test_loader = DeepsetsDataset.setup_data_loaders('jet_images_c8_minpt2_ptetaphi_robust_fast', batch_size, num_workers=4, prefetch_factor=True, pin_memory=True)

    device = torch.device(args.device) #args.deivce should be like 'cuda:0', or 'cpu', or others.



    batch_size = 4096 #1024
    num_workers = 4

    print('Loaded Dataset...')


    #printing input shape
    for batch in train_loader:
        inputs, targets = batch
        print("input shape: ", inputs.shape)
        break

    study = optuna.create_study(sampler=optuna.samplers.NSGAIISampler(population_size=20), directions=['maximize', 'minimize'])

    # Enqueue the selected model
    study.enqueue_trial(model_params)

    """
    study = optuna.create_study(sampler=optuna.samplers.NSGAIISampler(population_size = 20), directions=['minimize', 'minimize']) #min mean_distance and inference time
    
    #Queue OpenHLS & BraggNN architectures to show the search strategy what we want to beat.
    study.enqueue_trial(OpenHLS_params)
    study.enqueue_trial(BraggNN_params)
    study.enqueue_trial(Example1_params)
    study.enqueue_trial(Example2_params)
    study.enqueue_trial(Example3_params)
    study.optimize(BraggNN_objective, n_trials=1000)
    """


    study.optimize(Deepsets_objective, n_trials=500)
