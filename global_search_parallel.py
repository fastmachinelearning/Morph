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



# def sample_MLP(trial, in_dim, out_dim, prefix = 'MLP', num_layers = 4):
#     width_space = (4,8,16,32,64)
#     act_space = (nn.ReLU(), nn.LeakyReLU(negative_slope=0.01), None)
#     norm_space = (None, 'batch') #Note: Removed layer norm!
#     widths = [in_dim] + [width_space[trial.suggest_int(prefix + '_width_' + str(i), 0, len(width_space) - 1)] for i in range(num_layers-1)] + [out_dim]
#     acts = [act_space[trial.suggest_categorical(prefix + '_acts_' + str(i), (0,1,2))] for i in range(num_layers)]
#     norms = [trial.suggest_categorical(prefix + '_norms_' + str(i), norm_space) for i in range(num_layers)]
#     return widths, acts, norms



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
    phi = ConvPhi(widths, acts, norms)
    # bops +=  get_MLP_bops(phi, bit_width=8)*8
    phi_input_shape = (batch_size, in_dim, 8)  # 8 is the sequence length from your input
    bops += get_Conv_bops(phi, input_shape=phi_input_shape, bit_width=8)


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
    with open(f"./global_search_results2/global_search_{model_name}.txt", "a") as file:
        file.write(f"Trial {trial.number}, Accuracy: {accuracy}, BOPs: {bops}, Inference time: {inference_time}, Validation Loss: {validation_loss}, Param Count: {param_count}, Hyperparams: {trial.params}\n")
    
    return accuracy, bops

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python global_search.py <model_index>")
        sys.exit(1)

    print("All command-line arguments:", sys.argv)
    model_index = int(sys.argv[1])
    print("model index: ", model_index)

    model_configs = [
        {'name': 'Deepsets', 'params': {'bottleneck_dim': 5, 'aggregator_type': 0, 'phi_len': 3, 'phi_MLP_width_0': 3, 'phi_MLP_width_1': 3, 'phi_MLP_acts_0': 0, 'phi_MLP_acts_1': 0, 'phi_MLP_acts_2': 0, 'phi_MLP_norms_0': None, 'phi_MLP_norms_1': None, 'phi_MLP_norms_2': None, 'rho_len': 2, 'rho_MLP_width_0': 2, 'rho_MLP_acts_0': 0, 'rho_MLP_acts_1': 1, 'rho_MLP_norms_0': None, 'rho_MLP_norms_1': None}},
        {'name': 'large', 'params': {'bottleneck_dim': 5, 'aggregator_type': 0, 'phi_len': 2, 'phi_MLP_width_0': 3, 'phi_MLP_acts_0': 0, 'phi_MLP_acts_1': 0, 'phi_MLP_norms_0': 'batch', 'phi_MLP_norms_1': 'batch', 'rho_len': 3, 'rho_MLP_width_0': 3, 'rho_MLP_width_1': 4, 'rho_MLP_acts_0': 0, 'rho_MLP_acts_1': 0, 'rho_MLP_acts_2': 1, 'rho_MLP_norms_0': 'batch', 'rho_MLP_norms_1': None, 'rho_MLP_norms_2': 'batch'}},
        {'name': 'medium', 'params': {'bottleneck_dim': 4, 'aggregator_type': 0, 'phi_len': 2, 'phi_MLP_width_0': 3, 'phi_MLP_acts_0': 0, 'phi_MLP_acts_1': 0, 'phi_MLP_norms_0': 'batch', 'phi_MLP_norms_1': 'batch', 'rho_len': 4, 'rho_MLP_width_0': 4, 'rho_MLP_width_1': 1, 'rho_MLP_width_2': 3, 'rho_MLP_acts_0': 0, 'rho_MLP_acts_1': 1, 'rho_MLP_acts_2': 0, 'rho_MLP_acts_3': 0, 'rho_MLP_norms_0': 'batch', 'rho_MLP_norms_1': 'batch', 'rho_MLP_norms_2': 'batch', 'rho_MLP_norms_3': 'batch'}},
        {'name': 'small', 'params': {'bottleneck_dim': 3, 'aggregator_type': 0, 'phi_len': 2, 'phi_MLP_width_0': 1, 'phi_MLP_acts_0': 1, 'phi_MLP_acts_1': 0, 'phi_MLP_norms_0': 'batch', 'phi_MLP_norms_1': None, 'rho_len': 3, 'rho_MLP_width_0': 2, 'rho_MLP_width_1': 2, 'rho_MLP_acts_0': 1, 'rho_MLP_acts_1': 0, 'rho_MLP_acts_2': 1, 'rho_MLP_norms_0': 'batch', 'rho_MLP_norms_1': 'batch', 'rho_MLP_norms_2': None}},
        {'name': 'tiny', 'params': {'bottleneck_dim': 4, 'aggregator_type': 0, 'phi_len': 1, 'phi_MLP_acts_0': 0, 'phi_MLP_norms_0': 'batch', 'rho_len': 4, 'rho_MLP_width_0': 1, 'rho_MLP_width_1': 1, 'rho_MLP_width_2': 0, 'rho_MLP_acts_0': 0, 'rho_MLP_acts_1': 1, 'rho_MLP_acts_2': 0, 'rho_MLP_acts_3': 0, 'rho_MLP_norms_0': 'batch', 'rho_MLP_norms_1': None, 'rho_MLP_norms_2': None, 'rho_MLP_norms_3': 'batch'}}
    ]

    if model_index < 0 or model_index >= len(model_configs):
        print(f"Invalid model index. Please choose a number between 0 and {len(model_configs) - 1}")
        sys.exit(1)

    selected_model = model_configs[model_index]
    model_name = selected_model['name']
    model_params = selected_model['params']

    print(f"Processing {model_name} model with index {model_index}")


    
    # device = torch.device('cuda:0') #TODO: Change to fit anyones device
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')


    batch_size = 4096 #1024
    # batch_size = 1024
    num_workers = 4

    #train_loader, val_loader, test_loader = BraggNNDataset.setup_data_loaders(batch_size, IMG_SIZE = 11, aug=1, num_workers=4, pin_memory=False, prefetch_factor=2)
    train_loader, val_loader, test_loader = DeepsetsDataset.setup_data_loaders('jet_images_c8_minpt2_ptetaphi_robust_fast', batch_size, num_workers, prefetch_factor=True, pin_memory=True)
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
