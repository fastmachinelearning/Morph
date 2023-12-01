#Examples of how optuna samples hyperparameters from our search space
Example1_params = {
    'Proj_outchannel': 3, 'b0': 'None', 'b1': 'Conv', 'b2': 'Conv', 'b1_Conv_channels_0': 4, 'b1_Conv_channels_1': 5, 'b1_Conv_kernels_0': 1, 'b1_Conv_kernels_1': 3, 'b1_Conv_norms_0': 'layer', 'b1_Conv_norms_1': None, 'b1_Conv_acts_0': 0, 'b1_Conv_acts_1': 1, 'b2_Conv_channels_0': 1, 'b2_Conv_channels_1': 1, 'b2_Conv_kernels_0': 3, 'b2_Conv_kernels_1': 3, 'b2_Conv_norms_0': None, 'b2_Conv_norms_1': None, 'b2_Conv_acts_0': 1, 'b2_Conv_acts_1': 1, 'MLP_width_0': 1, 'MLP_width_1': 2, 'MLP_width_2': 4, 'MLP_acts_0': 3, 'MLP_acts_1': 3, 'MLP_acts_2': 1, 'MLP_acts_3': 0, 'MLP_norms_0': 'layer', 'MLP_norms_1': 'batch', 'MLP_norms_2': 'batch', 'MLP_norms_3': None
}
Example2_params = {
    'Proj_outchannel': 0, 'b0': 'Conv', 'b1': 'Conv', 'b2': 'Conv', 'b0_Conv_channels_0': 5, 'b0_Conv_channels_1': 4, 'b0_Conv_kernels_0': 3, 'b0_Conv_kernels_1': 3, 'b0_Conv_norms_0': 'batch', 'b0_Conv_norms_1': 'layer', 'b0_Conv_acts_0': 0, 'b0_Conv_acts_1': 0, 'b1_Conv_channels_0': 2, 'b1_Conv_channels_1': 2, 'b1_Conv_kernels_0': 3, 'b1_Conv_kernels_1': 3, 'b1_Conv_norms_0': 'layer', 'b1_Conv_norms_1': 'batch', 'b1_Conv_acts_0': 1, 'b1_Conv_acts_1': 1, 'b2_Conv_channels_0': 2, 'b2_Conv_channels_1': 2, 'b2_Conv_kernels_0': 1, 'b2_Conv_kernels_1': 3, 'b2_Conv_norms_0': 'batch', 'b2_Conv_norms_1': 'batch', 'b2_Conv_acts_0': 2, 'b2_Conv_acts_1': 0, 'MLP_width_0': 1, 'MLP_width_1': 4, 'MLP_width_2': 3, 'MLP_acts_0': 0, 'MLP_acts_1': 2, 'MLP_acts_2': 0, 'MLP_acts_3': 0, 'MLP_norms_0': None, 'MLP_norms_1': 'layer', 'MLP_norms_2': 'batch', 'MLP_norms_3': None
}
Example3_params = {
    'Proj_outchannel': 2, 'b0': 'Conv', 'b1': 'None', 'b2': 'Conv', 'b0_Conv_channels_0': 2, 'b0_Conv_channels_1': 2, 'b0_Conv_kernels_0': 1, 'b0_Conv_kernels_1': 1, 'b0_Conv_norms_0': None, 'b0_Conv_norms_1': None, 'b0_Conv_acts_0': 3, 'b0_Conv_acts_1': 3, 'b2_Conv_channels_0': 1, 'b2_Conv_channels_1': 4, 'b2_Conv_kernels_0': 1, 'b2_Conv_kernels_1': 1, 'b2_Conv_norms_0': 'layer', 'b2_Conv_norms_1': 'layer', 'b2_Conv_acts_0': 0, 'b2_Conv_acts_1': 0, 'MLP_width_0': 1, 'MLP_width_1': 2, 'MLP_width_2': 3, 'MLP_acts_0': 2, 'MLP_acts_1': 0, 'MLP_acts_2': 2, 'MLP_acts_3': 0, 'MLP_norms_0': 'layer', 'MLP_norms_1': 'batch', 'MLP_norms_2': 'batch', 'MLP_norms_3': None
}

#Equivalent hyperparams for recreating OpenHLS
OpenHLS_params = {
    'b0': 'ConvAttn',
    'b1': 'Conv',
    'b2': 'None',
    'Proj_outchannel': 1,
    'b0_ConvAttn_hiddenchannel' : 3,
    'b0_ConvAttn_act' : 0,
    'b1_Conv_channels_0' : 2,
    'b1_Conv_channels_1' : 0,
    'b1_Conv_kernels_0' : 3,
    'b1_Conv_kernels_1' : 3,
    'b1_Conv_acts_0' : 0,
    'b1_Conv_acts_1' : 0,
    'b1_Conv_norms_0' : None, 
    'b1_Conv_norms_1' : None,
    'MLP_width_0' : 2,
    'MLP_width_1' : 1,
    'MLP_width_2' : 0,
    'MLP_acts_0' : 0,
    'MLP_acts_1' : 0,
    'MLP_acts_2' : 0,
    'MLP_acts_3' : 0,
    'MLP_norms_0' : None,
    'MLP_norms_1' : None,
    'MLP_norms_2' : None,
    'MLP_norms_3' : None,
    }

#Equivalent hyperparams for recreating BraggNN
BraggNN_params = {
    'b0': 'ConvAttn',
    'b1': 'Conv',
    'b2': 'None',
    'Proj_outchannel': 3,
    'b0_ConvAttn_hiddenchannel' : 5,
    'b0_ConvAttn_act' : 2,
    'b1_Conv_channels_0' : 4,
    'b1_Conv_channels_1' : 2,
    'b1_Conv_kernels_0' : 3,
    'b1_Conv_kernels_1' : 3,
    'b1_Conv_acts_0' : 2,
    'b1_Conv_acts_1' : 2,
    'b1_Conv_norms_0' : None,
    'b1_Conv_norms_1' : None,
    'MLP_width_0' : 4,
    'MLP_width_1' : 3,
    'MLP_width_2' : 2,
    'MLP_acts_0' : 2,
    'MLP_acts_1' : 2,
    'MLP_acts_2' : 2,
    'MLP_acts_3' : 3,
    'MLP_norms_0' : None,
    'MLP_norms_1' : None,
    'MLP_norms_2' : None,
    'MLP_norms_3' : None,
    }
