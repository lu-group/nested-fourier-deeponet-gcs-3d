import numpy as np
import torch
import os

## Before run the code, put nested fno datasets in '../datasets_nested_fno'
## Generate GLOBAL data
task = 'dP'
level = 'GLOBAL'
load_path = f'../datasets_nested_fno/{task}_{level}/'
save_path = f'./{task}_{level}/'
if not os.path.exists(save_path):
    os.mkdir(save_path)

for dataset_type in ['train', 'test', 'val']:
    names_list = np.loadtxt(f'dP0_{dataset_type}.txt', dtype=str)

    inputs = []
    outputs = []

    for idx, names in enumerate(names_list):
        npz_data = torch.load(load_path + names + '.npz')
        inputs.append(npz_data['input'][:, :, :, :, 0, [0, 1, 2, 4, 5, 6, 7]])
        outputs.append(npz_data['output'])
    inputs = np.concatenate(inputs)
    outputs = np.concatenate(outputs)
    np.savez_compressed(f'{save_path}{task}_{level}_{dataset_type}_input.npz', input=inputs)
    np.savez_compressed(f'{save_path}{task}_{level}_{dataset_type}_output.npz', output=outputs)

    del inputs, outputs


## Generate LGR data
for dataset_type in ['train', 'test', 'val']:
    names_list = np.loadtxt(f'level1-4_{dataset_type}.txt', dtype=str)

    for task in ['dP', 'SG']:
        for level in ['LGR1','LGR2', 'LGR3', 'LGR4']:

            load_path = f'../datasets_nested_fno/{task}_{level}/'
            save_path = f'./{task}_{level}/'
            if not os.path.exists(save_path):
                os.mkdir(save_path)

            inputs1 = []
            inputs2 = []
            outputs = []

            for idx, names in enumerate(names_list):
                npz_data = torch.load(load_path + names[:-11] + level + '_' + names[-11:])
                inputs1.append(npz_data['input'][:, :, :, :, 0, [0, 1, 2, 4, 5, 6, 7]])
                inputs2.append(npz_data['input'][:, :, :, :, :, -1])
                outputs.append(npz_data['output'])
            inputs1 = np.concatenate(inputs1)
            inputs2 = np.concatenate(inputs2)
            outputs = np.concatenate(outputs)
            np.savez_compressed(f'{save_path}{task}_{level}_{dataset_type}_input1.npz', input=inputs1)
            np.savez_compressed(f'{save_path}{task}_{level}_{dataset_type}_input2.npz', input=inputs2)
            np.savez_compressed(f'{save_path}{task}_{level}_{dataset_type}_output.npz', output=outputs)

            del inputs1, inputs2, outputs


