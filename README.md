# Provably Powerful Graph Networks
This repository holds the code for the paper:
https://arxiv.org/abs/1905.11136

## Data
Data should be downloaded using the following commands, and will be ready to use.

```
cd ProvablyPowerfulGraphNetworks
python utils/get_data.py
```

This script will download all the data from Dropbox links. 

### Regarding QM9       
The usage of QM9 dataset is based on the pythorch-geometric package,
see: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/qm9.html#QM9

Using the above script will download QM9 and there is no need to continue over this section.

If you are interested in parsing the QM9 data from source, you can use the same code we did to download the original dataset and parse it.
(this specific script requires pytorch).

```
cd ProvablyPowerfulGraphNetworks
python utils/get_qm9_data.py
```

## Code

### Prerequisites

python3

TensorFlow gpu 1.9.0.

Additional modules: numpy, pandas, matplotlib, tqdm, easydict



### Running the tests

The folder main_scripts contains scripts that run different experiments:
1. To run 10-fold cross-validation with our hyper parameters run the main_10fold_experiment.py script. You can choose the dataset in 10fold_config.json or using the command line switch. These hyper parameters refer to "Ours 1" version from the paper. 
example to run 10-fold cross-validation experiment:
```
python main_scripts/main_10fold_experiment --config=configs/10fold_config.json --dataset_name=NCI1
```
2. To run the QM9 experiment with out hyper parameters run the main_qm9_experiment.py script.
```
python main_scripts/main_qm9_experiment.py --config=configs/qm9_config.json
```

### Running other vesions from the paper
#### Ours2
in configs/10fold_config.json, change:
```
    "new_suffix": false
```
in utils/config.py change:
```
LEARNING_RATES = {'COLLAB': 0.0001, 'IMDBBINARY': 0.00005, 'IMDBMULTI': 0.0001, 'MUTAG': 0.0001, 'NCI1':0.0001, 'NCI109':0.0001, 'PROTEINS': 0.001, 'PTC': 0.0001}
DECAY_RATES = {'COLLAB': 0.5, 'IMDBBINARY': 0.5, 'IMDBMULTI': 0.75, 'MUTAG': 1.0, 'NCI1':0.75, 'NCI109':0.75, 'PROTEINS': 0.5, 'PTC': 1.0}
CHOSEN_EPOCH = {'COLLAB': 150, 'IMDBBINARY': 100, 'IMDBMULTI': 150, 'MUTAG': 500, 'NCI1': 200, 'NCI109': 250, 'PROTEINS': 100, 'PTC': 400}
```

#### Ours3
in configs/10fold_config.json, change:
```
    "block_features": [256,256,256],
    "depth_of_mlp": 3,
    "new_suffix": true
```
in utils/config.py change:
```
LEARNING_RATES = {'COLLAB': 0.0005, 'IMDBBINARY': 0.00001, 'IMDBMULTI': 0.0001, 'MUTAG': 0.0005, 'NCI1':0.0005, 'NCI109':0.0001, 'PROTEINS': 0.0005, 'PTC': 0.001}
DECAY_RATES = {'COLLAB': 0.5, 'IMDBBINARY': 0.75, 'IMDBMULTI': 1.0, 'MUTAG': 0.5, 'NCI1':0.75, 'NCI109': 1.0, 'PROTEINS': 0.75, 'PTC': 0.5}
CHOSEN_EPOCH = {'COLLAB': 85, 'IMDBBINARY': 100, 'IMDBMULTI': 150, 'MUTAG': 150, 'NCI1': 100, 'NCI109': 300, 'PROTEINS': 100, 'PTC': 200}
```

