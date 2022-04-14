
## You could add some configs to perform other training experiments...

myLeNet_cfg = {
    'model_type': 'myLeNet',
    'data_root' : './p2_data/annotations/train_annos.json',
    
    # ratio of training images and validation images 
    'split_ratio': 0.9,
    # set a random seed to get a fixed initialization 
    'seed': 687,
    
    # training hyperparameters
    'batch_size': 16,
    'lr':0.001,
    'milestones': [15, 25],
    'num_out': 10,
    'num_epoch': 40,
    
}

ResNet18_cfg = {
    'model_type': 'ResNet18',
    'data_root' : './p2_data/annotations/train_annos.json',
    
    # ratio of training images and validation images 
    'split_ratio': 0.9,
    # set a random seed to get a fixed initialization 
    'seed': 687,
    
    # training hyperparameters
    'batch_size': 16,
    'lr':0.001,
    'milestones': [15, 25, 35],
    'num_out': 10,
    'num_epoch': 40,
}

ResNet34_cfg = {
    'model_type': 'ResNet34',
    'data_root' : './p2_data/annotations/train_annos.json',
    
    # ratio of training images and validation images 
    'split_ratio': 0.9,
    # set a random seed to get a fixed initialization 
    'seed': 687,
    
    # training hyperparameters
    'batch_size': 16,
    'lr':0.001,
    'milestones': [15, 25, 35],
    'num_out': 10,
    'num_epoch': 40,
}

myResNet18_cfg = {
    'model_type': 'myResNet18',
    'data_root' : './p2_data/annotations/train_annos.json',
    
    # ratio of training images and validation images 
    'split_ratio': 0.9,
    # set a random seed to get a fixed initialization 
    'seed': 687,
    
    # training hyperparameters
    'batch_size': 16,
    'lr':0.001,
    'milestones': [15, 25, 35],
    'num_out': 10,
    'num_epoch': 40,
}

myResNet34_cfg = {
    'model_type': 'myResNet34',
    'data_root' : './p2_data/annotations/train_annos.json',
    
    # ratio of training images and validation images 
    'split_ratio': 0.9,
    # set a random seed to get a fixed initialization 
    'seed': 687,
    
    # training hyperparameters
    'batch_size': 16,
    'lr':0.001,
    'milestones': [15, 25, 35],
    'num_out': 10,
    'num_epoch': 40,
}