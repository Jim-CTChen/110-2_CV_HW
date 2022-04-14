
import torch
import os
import argparse


from torch.utils.data import DataLoader
import torch.optim as optim 
import torch.nn as nn

from myModels import ResNet18, ResNet34, myLeNet, myResNet18, myResNet34
from myDatasets import  get_cifar10_train_val_set
from tool import train, fixed_seed, load_parameters

# Modify config if you are conducting different models
from cfg import myLeNet_cfg, myResNet18_cfg, myResNet34_cfg, ResNet18_cfg, ResNet34_cfg


def train_interface(model_name):
    # myLeNet
    if model_name == 'myLeNet':
        cfg = myLeNet_cfg

    # myResNet18
    elif model_name == 'myResNet18':
        cfg = myResNet18_cfg

    # myResNet34
    elif model_name == 'myResNet34':
        cfg = myResNet34_cfg

    # ResNet18
    elif model_name == 'ResNet18':
        cfg = ResNet18_cfg

    # ResNet34
    elif model_name == 'ResNet34':
        cfg = ResNet34_cfg
    
    """ input argumnet """

    data_root = cfg['data_root']
    model_type = cfg['model_type']
    num_out = cfg['num_out']
    num_epoch = cfg['num_epoch']
    split_ratio = cfg['split_ratio']
    seed = cfg['seed']
    
    # fixed random seed
    fixed_seed(seed)
    

    os.makedirs( os.path.join('./acc_log',  model_type), exist_ok=True)
    os.makedirs( os.path.join('./save_dir', model_type), exist_ok=True)    
    log_path = os.path.join('./acc_log', model_type, 'acc_' + model_type + '_.log')
    save_path = os.path.join('./save_dir', model_type)


    with open(log_path, 'w'):
        pass
    
    ## training setting ##
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'current device: {device}')
    #device = torch.device('cpu')
    
    
    """ training hyperparameter """
    lr = cfg['lr']
    batch_size = cfg['batch_size']
    milestones = cfg['milestones']
    
    
    ## Modify here if you want to change your model ##
    model = None

    # myLeNet
    if model_name == 'myLeNet':
        model = myLeNet(num_out=num_out)

    # myResNet18
    elif model_name == 'myResNet18':
        model = myResNet18(num_out=num_out)

    # myResNet34
    elif model_name == 'myResNet34':
        model = myResNet34(num_out=num_out)

    # ResNet18
    elif model_name == 'ResNet18':
        model = ResNet18(num_out=num_out)

    # ResNet34
    elif model_name == 'ResNet34':
        model = ResNet34(num_out=num_out)


    # load model if exist
    # path = os.path.join(save_path, 'epoch_25.pt')
    # load_parameters(model, path)

    # print model's architecture
    print(f'model: {model_type}')
    # print(model)

    # Get your training Data
    ## TO DO ##
    # You need to define your cifar10_dataset yourself to get images and labels for earch data
    # Check myDatasets.py
      
    train_set, val_set =  get_cifar10_train_val_set(root=data_root, ratio=split_ratio)    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    # define your loss function and optimizer to unpdate the model's parameters.
    
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9,weight_decay=1e-6, nesterov=True)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=milestones, gamma=0.1)
    
    # We often apply crossentropyloss for classification problem. Check it on pytorch if interested
    criterion = nn.CrossEntropyLoss()
    
    # Put model's parameters on your device
    model = model.to(device)
    
    ### TO DO ###
    # Complete the function train
    # Check tool.py
    train(model=model, train_loader=train_loader, val_loader=val_loader, 
          num_epoch=num_epoch, log_path=log_path, save_path=save_path,
          device=device, criterion=criterion, optimizer=optimizer, scheduler=scheduler,
          start_epoch=0, model_name=model_type)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    model_choice = ['myLeNet', 'myResNet18', 'myResNet34', 'ResNet18', 'ResNet34']
    parser.add_argument('--model', help='default ResNet18, ', type=str, default='ResNet18', choices=model_choice)
    args = parser.parse_args()

    model_name = args.model
    train_interface(model_name)




    