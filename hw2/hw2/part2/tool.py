
import torch
import torch.nn as nn

import numpy as np 
import time
from tqdm import tqdm
import os
import random
import matplotlib.pyplot as plt


def fixed_seed(myseed):
    np.random.seed(myseed)
    random.seed(myseed)
    torch.manual_seed(myseed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)
        torch.cuda.manual_seed(myseed)
        
        
def save_model(model, path):
    print(f'Saving model to {path}...')
    torch.save(model.state_dict(), path)
    print("End of saving !!!")


def load_parameters(model, path):
    print(f'Loading model parameters from {path}...')
    # param = torch.load(path, map_location={'cuda:0': 'cuda:1'})
    param = torch.load(path, map_location='cuda:0')
    # param = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(param)
    print("End of loading !!!")



## TO DO ##
def plot_learning_curve(x, y, title, x_label, y_label, path):
    """_summary_
    The function is mainly to show and save the learning curves. 
    input: 
        x: data of x axis 
        y: list of (data of y axis)
        title: list of title
        x_label: x label
        y_label: list of y label
    output: None 
    """
    for i in range(len(y)):
        plt.figure(i)
        plt.plot(x, y[i])
        plt.title(title[i])
        plt.xlabel(x_label)
        plt.ylabel(y_label[i])
        plt.savefig(f'{path}/{title[i]}.png')

    plt.show()
    

def train(model, train_loader, val_loader, num_epoch, log_path, save_path, device, criterion, scheduler, optimizer, start_epoch, model_name):
    start_train = time.time()

    overall_loss = np.zeros(num_epoch ,dtype=np.float32)
    overall_acc = np.zeros(num_epoch ,dtype = np.float32)
    overall_val_loss = np.zeros(num_epoch ,dtype=np.float32)
    overall_val_acc = np.zeros(num_epoch ,dtype = np.float32)

    best_acc = 0

    for i in range(num_epoch):
        current_epoch = start_epoch+i+1
        print(f'epoch = {current_epoch}')
        # epcoch setting
        start_time = time.time()
        train_loss = 0.0 
        corr_num = 0

        # training part
        # start training
        model.train()
        for batch_idx, ( data, label,) in enumerate(tqdm(train_loader)):
            # put the data and label on the device
            # note size of data (B,C,H,W) --> B is the batch size
            data = data.to(device)
            label = label.to(device)

            # pass forward function define in the model and get output 
            output = model(data) 

            # calculate the loss between output and ground truth
            loss = criterion(output, label)
            
            # discard the gradient left from former iteration 
            optimizer.zero_grad()

            # calcualte the gradient from the loss function 
            loss.backward()
            
            # if the gradient is too large, we dont adopt it
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm= 5.)
            
            # Update the parameters according to the gradient we calculated
            optimizer.step()

            train_loss += loss.item()

            # predict the label from the last layers' output. Choose index with the biggest probability 
            pred = output.argmax(dim=1)
            
            # correct if label == predict_label
            corr_num += (pred.eq(label.view_as(pred)).sum().item())

        # scheduler += 1 for adjusting learning rate later
        scheduler.step()
        
        # averaging training_loss and calculate accuracy
        train_loss = train_loss / len(train_loader.dataset) 
        train_acc = corr_num / len(train_loader.dataset)
                
        # record the training loss/acc
        overall_loss[i], overall_acc[i] = train_loss, train_acc
        
        ## TO DO ##
        # validation part 
        with torch.no_grad():
            model.eval()
            val_loss = 0
            corr_num = 0
            val_acc = 0
            
            ## TO DO ## 
            # Finish forward part in validation. You can refer to the training part 
            # Note : You don't have to update parameters this part. Just Calculate/record the accuracy and loss. 
            for batch_idx, ( data, label,) in enumerate(tqdm(val_loader)):
                # put the data and label on the device
                # note size of data (B,C,H,W) --> B is the batch size
                data = data.to(device)
                label = label.to(device)

                # pass forward function define in the model and get output 
                output = model(data) 
                # print(f'output shape: {output.shape}')
                # print(f'output: {output}')
                
                # print(f'label shape: {label.shape}')
                # print(f'label: {label}')

                # calculate the loss between output and ground truth
                loss = criterion(output, label)
                # print(f'loss: {loss}')
                # print(f'loss type: {type(loss)}')
                # print(f'loss.item(): {loss.item()}')
                # print(f'loss.item() type: {type(loss.item())}')
                # quit()

                val_loss += loss.item()

                # predict the label from the last layers' output. Choose index with the biggest probability 
                pred = output.argmax(dim=1)
                
                # correct if label == predict_label
                corr_num += (pred.eq(label.view_as(pred)).sum().item())

            val_loss = val_loss / len(val_loader.dataset) # avg loss
            val_acc  = corr_num / len(val_loader.dataset)

            overall_val_loss[i], overall_val_acc[i] = val_loss, val_acc

        
        # Display the results
        end_time = time.time()
        elp_time = end_time - start_time
        min = elp_time // 60 
        sec = elp_time % 60
        print('*'*10)
        #print(f'epoch = {i}')
        print('time = {:.4f} MIN {:.4f} SEC, total time = {:.4f} Min {:.4f} SEC '.format(elp_time // 60, elp_time % 60, (end_time-start_train) // 60, (end_time-start_train) % 60))
        print(f'training loss : {train_loss:.4f} ', f' train acc = {train_acc:.4f}' )
        print(f'val loss : {val_loss:.4f} ', f' val acc = {val_acc:.4f}' )
        print('========================\n')

        with open(log_path, 'a') as f :
            f.write(f'epoch = {current_epoch}\n', )
            f.write('time = {:.4f} MIN {:.4f} SEC, total time = {:.4f} Min {:.4f} SEC\n'.format(elp_time // 60, elp_time % 60, (end_time-start_train) // 60, (end_time-start_train) % 60))
            f.write(f'training loss : {train_loss}  train acc = {train_acc}\n' )
            f.write(f'val loss : {val_loss}  val acc = {val_acc}\n' )
            f.write('============================\n')

        # save model for every 5 epoch 
        if (current_epoch) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f'epoch_{current_epoch}.pt'))
        
        # save the best model if it gain performance on validation set
        if  val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pt'))


    x = range(0,num_epoch)
    overall_acc = overall_acc.tolist()
    overall_loss = overall_loss.tolist()
    overall_val_acc = overall_val_acc.tolist()
    overall_val_loss = overall_val_loss.tolist()

    with open(f'./acc_log/{model_name}/train_acc.log', 'w') as f:
      for acc in overall_acc:
        f.write(str(acc)+'\n')

    with open(f'./acc_log/{model_name}/train_loss.log', 'w') as f:
      for loss in overall_loss:
        f.write(str(loss)+'\n')

    with open(f'./acc_log/{model_name}/valid_acc.log', 'w') as f:
      for acc in overall_val_acc:
        f.write(str(acc)+'\n')

    with open(f'./acc_log/{model_name}/valid_loss.log', 'w') as f:
      for loss in overall_val_loss:
        f.write(str(loss)+'\n')

    # Plot Learning Curve
    ## TO DO ##
    # Consider the function plot_learning_curve(x, y) above
    epoch_list = [e for e in range(num_epoch)]
    y = [overall_acc, overall_loss, overall_val_acc, overall_val_loss]
    titles = ['overall train accuracy', 'overall train loss', 'overall validation accuracy', 'overall validation loss']
    y_labels = ['train_acc', 'train_loss', 'val_acc', 'val_loss']

    plot_learning_curve(epoch_list, y, titles, 'epoch', y_labels, f'./acc_log/{model_name}')
    

