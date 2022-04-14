import matplotlib.pyplot as plt
import argparse

def plot_learning_curve(x, y, title, x_label, y_label):
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

  plt.show()

if __name__ == '__main__':

  parser = argparse.ArgumentParser()

  model_choice = ['myLeNet', 'myResNet18', 'myResNet34', 'ResNet18', 'ResNet34']
  parser.add_argument('--model', help='default ResNet18, ', type=str, default='ResNet18', choices=model_choice)
  args = parser.parse_args()

  model_name = args.model

  train_loss = []
  train_acc  = []
  val_loss = []
  val_acc  = []

  with open(f'acc_log/{model_name}/train_acc.log', 'r') as f:
    lines = f.readlines()
    train_acc = [float(l[:-1]) for l in lines]

  with open(f'acc_log/{model_name}/train_loss.log', 'r') as f:
    lines = f.readlines()
    train_loss = [float(l[:-1]) for l in lines]

  with open(f'acc_log/{model_name}/valid_acc.log', 'r') as f:
    lines = f.readlines()
    val_acc = [float(l[:-1]) for l in lines]

  with open(f'acc_log/{model_name}/valid_loss.log', 'r') as f:
    lines = f.readlines()
    val_loss = [float(l[:-1]) for l in lines]
  
  epoch_list = [i for i in range(1, 41)]
  y = [train_acc, train_loss, val_acc, val_loss]
  titles = ['overall train accuracy', 'overall train loss', 'overall validation accuracy', 'overall validation loss']
  y_labels = ['train_acc', 'train_loss', 'val_acc', 'val_loss']

  plot_learning_curve(epoch_list, y, titles, 'epoch', y_labels)