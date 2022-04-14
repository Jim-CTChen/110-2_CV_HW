import torch
import sys
import argparse

# from torchsummary import summary

from tool import load_parameters
from myModels import myLeNet, myResNet18, myResNet34, ResNet34, ResNet18

def main():
    parser = argparse.ArgumentParser()
    model_choice = ['myLeNet', 'myResNet18', 'myResNet34', 'ResNet18', 'ResNet34']
    parser.add_argument('--model', help='default ResNet18, ', type=str, default='ResNet18', choices=model_choice)
    args = parser.parse_args()

    model_name = args.model

    num_out = 10

    # myLeNet
    if model_name == 'myLeNet':
        model = myLeNet(num_out=num_out)
        load_parameters(model, './save_dir/myLeNet/best_model.pt')

    # myResNet18
    elif model_name == 'myResNet18':
        model = myResNet18(num_out=num_out)
        load_parameters(model, './save_dir/myResNet18/best_model.pt')

    # myResNet34
    elif model_name == 'myResNet34':
        model = myResNet34(num_out=num_out)
        load_parameters(model, './save_dir/myResNet34/best_model.pt')

    # ResNet18
    elif model_name == 'ResNet18':
        model = ResNet18(num_out=num_out)
        load_parameters(model, './save_dir/ResNet18/best_model.pt')

    # ResNet34
    elif model_name == 'ResNet34':
        model = ResNet34(num_out=num_out)
        load_parameters(model, './save_dir/ResNet34/best_model.pt')


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    with open(model_name, 'w') as f:
        original_stdout = sys.stdout
        sys.stdout = f
        print(model)
        sys.stdout = original_stdout

    # summary(model, input_size=(3, 32, 32))

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))

if __name__ == '__main__':
	main()