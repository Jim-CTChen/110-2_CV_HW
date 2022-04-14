

from torchvision.transforms import transforms
from torch.utils.data import DataLoader

import torch 
import json
import argparse
from tqdm import tqdm

from tool import load_parameters
from myModels import myResNet18, myLeNet, myResNet34, ResNet34, ResNet18
from myDatasets import cifar10_dataset


# The function help you to calculate accuracy easily
# Normally you won't get annotation of test label. But for easily testing, we provide you this.
def test_result(test_loader, model, device):
    pred = []
    cnt = 0
    model.eval()
    with torch.no_grad():
        for img, label in tqdm(test_loader):
            img = img.to(device)
            label = label.to(device)
            pred = model(img)
            pred = torch.argmax(pred, axis=1)
            cnt += (pred.eq(label.view_as(pred)).sum().item())
    
    acc = cnt / len(test_loader.dataset)
    return acc

def main():

    parser = argparse.ArgumentParser()
    model_choice = ['myLeNet', 'myResNet18', 'myResNet34', 'ResNet18', 'ResNet34']
    parser.add_argument('--model', help='default ResNet18, ', type=str, default='ResNet18', choices=model_choice)
    # parser.add_argument('--path', help='model_path', type=str, default=f'./save_dir/{model_name}/best_model.pt')
    parser.add_argument('--test_anno', help='annotaion for test image', type=str, default= './p2_data/annotations/public_test_annos.json')
    args = parser.parse_args()

    model_name = args.model
    # path = args.path
    path = f'./save_dir/{model_name}/best_model.pt'
    test_anno = args.test_anno
    num_out = 10
    
    # change your model here 

    ## Modify here if you want to change your model ##
    if model_name == 'myLeNet':
        model = myLeNet(num_out=num_out)

    # myResNet18
    elif model_name == 'myResNet18':
        model = myResNet18(num_out=num_out)

    # myResnet34
    elif model_name == 'myResNet34': 
        model = myResNet34(num_out=num_out)

    # pytorch vision
    elif model_name == 'ResNet18':
        model = ResNet18(num_out=num_out)

    elif model_name == 'ResNet34':
        model = ResNet34(num_out=num_out)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    
    # Simply load parameters
    load_parameters(model=model, path=path)
    model.to(device)


    with open(test_anno, 'r') as f :
        data = json.load(f)    
    
    imgs, categories = data['images'], data['categories']
    
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(means, stds),
            ])
    
    test_set = cifar10_dataset(images=imgs, labels= categories, transform=test_transform, prefix = './p2_data/public_test/')
    #test_set = cifar10_dataset(images=imgs, labels= categories, transform=test_transform, prefix = './p2_data/private_test/')
    test_loader = DataLoader(test_set, batch_size=4, shuffle=False)
    acc = test_result(test_loader=test_loader, model=model, device=device)
    print("accuracy : ", acc)
    
if __name__ == '__main__':
    main()