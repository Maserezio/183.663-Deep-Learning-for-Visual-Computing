## Feel free to change the imports according to your implementation and needs
import argparse
import os
import torch
import torchvision.transforms.v2 as v2
import os
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

from torchvision.models import resnet18 # change to the model you want to test
from dlvc.models.class_model import DeepClassifier
from dlvc.metrics import Accuracy
from dlvc.datasets.cifar10 import CIFAR10Dataset
from dlvc.datasets.dataset import Subset
from dlvc.models.resnet18 import ModifiedResNet18


def modified_resnet18(num_classes=10, pretrained=False):
    model = resnet18(pretrained=pretrained)
    # Change the final layer to match the number of CIFAR-10 classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model

def test(args):
   
    
    transform = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Normalize(mean = [0.485, 0.456,0.406], std = [0.229, 0.224, 0.225])])
    
    test_data = CIFAR10Dataset('cifar-10-batches-py/', subset=Subset.TEST, transform = transform)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)
        
    # device = torch.device("cuda" if torch.cuda.is_available() and args.gpu_id != '-1' else "cpu")
    device = torch.device("cpu")

    # resnet18 = ModifiedResNet18()
    model = DeepClassifier(ModifiedResNet18())
    model.to(device)
    model.load(args.path_to_trained_model)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    
    test_metric = Accuracy(classes=test_data.classes)
    
    total_loss = 0.0

    with torch.no_grad():
        for inputs, targets in test_data_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            test_metric.update(outputs, targets)

    test_loss = total_loss / len(test_data)

    print(f"\nTest Loss: {test_loss:.4f}\n")
    print(test_metric)
    for i in range(len(test_metric.classes)):
        print(f'Accuracy for class: {test_metric.classes[i]} is {test_metric.per_class_accuracy()[i]:.2f}')

if __name__ == "__main__":
    ## Feel free to change this part - you do not have to use this argparse and gpu handling
    args = argparse.ArgumentParser(description='Training')
    args.add_argument('-d', '--gpu_id', default='5', type=str,
                      help='index of which GPU to use')
    
    if not isinstance(args, tuple):
        args = args.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    args.gpu_id = 0 
    args.path_to_trained_model = "saved_models/best_model.pth"

    test(args)
