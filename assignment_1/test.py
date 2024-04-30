## Feel free to change the imports according to your implementation and needs
import argparse
import torch
import torchvision.transforms.v2 as v2
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

from dlvc.wandb_logger import WandBLogger
from dlvc.models.class_model import DeepClassifier
from dlvc.metrics import Accuracy
from dlvc.datasets.cifar10 import CIFAR10Dataset
from dlvc.datasets.dataset import Subset
from dlvc.models.resnet18 import ResNet18
from dlvc.models.cnn import CNN
from dlvc.models.vit import ViT


def test(args):
   
    
    transform = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Normalize(mean = [0.485, 0.456,0.406], std = [0.229, 0.224, 0.225])])
    
    test_data = CIFAR10Dataset('cifar-10-batches-py/', subset=Subset.TEST, transform = transform)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == 'resnet18':
        model = DeepClassifier(ResNet18()).to(device)
    elif args.model == 'cnn':
        model = DeepClassifier(CNN()).to(device)
    elif args.model == 'vit':
            vit = ViT(
                      image_size=32,
                      patch_size=4,
                      num_classes=10,
                      dim=512,
                      depth=6,
                      heads=8,
                      mlp_dim=512,
                      dropout=0.1,
                      emb_dropout=0.1)
            model = DeepClassifier(vit)
            model.to(device)
    else:
        raise ValueError("Model not supported")
        
    model.load(args.path_to_model)
    
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

    print(f"\nTest Loss: {test_loss:.2f}\n")
    print(test_metric)
    WandBLogger(run_name = model.net.__class__.__name__).log({"test loss": test_loss, "test accuracy": test_metric.accuracy()})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('-m', '--model', default='vit', type=str,
                      help='model to be tested')
    parser.add_argument('-p', '--path_to_model', default='saved_models/ViT__l_rate_0.0001_optim_Adam_scheduler_CosineAnnealingLR_num_epochs_200_batch_size_128.pth', type=str,
                      help='path to the trained model')
    
    args = parser.parse_args()
    
#     args.path_to_trained_model = "saved_models/model_vit_154.pth"

    test(args)
