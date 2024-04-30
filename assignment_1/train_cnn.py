import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ExponentialLR, StepLR
from torch.optim import Adam, AdamW, SGD
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from dlvc.models.class_model import DeepClassifier
from dlvc.metrics import Accuracy
from dlvc.trainer import ImgClassificationTrainer
from dlvc.datasets.cifar10 import CIFAR10Dataset
from dlvc.datasets.dataset import Subset

from dlvc.models.cnn import CNN
from dlvc.randomaug import RandAugment

def train(args):
    # Define data transformations based on augmentation flag

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if args.aug:
        N = 2
        M = 14
        train_transform.transforms.insert(0, RandAugment(N, M))

    train_data = CIFAR10Dataset('cifar-10-batches-py', subset=Subset.TRAINING, transform=train_transform)
    val_data = CIFAR10Dataset('cifar-10-batches-py', subset=Subset.VALIDATION, transform=val_transform)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DeepClassifier(CNN()).to(device)

    if args.optimizer.lower() == 'adam':
        optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'adamw':
        optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'sgd':
        optimizer = SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    else:
        raise ValueError("Optimizer not supported")
    
    loss_fn = nn.CrossEntropyLoss()

    train_metric = Accuracy(classes=train_data.classes)
    val_metric = Accuracy(classes=val_data.classes)
    val_frequency = 5

    if args.scheduler.lower() == 'exp':
        lr_scheduler = ExponentialLR(optimizer, gamma=args.scheduler_gamma)
    elif args.scheduler.lower() == 'step':
        lr_scheduler = StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)
    else:
        raise ValueError("Scheduler not supported")
    
    model_save_dir = Path("saved_models")
    
    trainer = ImgClassificationTrainer(model,
                                       optimizer,
                                       loss_fn,
                                       lr_scheduler,
                                       train_metric,
                                       val_metric,
                                       train_data,
                                       val_data,
                                       device,
                                       args.num_epochs,
                                       model_save_dir,
                                       batch_size=args.batch_size,
                                       val_frequency=val_frequency)
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('-e', '--num_epochs', default=15, type=int, help='number of epochs for training')
    parser.add_argument('-lr', '--learning_rate', default=0.001, type=float, help='learning rate for optimizer')
    parser.add_argument('-opt', '--optimizer', default='adam', type=str, help='optimizer to use (adam, adamw, sgd)')
    parser.add_argument('-sch', '--scheduler', default='exp', type=str, help='learning rate scheduler to use (exp, step)')
    parser.add_argument('-sg', '--scheduler_gamma', default=0.9, type=float, help='gamma value for the scheduler')
    parser.add_argument('-ss', '--scheduler_step_size', default=1, type=int, help='step size for the scheduler')
    parser.add_argument('-bs', '--batch_size', default=128, type=int, help='batch size for training')
    parser.add_argument('-wd', '--weight_decay', default=1e-4, type=float, help='weight decay for regularization')
    parser.add_argument('--aug', action='store_true', help='enable data augmentation')

    args = parser.parse_args()

    train(args)
