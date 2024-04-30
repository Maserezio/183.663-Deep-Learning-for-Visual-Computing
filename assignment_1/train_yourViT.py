## Feel free to change the imports according to your implementation and needs
import argparse
import os
import torch
import torchvision.transforms.v2 as v2

from pathlib import Path
from dlvc.models.class_model import DeepClassifier  # etc. change to your model
from dlvc.metrics import Accuracy
from dlvc.trainer import ImgClassificationTrainer
from dlvc.datasets.cifar10 import CIFAR10Dataset
from dlvc.datasets.dataset import Subset

from dlvc.randomaug import RandAugment
from dlvc.models.vit import ViT


def train(args):
    ### Implement this function so that it trains a specific model as described in the instruction.md file
    ## feel free to change the code snippets given here, they are just to give you an initial structure
    ## but do not have to be used if you want to do it differently
    ## For device handling you can take a look at pytorch documentation

    # train_transform = v2.Compose([v2.ToImage(),
    #                               v2.RandomHorizontalFlip(),
    #                               v2.ToDtype(torch.float32, scale=True),
    #                               v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    #
    # val_transform = v2.Compose([v2.ToImage(),
    #                             v2.ToDtype(torch.float32, scale=True),
    #                             v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    imsize = int(args.size)
    if args.net == "vit_timm":
        size = 384
    else:
        size = imsize

    train_transform = v2.Compose([v2.ToImage(),
                                  v2.RandomCrop(32, padding=4),
                                  v2.Resize(size),
                                  v2.RandomHorizontalFlip(),
                                  v2.ToDtype(torch.float32, scale=True),
                                  v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    val_transform = v2.Compose([v2.ToImage(),
                                v2.ToDtype(torch.float32, scale=True),
                                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # train_transform = v2.Compose([
    #     v2.RandomCrop(32, padding=4),
    #     v2.Resize(size),
    #     v2.RandomHorizontalFlip(),
    #     v2.ToDtype(torch.float32, scale=True),
    #     v2.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])
    #
    # val_transform = v2.Compose([
    #     v2.Resize(size),
    #     v2.ToDtype(torch.float32, scale=True),
    #     v2.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])

    if args.aug:
        N = 2
        M = 14
        train_transform.transforms.insert(0, RandAugment(N, M))

    train_data = CIFAR10Dataset('cifar-10-batches-py', subset=Subset.TRAINING, transform=train_transform)
    val_data = CIFAR10Dataset('cifar-10-batches-py', subset=Subset.VALIDATION, transform=val_transform)

    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu_id != '-1' else "cpu")

    # model = DeepClassifier(resnet18(pretrained=False))
    vit = ViT(
        image_size=32,
        patch_size=args.patch,
        num_classes=10,
        dim=int(args.dimhead),
        depth=6,
        heads=8,
        mlp_dim=512,
        dropout=0.1,
        emb_dropout=0.1
    )
    model = DeepClassifier(vit).to(device)
    # num_features = model.fc.in_features
    # model.fc = nn.Linear(num_features, model.num_classes())
    model.to(device)

    if args.opt == "adamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, amsgrad=True)
    elif args.opt == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.opt == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    loss_fn = torch.nn.CrossEntropyLoss()

    train_metric = Accuracy(classes=train_data.classes)
    val_metric = Accuracy(classes=val_data.classes)
    val_frequency = 5

    if args.scheduler == "step":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif args.scheduler == "exp":
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    elif args.scheduler == "cos":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs)

    model_save_dir = Path("saved_models")
    model_save_dir.mkdir(exist_ok=True)

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
                                       batch_size=128,  # feel free to change
                                       val_frequency=val_frequency)
    trainer.train()


if __name__ == "__main__":
    ## Feel free to change this part - you do not have to use this argparse and gpu handling
    args = argparse.ArgumentParser(description='Training')
    args.add_argument('-d', '--gpu_id', default='0', type=str,
                      help='index of which GPU to use')
    args.add_argument('--num_epochs', default=200, type=int, help="Number of epochs")
    args.add_argument('--opt', default='adam', type=str, help="Optimizer")
    args.add_argument('--scheduler', default='cos', type=str, help="Scheduler")
    args.add_argument('--lr', default=1e-4, type=float, help="Learning rate")
    args.add_argument('--aug', default=False, type=bool, help="Augmentation")
    args.add_argument('--patch', default='4', type=int, help="patch for ViT")
    args.add_argument('--dimhead', default="512", type=int)
    args.add_argument('--size', default='32', type=int)
    args.add_argument('--net', default='', type=str)

    if not isinstance(args, tuple):
        args = args.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    args.gpu_id = 0

    train(args)
