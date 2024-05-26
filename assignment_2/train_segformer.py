
import argparse
import os
import torch
import torchvision.transforms.v2 as v2
from pathlib import Path
import os

from dlvc.models.segformer import  SegFormer
from dlvc.models.segment_model import DeepSegmenter
from dlvc.dataset.cityscapes import CityscapesCustom
from dlvc.dataset.oxfordpets import OxfordPetsCustom
from dlvc.metrics import SegMetrics
from dlvc.trainer import ImgSemSegTrainer


def train(args):

    train_transform = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST),
                            v2.Normalize(mean = [0.485, 0.456,0.406], std = [0.229, 0.224, 0.225])])

    train_transform2 = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.long, scale=False),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST)])#,
    
    val_transform = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST),
                            v2.Normalize(mean = [0.485, 0.456,0.406], std = [0.229, 0.224, 0.225])])
    val_transform2 = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.long, scale=False),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST)])

    if args.dataset == "oxford":
        train_data = OxfordPetsCustom(root="path_to_dataset", 
                                split="trainval",
                                target_types='segmentation', 
                                transform=train_transform,
                                target_transform=train_transform2,
                                download=True)

        val_data = OxfordPetsCustom(root="path_to_dataset", 
                                split="test",
                                target_types='segmentation', 
                                transform=val_transform,
                                target_transform=val_transform2,
                                download=True)
    if args.dataset == "city":
        train_data = CityscapesCustom(root="path_to_dataset/cityscapes",
                                split="train",
                                mode="fine",
                                target_type='semantic', 
                                transform=train_transform,
                                target_transform=train_transform2)
        # val_data = CityscapesCustom(root="/data/databases/cityscapes",
        val_data = CityscapesCustom(root="path_to_dataset/cityscapes",
                                split="val",
                                mode="fine",
                                target_type='semantic', 
                                transform=val_transform,
                                target_transform=val_transform2)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_classes = len(train_data.classes_seg)
    model = DeepSegmenter(SegFormer(num_classes=num_classes))
    # If you are in the fine-tuning phase:
    if args.dataset == 'oxford':
        ##TODO update the encoder weights of the model with the loaded weights of the pretrained model
        # e.g. load pretrained weights with: state_dict = torch.load("path to model", map_location='cpu')
        state_dict = torch.load("saved_models/SegFormer_pretrained.pth", map_location='cpu')
        model.net.encoder.load_state_dict(state_dict, strict=False)

        if args.freeze_encoder:
            # Option B: Freeze the encoder
            for param in model.net.encoder.parameters():
                param.requires_grad = False

            optimizer = torch.optim.AdamW(model.net.decoder.parameters(), lr=0.001)
        else:
            # Option A: Fine-tune entire model
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    model.to(device)

    ignore_index = 255 if args.dataset == "city" else -1
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=ignore_index) # remember to ignore label value 255 when training with the Cityscapes datset
    
    train_metric = SegMetrics(classes=train_data.classes_seg)
    val_metric = SegMetrics(classes=val_data.classes_seg)
    val_frequency = 2 # for 

    model_save_dir = Path("saved_models")
    model_save_dir.mkdir(exist_ok=True)

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    
    trainer = ImgSemSegTrainer(model, 
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
                    batch_size=16,
                    val_frequency = val_frequency)
    trainer.train()
    # see Reference implementation of ImgSemSegTrainer
    # just comment if not used
    trainer.dispose() 

if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Training')
    args.add_argument('-d', '--gpu_id', default='0', type=str,
                      help='index of which GPU to use')
    args.add_argument('--num_epochs', type=int, default=40, help='Number of epochs to train')
    args.add_argument('--dataset', type=str, default='city', choices=['oxford', 'city'], help='Dataset to train on')
    args.add_argument('--freeze_encoder', type=bool, default=True, help='Whether to freeze the encoder of the model')

    if not isinstance(args, tuple):
        args = args.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    args.gpu_id = 0
    args.num_epochs = 40
    # args.dataset = "oxford"
    args.dataset = "city"

    train(args)