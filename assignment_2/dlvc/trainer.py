import collections
import torch
from typing import  Tuple
from abc import ABCMeta, abstractmethod
from pathlib import Path
from tqdm import tqdm

from dlvc.wandb_logger import WandBLogger
from dlvc.dataset.oxfordpets import OxfordPetsCustom

class BaseTrainer(metaclass=ABCMeta):
    '''
    Base class of all Trainers.
    '''

    @abstractmethod
    def train(self) -> None:
        '''
        Returns the number of samples in the dataset.
        '''

        pass

    @abstractmethod
    def _val_epoch(self) -> Tuple[float, float]:
        '''
        Returns the number of samples in the dataset.
        '''

        pass

    @abstractmethod
    def _train_epoch(self) -> Tuple[float, float]:
        '''
        Returns the number of samples in the dataset.
        '''

        pass

class ImgSemSegTrainer(BaseTrainer):
    """
    Class that stores the logic for training a model for image classification.
    """
    def __init__(self, 
                 model, 
                 optimizer,
                 loss_fn,
                 lr_scheduler,
                 train_metric,
                 val_metric,
                 train_data,
                 val_data,
                 device,
                 num_epochs: int, 
                 training_save_dir: Path,
                 batch_size: int = 4,
                 val_frequency: int = 5):
        '''
        Args and Kwargs:
            model (nn.Module): Deep Network to train
            optimizer (torch.optim): optimizer used to train the network
            loss_fn (torch.nn): loss function used to train the network
            lr_scheduler (torch.optim.lr_scheduler): learning rate scheduler used to train the network
            train_metric (dlvc.metrics.SegMetrics): SegMetrics class to get mIoU of training set
            val_metric (dlvc.metrics.SegMetrics): SegMetrics class to get mIoU of validation set
            train_data (dlvc.datasets...): Train dataset
            val_data (dlvc.datasets...): Validation dataset
            device (torch.device): cuda or cpu - device used to train the network
            num_epochs (int): number of epochs to train the network
            training_save_dir (Path): the path to the folder where the best model is stored
            batch_size (int): number of samples in one batch 
            val_frequency (int): how often validation is conducted during training (if it is 5 then every 5th 
                                epoch we evaluate model on validation set)

        What does it do:
            - Stores given variables as instance variables for use in other class methods e.g. self.model = model.
            - Creates data loaders for the train and validation datasets
            - Optionally use weights & biases for tracking metrics and loss: initializer W&B logger

        '''
        ##TODO implement
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.lr_scheduler = lr_scheduler
        self.train_metric = train_metric
        self.train_data = train_data
        self.val_data = val_data
        self.device = device
        self.num_epochs = num_epochs
        self.training_save_dir = training_save_dir
        self.batch_size = batch_size
        self.val_frequency = val_frequency
        self.val_metric = val_metric
        self.num_train_data = len(train_data)
        self.num_val_data = len(val_data)
        self.subtract_one = isinstance(train_data, OxfordPetsCustom)

        self.train_data_loader = torch.utils.data.DataLoader(train_data,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=2)
    
        self.val_data_loader = torch.utils.data.DataLoader(val_data,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=1)
        
        self.wandb_logger = WandBLogger(enabled=True, model=model, run_name=model.net._get_name(), config={'batch_size': batch_size,
                                                                                                           'num_epochs': num_epochs,
                                                                                                           'optimizer': optimizer,
                                                                                                           'loss_fn': loss_fn,
                                                                                                           'lr_scheduler': lr_scheduler.__class__.__name__})


    def _train_epoch(self, epoch_idx: int) -> Tuple[float, float]:
        """
        Training logic for one epoch. 
        Prints current metrics at end of epoch.
        Returns loss, mean IoU for this epoch.

        epoch_idx (int): Current epoch number
        """
        self.model.train()
        epoch_loss = 0.
        self.train_metric.reset()
        
        # train epoch
        for i, batch in tqdm(enumerate(self.train_data_loader), desc="train", total=len(self.train_data_loader)):

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = batch
            labels = labels.squeeze(1) - int(self.subtract_one)

            batch_size = inputs.shape[0] # b x ..?

            # Make predictions for this batch
            outputs = self.model(inputs.to(self.device))
            if isinstance(outputs, collections.OrderedDict):
                outputs = outputs['out']


            # Compute the loss and its gradients
            loss = self.loss_fn(outputs, labels.to(self.device))
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # Gather metrics
            epoch_loss += (loss.item() * batch_size)
            self.train_metric.update(outputs.detach().cpu(), labels.detach().cpu())
        
        self.lr_scheduler.step()
        epoch_loss /= self.num_train_data
        epoch_mIoU = self.train_metric.mIoU()
        
        return round(epoch_loss, 4), round(epoch_mIoU, 4)

    def _val_epoch(self, epoch_idx:int) -> Tuple[float, float]:
        """
        Validation logic for one epoch. 
        Prints current metrics at end of epoch.
        Returns loss, mean IoU for this epoch on the validation data set.

        epoch_idx (int): Current epoch number
        """
        self.val_metric.reset()
        epoch_loss = 0.
        for batch_idx, batch in tqdm(enumerate(self.val_data_loader), desc="eval", total=len(self.val_data_loader)):
            self.model.eval()
            with torch.no_grad():
                # get the inputs; data is a tuple of [inputs, labels]
                inputs, labels = batch
                labels = labels.squeeze(1) - int(self.subtract_one)
                batch_size = inputs.shape[0] 

                # Make predictions for this batch
                outputs = self.model(inputs.to(self.device))
                if isinstance(outputs, collections.OrderedDict):
                    outputs = outputs['out']

                # Compute the loss and its gradients
                loss = self.loss_fn(outputs, labels.to(self.device))
                # Gather metrics
                epoch_loss += (loss.item() * batch_size)
                self.val_metric.update(outputs.cpu(), labels.cpu())

        epoch_loss /= self.num_val_data
        epoch_mIoU = self.val_metric.mIoU()
        
        return round(epoch_loss, 4), round(epoch_mIoU, 4)

    def train(self) -> None:
        """
        Full training logic that loops over num_epochs and
        uses the _train_epoch and _val_epoch methods.
        Save the model if mean IoU on validation data set is higher
        than currently saved best mean IoU or if it is end of training. 
        Depending on the val_frequency parameter, validation is not performed every epoch.
        """
        best_mIoU = 0.
        for epoch in range(self.num_epochs):
            train_loss, train_mIoU = self._train_epoch(epoch)
            print(f"Epoch {epoch}, Train Loss: {train_loss}, Train mIoU: {train_mIoU}")
            self.lr_scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            if epoch % self.val_frequency == 0:
                val_loss, val_mIoU = self._val_epoch(epoch)
                print(f"Epoch {epoch}, Val Loss: {val_loss}, Val mIoU: {val_mIoU}")
                if val_mIoU > best_mIoU:
                    best_mIoU = val_mIoU
                    self.model.save(self.training_save_dir, suffix='best')
                    print(f"Best model saved with mIoU: {best_mIoU}")

            print(f"Epoch {epoch}, Train Loss: {train_loss}, Train mIoU: {train_mIoU}, Val Loss: {val_loss}, Val mIoU: {val_mIoU}")

            self.wandb_logger.log({
                    'train_loss': train_loss,
                    'train_mIoU': train_mIoU,
                    'val_loss': val_loss,
                    'val_mIoU': val_mIoU,
                    'lr': current_lr,
                    'epoch': epoch,
            })

    def dispose(self) -> None:
        self.wandb_logger.finish()