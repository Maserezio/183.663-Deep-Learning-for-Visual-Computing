import torch
from typing import Tuple
from abc import ABCMeta, abstractmethod
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

# for wandb users:
from dlvc.wandb_logger import WandBLogger


# from wandb_logger import WandBLogger

class BaseTrainer(metaclass=ABCMeta):
    '''
    Base class of all Trainers.
    '''

    @abstractmethod
    def train(self) -> None:
        '''
        Holds training logic.
        '''

        pass

    @abstractmethod
    def _val_epoch(self) -> Tuple[float, float, float]:
        '''
        Holds validation logic for one epoch.
        '''

        pass

    @abstractmethod
    def _train_epoch(self) -> Tuple[float, float, float]:
        '''
        Holds training logic for one epoch.
        '''

        pass


class ImgClassificationTrainer(BaseTrainer):
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
                 val_frequency: int = 5) -> None:
        '''
        Args and Kwargs:
            model (nn.Module): Deep Network to train
            optimizer (torch.optim): optimizer used to train the network
            loss_fn (torch.nn): loss function used to train the network
            lr_scheduler (torch.optim.lr_scheduler): learning rate scheduler used to train the network
            train_metric (dlvc.metrics.Accuracy): Accuracy class to get mAcc and mPCAcc of training set
            val_metric (dlvc.metrics.Accuracy): Accuracy class to get mAcc and mPCAcc of validation set
            train_data (dlvc.datasets.cifar10.CIFAR10Dataset): Train dataset
            val_data (dlvc.datasets.cifar10.CIFAR10Dataset): Validation dataset
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

        ## TODO implement
        self.last_val_loss = None
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.lr_scheduler = lr_scheduler
        self.train_metric = train_metric
        self.val_metric = val_metric
        self.train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.val_data = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        self.device = device
        self.num_epochs = num_epochs
        self.training_save_dir = Path(training_save_dir)
        self.batch_size = batch_size
        self.val_frequency = val_frequency
        self.best_accuracy = 0.0

        self.logger = WandBLogger(run_name=self.model.net.__class__.__name__,
                                  config={'optimizer': self.optimizer.__class__.__name__,
                                          'learning_rate': self.optimizer.param_groups[0]['lr'],
                                          'scheduler': self.lr_scheduler.__class__.__name__,
                                          'num_epochs': self.num_epochs,
                                          'batch_size': self.batch_size})
        # Check if training save directory exists
        training_save_dir.mkdir(parents=True, exist_ok=True)

    def _train_epoch(self, epoch_idx: int) -> Tuple[float, float, float]:
        """
        Training logic for one epoch. 
        Prints current metrics at end of epoch.
        Returns loss, mean accuracy and mean per class accuracy for this epoch.

        epoch_idx (int): Current epoch number
        """
        ## TODO implement
        self.model.train()
        total_loss = 0
        self.train_metric.reset()

        for data, target in self.train_data:
            # print(data.shape, target.shape)
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output, target)
            loss.backward()
            self.optimizer.step()
            # if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            #     self.lr_scheduler.step()  # for step, cos scheduler

            total_loss += loss.item()

            self.train_metric.update(output, target)

        average_loss = total_loss / len(self.train_data)
        accuracy = self.train_metric.accuracy()
        per_class_accuracy = self.train_metric.per_class_accuracy()
        mean_per_class_accuracy = per_class_accuracy[torch.isfinite(per_class_accuracy)].mean().item()

        current_lr = self.optimizer.param_groups[0]['lr']
        self.logger.log({
            'epoch': epoch_idx,
            'train_loss': average_loss,
            'train_accuracy': accuracy,
            'train_per_class_accuracy': mean_per_class_accuracy,
            'learning_rate': current_lr
        })
        print(f"______Epoch {epoch_idx}\n")
        print(f"Train Loss {average_loss:.2f}")
        print(self.train_metric)

        return average_loss, accuracy, mean_per_class_accuracy

    def _val_epoch(self, epoch_idx: int) -> Tuple[float, float, float]:
        """
        Validation logic for one epoch. 
        Prints current metrics at end of epoch.
        Returns loss, mean accuracy and mean per class accuracy for this epoch on the validation data set.

        epoch_idx (int): Current epoch number
        """
        ## TODO implement
        self.model.eval()
        total_loss = 0
        self.val_metric.reset()

        with torch.no_grad():
            for data, target in self.val_data:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.loss_fn(output, target)
                total_loss += loss.item()
                self.val_metric.update(output, target)

        average_loss = total_loss / len(self.val_data)
        accuracy = self.val_metric.accuracy()
        per_class_accuracy = self.val_metric.per_class_accuracy()
        mean_per_class_accuracy = per_class_accuracy[torch.isfinite(per_class_accuracy)].mean().item()

        print(f"______Epoch {epoch_idx}\n")
        print(f"Val Loss {average_loss:.2f}")
        print(self.val_metric)

        return average_loss, accuracy, mean_per_class_accuracy

    def train(self) -> None:
        """
        Full training logic that loops over num_epochs and
        uses the _train_epoch and _val_epoch methods.
        Save the model if mean per class accuracy on validation data set is higher
        than currently saved best mean per class accuracy. 
        Depending on the val_frequency parameter, validation is not performed every epoch.
        """
        ## TODO implement
        for epoch in range(self.num_epochs):
            train_loss, train_accuracy, train_per_class_accuracy = self._train_epoch(epoch)
            # if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            #     self.lr_scheduler.step()  # for step, cos scheduler
            # self.logger.log({'epoch': epoch, 'train_loss': train_loss, 'train_accuracy': train_accuracy})

            if (epoch + 1) % self.val_frequency == 0:
                val_loss, val_accuracy, val_per_class_accuracy = self._val_epoch(epoch)
                # self.logger.log({'epoch': epoch, 'val_loss': val_loss, 'val_accuracy': val_accuracy})
                self.logger.log({
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy,
                    'val_per_class_accuracy': val_per_class_accuracy
                })
                # self.lr_scheduler.step(self.optimizer)
                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(val_loss)
                    # self.lr_scheduler.step(self.optimizer)
                # else:
                #     self.lr_scheduler.step()
                    # Save the best model
                if val_per_class_accuracy > self.best_accuracy:
                    self.best_accuracy = val_per_class_accuracy
                    # lr = self.optimizer.param_groups[0]['lr']
                    lr = self.optimizer.param_groups[0]['lr']
                    optimizer_name = self.optimizer.__class__.__name__
                    scheduler_name = self.lr_scheduler.__class__.__name__
                    suffix = f"_l_rate_{lr}_optim_{optimizer_name}_scheduler_{scheduler_name}_num_epochs_{self.num_epochs}_batch_size_{self.batch_size}"
                    self.model.save(self.training_save_dir, suffix=suffix)

            else:
                # Only log and step the scheduler with the last known validation loss if not validating this epoch
                if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step()

