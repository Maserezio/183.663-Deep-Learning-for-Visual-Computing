from abc import ABCMeta, abstractmethod
import torch

class PerformanceMeasure(metaclass=ABCMeta):
    '''
    A performance measure.
    '''

    @abstractmethod
    def reset(self):
        '''
        Resets internal state.
        '''

        pass

    @abstractmethod
    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        Raises ValueError if the data shape or values are unsupported.
        '''

        pass

    @abstractmethod
    def __str__(self) -> str:
        '''
        Return a string representation of the performance.
        '''

        pass



class Accuracy(PerformanceMeasure):
    '''
    Average classification accuracy.
    '''

    def __init__(self, classes) -> None:
        self.classes = classes

        self.reset()

    def reset(self) -> None:
        '''
        Resets the internal state.
        '''
        ## TODO implement
        self.total_correct = 0
        self.total_samples = 0
        self.class_correct = torch.zeros(len(self.classes))
        self.class_total = torch.zeros(len(self.classes))

    def update(self, prediction: torch.Tensor, 
               target: torch.Tensor) -> None:
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        prediction must have shape (s,c) with each row being a class-score vector.
        target must have shape (s,) and values between 0 and c-1 (true class labels).
        Raises ValueError if the data shape or values are unsupported.
        '''

        ## TODO implement
        if prediction.dim() != 2 or target.dim() != 1:
            raise ValueError("Unsupported tensor dimensions")
        if prediction.shape[0] != target.shape[0]:
            raise ValueError("Number of predictions and targets must match")
        if prediction.shape[1] != len(self.classes):
            # print(prediction.shape[0], prediction.shape[1], len(self.classes))
            raise ValueError("Number of predictions per sample must match number of classes")
        if target.max() >= len(self.classes):
            raise ValueError("Target contains out of range class values")

        predicted_classes = torch.argmax(prediction, dim=1)
        correct = predicted_classes == target

        # totals
        self.total_correct += correct.sum().item()
        self.total_samples += target.size(0)

        # per class
        for i in range(len(self.classes)):
            self.class_correct[i] += correct[target == i].sum().item()
            self.class_total[i] += (target == i).sum().item()

    def __str__(self):
        '''
        Return a string representation of the performance, accuracy and per class accuracy.
        '''

        ## TODO implement
        acc = self.accuracy()
        per_class_acc = self.per_class_accuracy()
        return f"Overall Accuracy: {acc:.4f}, Per class accuracy: {per_class_acc:.4f}"


    def accuracy(self) -> float:
        '''
        Compute and return the accuracy as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        '''

        ## TODO implement
        if self.total_samples == 0:
            return 0.0
        return self.total_correct / self.total_samples
    
    def per_class_accuracy(self) -> float:
        '''
        Compute and return the per class accuracy as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        '''
        ## TODO implement
        if torch.sum(self.class_total) == 0:
            return 0.0
        per_class_acc = self.class_correct / self.class_total
        return per_class_acc[torch.isfinite(per_class_acc)].mean().item()
       