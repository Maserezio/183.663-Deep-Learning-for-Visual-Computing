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


class SegMetrics(PerformanceMeasure):
    '''
    Mean Intersection over Union.
    '''

    def __init__(self, classes):
        self.classes = classes

        self.reset()

    def reset(self) -> None:
        '''
        Resets the internal state.
        '''
        self.intersection = torch.zeros(len(self.classes))
        self.union = torch.zeros(len(self.classes))
        self.total_pixels = torch.zeros(len(self.classes))




    def update(self, prediction: torch.Tensor, 
               target: torch.Tensor) -> None:
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        prediction must have shape (b,c,h,w) where b=batchsize, c=num_classes, h=height, w=width.
        target must have shape (b,h,w) and values between 0 and c-1 (true class labels).
        Raises ValueError if the data shape or values are unsupported.
        Make sure to not include pixels of value 255 in the calculation since those are to be ignored. 
        '''

        for b in range(prediction.shape[0]):
            for c in range(len(self.classes)):
                class_mask = target[b] == c
                pred_mask = prediction[b].argmax(dim=0) == c
                intersection = (class_mask & pred_mask).sum().float()
                union = (class_mask | pred_mask).sum().float()
                total_pixels = class_mask.sum().float()

                self.intersection[c] += intersection
                self.union[c] += union
                self.total_pixels[c] += total_pixels
   

    def __str__(self):
        '''
        Return a string representation of the performance, mean IoU.
        e.g. "mIou: 0.54"
        '''
        return f"mIoU: {self.mIoU():.4f}"

          

    
    def mIoU(self) -> float:
        '''
        Compute and return the mean IoU as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        If the denominator for IoU calculation for one of the classes is 0,
        use 0 as IoU for this class.
        '''
        iou = (self.intersection / self.union).mean()
        return iou.item() if not torch.isnan(iou) else 0.0





