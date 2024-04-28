import torch.nn as nn
from torchvision.models import resnet18

class ModifiedResNet18(nn.Module):
    def __init__(self, num_classes=10, pretrained=False):
        super(ModifiedResNet18, self).__init__()
        self.model = resnet18(pretrained=pretrained)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)


    def forward(self, x):
        return self.model(x)