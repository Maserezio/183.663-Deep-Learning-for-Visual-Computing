import torch
import torch.nn as nn
from pathlib import Path

class DeepClassifier(nn.Module):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)
    

    def save(self, save_dir: Path, suffix=None):
        '''
        Saves the model, adds suffix to filename if given
        '''

        ## TODO implement
        filename = "model"
        if suffix:
            filename += f"_{suffix}"
        filename += ".pth"
        save_path = save_dir / filename
        torch.save(self.net.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    def load(self, path):
        '''
        Loads model from path
        Does not work with transfer model
        '''
        ## TODO implement
        state_dict = torch.load(path, map_location=torch.device('cude' if torch.cuda.is_available() else 'cpu'))
        self.net.load_state_dict({k.replace('net.', 'model.'): v for k, v in state_dict.items()})
        print(f"Model loaded from {path}")
        self.net.eval()
