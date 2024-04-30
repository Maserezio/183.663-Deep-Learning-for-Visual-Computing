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
        filename = self.net.__class__.__name__
        if suffix:
            filename += f"_{suffix}"
        filename += ".pth"
        save_path = f"{save_dir}/{filename}"
        torch.save(self.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    def load(self, path):
        '''
        Loads model from path
        Does not work with transfer model
        '''
        ## TODO implement
        state_dict = torch.load(path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.net.load_state_dict({k.replace('net.', ''): v for k, v in state_dict.items()})
        # self.net.load_state_dict(state_dict)
        print(f"Model loaded from {path}")

        self.net.eval()
