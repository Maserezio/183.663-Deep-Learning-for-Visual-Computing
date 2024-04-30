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
        # torch.save(self.state_dict(), save_path)
        torch.save(self.net.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    def load(self, path):
        '''
        Loads model from path
        Does not work with transfer model
        '''
        ## TODO implement
        state_dict = torch.load(path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        if( self.net.__class__.__name__ == 'ResNet18' or self.net.__class__.__name__ == 'CNN'):
            self.net.load_state_dict({k.replace('net.', ''): v for k, v in state_dict.items()})
        else:
            adjusted_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace('net.', '')  # remove 'net.' prefix

                # Fixing specific layer references - adjust these based on your model's architecture
                new_key = new_key.replace('fn.0.', 'fn.net.0.')  # adjusting according to error messages
                new_key = new_key.replace('fn.3.', 'fn.net.3.')  # adjusting according to error messages

                adjusted_state_dict[new_key] = value

            try:
                self.net.load_state_dict(adjusted_state_dict)
            except RuntimeError as e:
                print(f"Failed to load state_dict: {str(e)}")
                print("Expected keys:")
                print(self.net.state_dict().keys())
                print("Loaded keys:")
                print(adjusted_state_dict.keys())

            # self.net.load_state_dict(adjusted_state_dict)
        print(f"Model loaded from {path}")

        self.net.eval()
