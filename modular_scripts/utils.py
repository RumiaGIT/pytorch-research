"""
Contains various utility functions for PyTorch model training and saving.
"""
import torch
from pathlib import Path

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str) -> None:
    
    """
    Saves a PyTorch model to a target directory.
    
    Parameters:
        model: A PyTorch model to save.
        target_dir: The directory to save the model to.
        model_name: The filename to give to the model, should use ".pth" or ".pt" as the file extension.
    """
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    
    assert model_name.endswith(".pth") or model_name.endswith(".pt")
    model_save_path = Path(f"{target_dir_path}/{model_name}")
    
    print(f"Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)
