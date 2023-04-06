"""
Contains PyTorch code to instantiate a TensorBoard writer.
"""
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from pathlib import Path

def create_writer(
    experiment_name: str,
    model_name: str,
    extra: str=None
) -> torch.utils.tensorboard.writer.SummaryWriter():
    """
    Creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a specific log_dir.
    log_dir is a combination of runs/timestamp/experiment_name/model_name/extra.
    Timestamp is the current date in YYYY-MM-DD format.
    
    Args:
        experiment_name: Name of the experiment.
        model_name: Name of the model.
        extra: Anything extra to add to the directory.
        
    Returns:
        torch.utils.tensorboard.writer.SummaryWriter(): Instance of a writer saving to log_dir.
    """   
    timestamp = datetime.now().strftime("%Y-%m-%d")
    
    if extra:
        log_dir = Path(f"runs/{timestamp}/{experiment_name}/{model_name}/{extra}")
    else:
        log_dir = Path(f"runs/{timestamp}/{experiment_name}/{model_name}")
        
    print(f"[INFO] Created SummaryWriter(), saving to: {log_dir}")
    return SummaryWriter(log_dir=log_dir)
