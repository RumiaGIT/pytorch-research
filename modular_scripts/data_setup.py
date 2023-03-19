"""
Contains functionality for creating PyTorch DataLoaders for custom image classification data.
"""
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int=NUM_WORKERS
):
    """
    Creates training and testing DataLoaders
    
    Takes in training and testing directory paths and turns their contents into PyTorch datasets, and then into PyTorch DataLoaders
    
    Parameters:
        train_dir: Path to the training directory
        test_dir: Path to the testing directory
        transform: A Torchvision transform to perform on the training and testing data
        batch_size: Sample size for the batches in the DataLoaders
        num_workers: Number of workers (CPU/GPU cores) per DataLoader
        
    Returns:
        A tuple of (train_dataloader, test_dataloader, class_names).
        Where class_names is a list of the target classes.
    """
    # Create datasets with datasets.ImageFolder()
    train_data = datasets.ImageFolder(root=train_dir, transform=transform)
    test_data = datasets.ImageFolder(root=test_dir, transform=transform)
    
    # Get class names
    class_names = train_data.classes
    
    # Transform datasets into DataLoaders
    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, num_workers=NUM_WORKERS, shuffle=True)
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, num_workers=NUM_WORKERS, shuffle=False)
    
    return train_dataloader, test_dataloader, class_names
