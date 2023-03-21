"""
Contains functions for training and testing a PyTorch model.
"""
import torch
from typing import Dict, List, Tuple
import torchmetrics

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               metric_fn: torchmetrics.classification) -> Tuple[float, float]:
    """
    Trains a PyTorch model for a single epoch.
    
    Parameters:
        model: A PyTorch model to be trained.
        dataloader: A DataLoader instance for the model to be trained on.
        loss_fn: A PyTorch loss function to minimize.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        metric_fn: A PyTorch metric to track how well the model performs.
        
    Returns:
        A tuple of training loss and training accuracy metrics.
        In the form (train_loss, train_accuracy).
    """
    train_loss, train_acc = 0, 0
    
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        y_logits = model(X)
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
        loss = loss_fn(y_logits, y)
        train_loss += loss
        train_acc += metric_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    
    return train_loss, train_acc

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              metric_fn: torchmetrics.classification) -> Tuple[float, float]:
    """
    Tests a PyTorch model for a single epoch.
    
    Parameters:
        model: A PyTorch model to be trained.
        dataloader: A DataLoader instance for the model to be tested on.
        loss_fn: A PyTorch loss function to minimize.
        metric_fn: A PyTorch metric to track how well the model performs.
        
    Returns:
        A tuple of testing loss and testing accuracy metrics.
        In the form (test_loss, test_accuracy).
    """
    test_loss, test_acc = 0, 0
    
    model.eval()
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            test_logits = model(X)
            test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
            test_loss += loss_fn(test_logits, y)
            test_acc += metric_fn(test_pred, y)
            
        test_loss /= len(dataloader)
        test_acc /= len(dataloader)
    
    return test_loss, test_acc

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          metric_fn: torchmetrics.classification,
          epochs: int) -> Dict[str, List]:
    """
    Trains and tests a PyTorch model.
    
    Passes a target PyTorch models through train_step() and test_step() functions.
    For a number of epochs, training and testing steps for model the model are performed.
    Calculates, prints and stores evaluation metrics throughout.
    
    Parameters:
        model: A PyTorch model to be trained.
        train_dataloader: A DataLoader instance for the model to be trained on.
        test_dataloader: A DataLoader instance for the model to be tested on.
        loss_fn: A PyTorch loss function to minimize.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        metric_fn: A PyTorch metric to track how well the model performs.
        epochs: An integer indicating how many epochs to train for.
        
    Returns:
        A dictionary of training and testing loss as well as training and testing accuracy metrics. 
        Each metric has a value in a list for each epoch.
    """
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    
    for epoch in range(epochs):
        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, metric_fn)
        test_loss, test_acc = test_step(model, test_dataloader, loss_fn, metric_fn)
        
        print(f"Epoch: {epoch}\n--------")
        print(f"Train Loss: {train_loss}, Train Acc: {train_acc} | Test Loss: {test_loss}, Test Acc: {test_acc}")
        
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        
    #return results
