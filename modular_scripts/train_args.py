"""
Trains a PyTorch image classification model.
"""
import os
import argparse
import torch
from pathlib import Path
from torchvision import transforms
from torchmetrics import Accuracy
import data_setup, engine, model_builder, utils

def main():
    parser = argparse.ArgumentParser(description="Training a model with modular scripts with hyperparameters")
    parser.add_argument("--train_dir", default="data/PizzaSteakSushi/train", type=str, help="The directory of the training data")
    parser.add_argument("--test_dir", default="data/PizzaSteakSushi/test", type=str, help="The directory of the testing data")
    parser.add_argument("--num_epochs", default=10, type=int, help="The number of epochs to train for")
    parser.add_argument("--batch_size", default=32, type=int, help="The number of samples per batch")
    parser.add_argument("--hidden_units", default=10, type=int, help="The number of hidden units in the model")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="The learning rate of the model")

    args=parser.parse_args()

    NUM_EPOCHS = args.num_epochs
    BATCH_SIZE = args.batch_size
    HIDDEN_UNITS = args.hidden_units
    LEARNING_RATE = args.learning_rate
    print(f"[INFO] Training for {NUM_EPOCHS} epochs, batch size: {BATCH_SIZE}, hidden units: {HIDDEN_UNITS}, learning rate: {LEARNING_RATE}")

    TRAIN_DIR = args.train_dir
    TEST_DIR = args.test_dir

    data_transform = transforms.Compose([
        transforms.Resize(size=(64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ])

    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=TRAIN_DIR,
        test_dir=TEST_DIR,
        transform=data_transform,
        batch_size=BATCH_SIZE
    )

    torch.manual_seed(42)
    model = model_builder.TinyVGG(
        in_shape=3,
        hidden=HIDDEN_UNITS,
        out_shape=len(class_names)
    )

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    metric_fn = Accuracy(task="multiclass", num_classes=len(class_names))

    engine.train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        metric_fn=metric_fn,
        epochs=NUM_EPOCHS
    )
    
    utils.save_model(
        model=model,
        target_dir="models",
        model_name="05_pytorch_going_modular_tinyvgg_args.pth"
    )
    
if __name__ == "__main__":
    main()
