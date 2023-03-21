"""
Trains a PyTorch image classification model.
"""
import os
import torch
from pathlib import Path
from torchvision import transforms
from torchmetrics import Accuracy
import data_setup, engine, model_builder, utils

def main():
    EPOCHS = 5
    BATCH_SIZE = 32
    HIDDEN_UNITS = 10
    LEARNING_RATE = 0.001

    IMAGE_PATH = Path("../data/PizzaSteakSushi")
    TRAIN_DIR = Path(f"{IMAGE_PATH}/train")
    TEST_DIR = Path(f"{IMAGE_PATH}/test")

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
        epochs=EPOCHS
    )

    utils.save_model(
        model=model,
        target_dir="../models",
        model_name="05_pytorch_going_modular_tinyvgg.pth"
    )
    
if __name__ == "__main__":
    main()
