import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
import torchvision.transforms.v2 as v2_transforms
from tqdm import tqdm
from UNet_model import UNet
from utils import (
    check_accuracy,
    get_loaders,
    load_checkpoint,
    save_checkpoint,
    save_predictions_as_images,
)

# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 10
NUM_WORKERS = 2
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
PIN_MEMORY = True
LOAD_MODEL = False
DATASET_DIR = "dataset"
ORIGINAL_DIR = "Original"
MASK_DIR = "Segmented"


def train_fn(loader, model, optimizer, loss_fn, scaler):
    batch_data = tqdm(loader)

    for _, (data, targets) in enumerate(batch_data):
        data = data.to(device=DEVICE)
        targets = targets.to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm
        batch_data.set_postfix(loss=loss.item())


class TrainTransform(torch.nn.Module):
    def forward(self, image, mask):
        # Resize
        # resize = v2_transforms.Resize(size=(IMAGE_HEIGHT, IMAGE_WIDTH))
        # image = resize(image)
        # mask = resize(mask)

        # Upsample
        upscale = nn.Upsample(scale_factor=5)
        image, mask = upscale(image.unsqueeze(0)), upscale(mask.unsqueeze(0))
        image, mask = image.squeeze(0), mask.squeeze(0)

        # Rotate with a random angle
        range_angle = 35
        angle = (-range_angle - range_angle) * torch.rand(1) + range_angle
        image = TF.rotate(image, angle.item())
        mask = TF.rotate(mask, angle.item())

        # Random horizontal flipping
        if torch.rand(1).item() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if torch.rand(1).item() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        return image, mask


class ValTransform(torch.nn.Module):
    def forward(self, image, mask):
        # Resize
        # resize = v2_transforms.Resize(size=(IMAGE_HEIGHT, IMAGE_WIDTH))
        # image = resize(image)
        # mask = resize(mask)

        # Upsample
        upscale = nn.Upsample(scale_factor=5)
        image, mask = upscale(image.unsqueeze(0)), upscale(mask.unsqueeze(0))
        image, mask = image.squeeze(0), mask.squeeze(0)

        return image, mask


train_transform = v2_transforms.Compose([TrainTransform()])
val_transform = v2_transforms.Compose([ValTransform()])


def main():
    model = UNet(in_channels=3, out_channels=5).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        DATASET_DIR,
        ORIGINAL_DIR,
        MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform,
        NUM_WORKERS,
        PIN_MEMORY,
        should_split=False,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
        check_accuracy(val_loader, model, device=DEVICE)
        save_predictions_as_images(
            val_loader, model, folder="predictions", device=DEVICE
        )
        return

    scaler = torch.cuda.amp.GradScaler()

    for _ in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save the model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

    # save some examples to a folder
    save_predictions_as_images(val_loader, model, folder="predictions", device=DEVICE)


if __name__ == "__main__":
    main()
