import os

import torch
import torchvision
from dataset_creator import MedicalImageDataset, get_label, split_data
from torch.utils.data import ConcatDataset, DataLoader


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):  # pragma: no cover
    print("=> Saving Checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):  # pragma: no cover
    print("=> Loading Checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def extract_mask_label(multi_channel_mask):
    output_mask = multi_channel_mask[0]
    output_mask = output_mask.unsqueeze(0)

    means = []
    for dim in range(1, multi_channel_mask.size(0)):
        mean = multi_channel_mask[dim].mean()
        means.append(mean)

    label_id = torch.argmax(torch.tensor(means))
    return output_mask, label_id


def extract_mask_label_from_batch(multi_channel_mask_batch):
    masks = []
    labels = []
    for mask_idx in range(multi_channel_mask_batch.size(0)):
        mask, label = extract_mask_label(multi_channel_mask_batch[mask_idx])
        masks.append(mask.detach().cpu())
        labels.append(label.detach().cpu())

    return masks, labels


def get_loaders(
    dataset_dir,
    original_dir,
    mask_dir,
    batch_size,
    train_transform,
    val_transform,
    num_workders=4,
    pin_memory=True,
    should_split=False,
):  # pragma: no cover
    if should_split:
        # first we split data into train and test dir
        train_dir, test_dir = split_data(dataset_dir, original_dir, mask_dir)
    else:
        train_dir, test_dir = os.path.join(dataset_dir, "Train_Data"), os.path.join(
            dataset_dir, "Test_Data"
        )

    # now we create dataset from the train_dir and test_dir
    # there will be multiple datasets, so we need to keep that in mind
    image_dir = "Original"
    mask_dir = "Segmented"

    train_datasets = []
    val_datasets = []

    for d in os.listdir(os.path.join(train_dir, image_dir)):
        img_dir = os.path.join(train_dir, image_dir, d)
        msk_dir = os.path.join(train_dir, mask_dir, d)
        dataset = MedicalImageDataset(
            image_dir=img_dir, mask_dir=msk_dir, transform=train_transform
        )
        train_datasets.append(dataset)

    train_dataset = ConcatDataset(train_datasets)

    for d in os.listdir(os.path.join(test_dir, image_dir)):
        img_dir = os.path.join(test_dir, image_dir, d)
        msk_dir = os.path.join(test_dir, mask_dir, d)
        dataset = MedicalImageDataset(img_dir, msk_dir, transform=val_transform)
        val_datasets.append(dataset)

    val_dataset = ConcatDataset(val_datasets)

    # now create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workders,
        pin_memory=pin_memory,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workders,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


def check_accuracy(loader, model, device="cuda"):  # pragma: no cover
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    print(
        f"Got {num_correct}/{num_pixels} with accuracy {num_correct/num_pixels*100:.2f}\nDice Score: {dice_score/len(loader)}"
    )
    model.train()


def save_predictions_as_images(
    loader, model, folder="predictions", device="cuda"
):  # pragma: no cover
    model.eval()
    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = preds > 0.5

        pred_images, labels = extract_mask_label_from_batch(preds)
        original_imgs, original_labels = extract_mask_label_from_batch(y)
        for idx in range(len(pred_images)):
            torchvision.utils.save_image(
                pred_images[idx],
                os.path.join(folder, f"pred_{get_label(labels[idx])}_{batch_idx}.jpg"),
            )
            torchvision.utils.save_image(
                original_imgs[idx],
                os.path.join(
                    folder,
                    f"original_{get_label(original_labels[idx])}_{batch_idx}.jpg",
                ),
            )
