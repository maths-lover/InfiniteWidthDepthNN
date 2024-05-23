import os

import torch
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import v2 as v2_transforms
from tqdm import tqdm


def get_label_id(label):
    match label:
        case "Benign":
            return 0
        case "Early":
            return 1
        case "Pre":
            return 2
        case "Pro":
            return 3
        case _:
            return -1


def get_label(id):
    match id:
        case 0:
            return "Benign"
        case 1:
            return "Early_Malignant"
        case 2:
            return "Pre_Malignant"
        case 3:
            return "Pro_Malignant"
        case _:
            return "Unknown"


class MedicalImageDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])

        # original image
        image = read_image(img_path, mode=ImageReadMode.RGB)
        image = image.float() / 255.0

        # mask image
        mask = read_image(mask_path, mode=ImageReadMode.RGB)
        mask = v2_transforms.Grayscale()(mask)

        # set black or white
        mask[mask < 100] = 0.0
        mask[mask >= 100] = 1.0

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        label = get_label_id(os.path.basename(os.path.dirname(img_path)))

        # remove dimension
        mask = torch.squeeze(mask)

        # create a multi-channel mask with zeroes in the beginning
        multi_channel_mask = torch.zeros((5, mask.size(1), mask.unsqueeze(0).size(2)))

        # copy the mask to first channel
        multi_channel_mask[0][mask == 1.0] = 1.0

        # have a copy of the mask in label's channel
        multi_channel_mask[1][mask > 0] = 1.0 if label == 0 else 0.0
        multi_channel_mask[2][mask > 0] = 1.0 if label == 1 else 0.0
        multi_channel_mask[3][mask > 0] = 1.0 if label == 2 else 0.0
        multi_channel_mask[4][mask > 0] = 1.0 if label == 3 else 0.0

        # we need to use sigmoid on last activation

        return image, multi_channel_mask


# split_data splits data into train and test directory
def split_data(dataset_dir, orig, msk_dir):  # pragma: no cover
    original_dir = os.path.join(dataset_dir, orig)
    mask_dir = os.path.join(dataset_dir, msk_dir)

    train_dir = os.path.join(dataset_dir, "Train_Data")
    test_dir = os.path.join(dataset_dir, "Test_Data")

    # creat train and test directories if they don't exist
    train_original_dir = os.path.join(train_dir, orig)
    test_original_dir = os.path.join(test_dir, orig)
    train_mask_dir = os.path.join(train_dir, msk_dir)
    test_mask_dir = os.path.join(test_dir, msk_dir)
    if not os.path.exists(train_original_dir):
        os.makedirs(train_original_dir)
    if not os.path.exists(test_original_dir):
        os.makedirs(test_original_dir)
    if not os.path.exists(train_mask_dir):
        os.makedirs(train_mask_dir)
    if not os.path.exists(test_mask_dir):
        os.makedirs(test_mask_dir)

    # now we copy 80% of the data to train and 20% to test
    # while maintaining the same directory structure
    labels = tqdm(os.listdir(original_dir), position=0)

    for label in labels:
        label_dir = os.path.join(original_dir, label)
        label_mask_dir = os.path.join(mask_dir, label)

        train_original_label_dir = os.path.join(train_original_dir, label)
        test_original_label_dir = os.path.join(test_original_dir, label)
        train_mask_label_dir = os.path.join(train_mask_dir, label)
        test_mask_label_dir = os.path.join(test_mask_dir, label)
        if not os.path.exists(train_original_label_dir):
            os.makedirs(train_original_label_dir)
        if not os.path.exists(test_original_label_dir):
            os.makedirs(test_original_label_dir)
        if not os.path.exists(train_mask_label_dir):
            os.makedirs(train_mask_label_dir)
        if not os.path.exists(test_mask_label_dir):
            os.makedirs(test_mask_label_dir)

        images = tqdm(os.listdir(label_dir), position=1, leave=False)
        for i, img in enumerate(images):
            img_path = os.path.join(label_dir, img)
            mask_path = os.path.join(label_mask_dir, img)
            if i < len(images) * 0.2:
                os.system(f"cp {img_path} {train_original_label_dir}")
                os.system(f"cp {mask_path} {train_mask_label_dir}")
            elif i < len(images) * 0.25:
                os.system(f"cp {img_path} {test_original_label_dir}")
                os.system(f"cp {mask_path} {test_mask_label_dir}")
            else:
                break
    print("Data split successfully!")
    return train_dir, test_dir
