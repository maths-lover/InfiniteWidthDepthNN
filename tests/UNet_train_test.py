import torch

from unet_segmentation.UNet_train import train_transform, val_transform


def test_upsample():
    x = torch.rand(3, 224, 224)
    y = torch.rand(1, 224, 224)

    train_image, train_mask = train_transform(x, y)
    val_image, val_mask = val_transform(x, y)

    final_image_size = torch.Size([3, 224 * 5, 224 * 5])
    final_mask_size = torch.Size([1, 224 * 5, 224 * 5])

    assert final_image_size == train_image.shape
    assert final_mask_size == train_mask.shape
    assert final_image_size == val_image.shape
    assert final_mask_size == val_mask.shape
