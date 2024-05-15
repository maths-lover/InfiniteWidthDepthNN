import torch

from unet_segmentation.UNet_model import UNet


def test_UNet_model():
    x = torch.rand((3, 3, 224, 224))
    y = torch.rand((3, 5, 224, 224))
    model = UNet(in_channels=3, out_channels=5)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == y.shape
