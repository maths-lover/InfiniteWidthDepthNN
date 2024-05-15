import torch

from unet_segmentation.utils import extract_mask_label, extract_mask_label_from_batch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_extract_mask_label():
    # x is a 3 channel 5x5 b&w image with a white dot in between
    x = torch.zeros(3, 5, 5).to(device=DEVICE).float()
    x[0][2][2] = 1.0

    # there are two possible labels for this one, so setting label for second second channel
    x[2][2][2] = 1.0

    mask, label = extract_mask_label(x)

    assert label == 1
    assert torch.equal(mask[0], x[0])


def test_extract_mask_label_from_batch():
    # x is a batch of 16, 3-channel, 5x5 images
    x = torch.zeros(16, 3, 5, 5)
    for i in range(x.size(0)):
        x[i][0][2][2] = 1.0

    r1 = 1
    r2 = 3
    valid_labels = []
    for i in range(x.size(0)):
        # get a random int between 1 and 3 (half inclusive) i.e., [1,3)
        approved_label = int(torch.rand(1).uniform_(r1, r2))
        x[i][approved_label][2][2] = 1.0
        valid_labels.append(approved_label)

    masks, labels = extract_mask_label_from_batch(x)
    for idx in range(len(masks)):
        assert torch.equal(masks[idx][0], x[idx][0])
        assert labels[idx] == (valid_labels[idx] - 1)
