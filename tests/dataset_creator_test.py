from unet_segmentation.dataset_creator import get_label, get_label_id


def test_get_label_id():
    id = get_label_id("Benign")
    assert id == 0

    id = get_label_id("Early")
    assert id == 1

    id = get_label_id("Pre")
    assert id == 2

    id = get_label_id("Pro")
    assert id == 3

    id = get_label_id("Something unknown")
    assert id == -1


def test_get_label():
    label = get_label(0)
    assert label == "Benign"

    label = get_label(1)
    assert label == "Early_Malignant"

    label = get_label(2)
    assert label == "Pre_Malignant"

    label = get_label(3)
    assert label == "Pro_Malignant"

    label = get_label(4)
    assert label == "Unknown"
