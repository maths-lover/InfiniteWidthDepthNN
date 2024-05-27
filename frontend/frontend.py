import gradio as gr
import torch
import torchvision.transforms.functional as TF

from unet_segmentation.UNet_model import UNet
from unet_segmentation.UNet_train import DEVICE
from unet_segmentation.utils import extract_mask_label, load_checkpoint

# initiate the model
model = UNet(in_channels=3, out_channels=5).to(DEVICE)

# load the model
load_checkpoint(torch.load("my_checkpoint.pth.zip"), model)


def process_image(image):
    tensor_image = TF.pil_to_tensor(image).to(device=DEVICE).float()
    with torch.no_grad():
        pred = torch.sigmoid(model(tensor_image.unsqueeze(0)))
        pred = (pred > 0.5).float()

    pred_image, _ = extract_mask_label(pred.squeeze(0))
    print(pred_image.shape)
    return TF.to_pil_image(pred_image.byte() * 255, mode="L")


with gr.Blocks() as demo:
    # Add heading
    gr.Markdown(
        "<h1>Infinite-Depth Neural Networks with Segmentation for Cancer Care</h1>"
    )

    # Add input and output image boxes in the same column with a button
    with gr.Column():
        image_input = gr.Image(type="pil", label="Input Image")
        image_output = gr.Image(type="pil", label="Output Image")
        process_button = gr.Button("Process")

    # Define the button click action
    process_button.click(fn=process_image, inputs=image_input, outputs=image_output)

# Launch the interface
demo.launch(show_api=False)
