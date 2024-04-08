import timm
import torch

from datasets import load_dataset
from transformers import AutoImageProcessor, ViTForImageClassification

def test_model_inference():
    device = torch.device("mps")
    image = load_dataset("huggingface/cats-image", trust_remote_code=True)["test"]["image"][0]

    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    image = image_processor(image, return_tensors="pt").pixel_values.to(device)

    model = timm.create_model('vit_base_patch16_224.orig_in21k_ft_in1k', pretrained=True).to(device).eval()
    logits = model(image)
    predicted_label = logits.argmax(-1).item()
    print("predicted_label:", predicted_label)

    hf_model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224").to(device)
    print(hf_model.config.id2label[predicted_label])

    assert predicted_label == 285, f"Expected {285}, got {predicted_label}"

