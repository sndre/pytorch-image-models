import timm
import torch

from datasets import load_dataset
from transformers import AutoImageProcessor, ViTForImageClassification

import slicegpt.layernorm_fusion as layernorm_fusion
import slicegpt.rotate as rotate
from slicegpt.adapters.vit_adapter import VitModelAdapter
from slicegpt.slicing_scheduler import ConstSlicingScheduler

device = torch.device("cpu")
def batch_loader():
    torch.manual_seed(42)
    yield {"x": torch.randn((1, 3, 224, 224))}

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

def convert_model_optimized():
    print("slicing model using optimized PCA computation method")
    # model to slice using optimized PCA computation
    optimized_model = timm.create_model('vit_base_patch16_224.orig_in21k_ft_in1k', pretrained=True).to(device).eval()
    optimized_model_adapter = VitModelAdapter(optimized_model)
    layernorm_fusion.replace_layers(optimized_model_adapter)
    layernorm_fusion.fuse_modules(optimized_model_adapter)
    sparsity, round_interval = 0.25, 8
    new_embedding_dimension = int((1 - sparsity) * optimized_model_adapter.hidden_size)
    new_embedding_dimension -= new_embedding_dimension % round_interval
    scheduler = ConstSlicingScheduler(new_embedding_dimension)
    rotate.optimized_rotate_and_slice_sequential(optimized_model_adapter, batch_loader, scheduler, apply_mask=False, final_orientation="pca")
    optimized_model.to(device)
    return optimized_model_adapter

def convert_model_original():
    print("slicing model using original PCA computation method")
    # model to slice using original PCA computation
    original_model = timm.create_model('vit_base_patch16_224.orig_in21k_ft_in1k', pretrained=True).to(device).eval()
    original_model_adapter = VitModelAdapter(original_model)
    layernorm_fusion.replace_layers(original_model_adapter)
    layernorm_fusion.fuse_modules(original_model_adapter)
    sparsity, round_interval = 0.25, 8
    new_embedding_dimension = int((1 - sparsity) * original_model_adapter.hidden_size)
    new_embedding_dimension -= new_embedding_dimension % round_interval
    scheduler = ConstSlicingScheduler(new_embedding_dimension)
    rotate.rotate_and_slice_sequential(original_model_adapter, batch_loader(), scheduler, apply_mask=False, final_orientation="pca")
    original_model.to(device)
    return original_model_adapter

def test_pca_equality():
    optimized_model = convert_model_optimized().model
    original_model = convert_model_original().model

    # run inference against both models and compare results
    torch.manual_seed(11)
    sample_image = torch.randn((1, 3, 224, 224))

    expected_result = original_model(sample_image)
    actual_result = optimized_model(sample_image)
    assert torch.allclose(expected_result, actual_result, atol=1e-5, rtol=1e-5)

def test_Q1_equality():
    optimized_model_adapter = convert_model_optimized()
    original_model_adapter = convert_model_original()
    assert torch.allclose(original_model_adapter.Q1, optimized_model_adapter.Q1, atol=1e-5, rtol=1e-5)

def test_Q2_equality():
    optimized_model_adapter = convert_model_optimized()
    original_model_adapter = convert_model_original()
    for idx, Q2 in enumerate(original_model_adapter.Q2s):
        # print("layer's", idx, "Q2s match =", torch.allclose(Q2, optimized_model_adapter.Q2s[idx], atol=1e-5, rtol=1e-5))
        assert torch.allclose(Q2, optimized_model_adapter.Q2s[idx], atol=1e-5, rtol=1e-5)
