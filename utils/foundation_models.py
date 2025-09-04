import os
import logging

import torch
import timm
from transformers import AutoImageProcessor, AutoModel

logger = logging.getLogger("DeTok")


def create_foundation_model(model_type: str = "dinov2"):
    if model_type == "dinov2":
        ckpt_path = os.path.join("offline_models", "dinov2_vit_large_patch14", "pytorch_model.bin")

        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location="cpu")
            
            foundation_model = timm.create_model(
                "vit_large_patch14_dinov2.lvd142m", 
                pretrained=False,
                dynamic_img_size=True,
            )
            foundation_model.load_state_dict(state_dict["model_state_dict"])
            logger.info(f"[Foundation Model] Loaded foundation model DINOv2 from {ckpt_path}")            
        else:
            foundation_model = timm.create_model("vit_large_patch14_dinov2.lvd142m", pretrained=True, dynamic_img_size=True, img_size=224)
            logger.info(f"[Foundation Model] Loaded foundation model DINOv2 from timm")

    elif model_type == "dinov3":
        ckpt_path = os.path.join("offline_models", "dinov3_vit_large_patch14")

        if os.path.exists(ckpt_path):
            transforms = AutoImageProcessor.from_pretrained(ckpt_path)
            foundation_model = AutoModel.from_pretrained(
                ckpt_path, 
            )
            logger.info(f"[Foundation Model] Loaded foundation model DINOv3 from {ckpt_path}")
        else:
            transforms = AutoImageProcessor.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m")
            foundation_model = AutoModel.from_pretrained(
                "facebook/dinov3-vitl16-pretrain-lvd1689m", 
            )
            logger.info(f"[Foundation Model] Loaded foundation model DINOv3 from transformers")

        return foundation_model, transforms

    elif model_type == "clip":
        ckpt_path = os.path.join("offline_models", "clip_vit_large_patch14", "pytorch_model.bin")

        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location="cpu")

            foundation_model = timm.create_model(
                "vit_large_patch14_clip.lvd142m", 
                pretrained=False, 
                dynamic_img_size=True
            )
            foundation_model.load_state_dict(state_dict["model_state_dict"])
            logger.info(f"[Foundation Model] Loaded foundation model CLIP from {ckpt_path}")
        else:
            foundation_model = timm.create_model(
                "vit_large_patch14_clip.lvd142m", 
                pretrained=True, 
                dynamic_img_size=True
            )
            logger.info(f"[Foundation Model] Loaded foundation model CLIP from timm")

    elif model_type == "siglip":
        ckpt_path = os.path.join("offline_models", "siglip_vit_large_patch14", "pytorch_model.bin")

        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location="cpu")

            foundation_model = timm.create_model(
                "vit_large_patch16_siglip_256", 
                pretrained=False, 
                dynamic_img_size=True, 
            )
            foundation_model.load_state_dict(state_dict)
            logger.info(f"[Foundation Model] Loaded foundation model SigLIP from {ckpt_path}")
        else:
            foundation_model = timm.create_model("vit_large_patch16_siglip_256", pretrained=True, dynamic_img_size=True, img_size=256)
            logger.info(f"[Foundation Model] Loaded foundation model SigLIP from timm")

    else:
        raise ValueError(f"Unsupported foundation model type: {model_type}")
    
    # Initialize preprocessor
    data_config = timm.data.resolve_model_data_config(foundation_model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    return foundation_model, transforms