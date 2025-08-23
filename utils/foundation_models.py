import os
import logging
import torch
import timm

logger = logging.getLogger("DeTok")


def create_foundation_model(model_type: str = "dinov2"):
    if model_type == "dinov2":
        ckpt_path = os.path.join("offline_models", "dinov2_vit_large_patch14", "pytorch_model.bin")

        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location="cpu")
            
            foundation_model = timm.create_model("vit_large_patch14_dinov2.lvd142m", pretrained=False, dynamic_img_size=True)
            foundation_model.load_state_dict(state_dict["model_state_dict"])
            logger.info(f"[Foundation Model] Loaded foundation model from {ckpt_path}")            
        else:
            foundation_model = timm.create_model("vit_large_patch14_dinov2.lvd142m", pretrained=True, dynamic_img_size=True)
            logger.info(f"[Foundation Model] Loaded foundation model from timm")
        
        return foundation_model
    elif model_type == "clip":
        ckpt_path = os.path.join("offline_models", "clip_vit_large_patch14", "pytorch_model.bin")

        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location="cpu")

            foundation_model = timm.create_model("vit_large_patch14_clip.lvd142m", pretrained=False, dynamic_img_size=True)
            foundation_model.load_state_dict(state_dict["model_state_dict"])
            logger.info(f"[Foundation Model] Loaded foundation model from {ckpt_path}")
        else:
            foundation_model = timm.create_model("vit_large_patch14_clip.lvd142m", pretrained=True, dynamic_img_size=True)
            logger.info(f"[Foundation Model] Loaded foundation model from timm")

        return foundation_model
    else:
        raise ValueError(f"Unsupported foundation model type: {model_type}") 