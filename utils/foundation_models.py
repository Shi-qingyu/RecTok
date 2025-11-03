import os
import logging

import torch
import torch.nn as nn
import timm

from transformers import (
    AutoImageProcessor, 
    AutoModel,
    SamVisionModel,
    SamImageProcessor,
    SamVisionConfig,
)

import models

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

    elif model_type == "dinov3b":
        ckpt_path = os.path.join("offline_models", "dinov3_vit_base_patch14")

        if os.path.exists(ckpt_path):
            transforms = AutoImageProcessor.from_pretrained(ckpt_path)
            foundation_model = AutoModel.from_pretrained(
                ckpt_path, 
            )
            logger.info(f"[Foundation Model] Loaded foundation model DINOv3 from {ckpt_path}")
        else:
            transforms = AutoImageProcessor.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")
            foundation_model = AutoModel.from_pretrained(
                "facebook/dinov3-vitb16-pretrain-lvd1689m", 
            )
            logger.info(f"[Foundation Model] Loaded foundation model DINOv3 from transformers")

        return foundation_model, transforms

    elif model_type == "sam":
        ckpt_path = os.path.join("offline_models", "sam-vit-large")
        if os.path.exists(ckpt_path):
            cfg = SamVisionConfig.from_pretrained(ckpt_path)
            cfg.image_size = 256
            cfg.use_abs_pos = False
            cfg.use_rel_pos = False
            
            transforms = SamImageProcessor.from_pretrained(ckpt_path)
            foundation_model = SamVisionModel.from_pretrained(
                ckpt_path,
                config=cfg,
                ignore_mismatched_sizes=True,
                low_cpu_mem_usage=True,
            )
            logger.info(f"[Foundation Model] Loaded foundation model SAM from {ckpt_path}")
        else:
            cfg = SamVisionConfig.from_pretrained("facebook/sam-vit-large")
            cfg.image_size = 256
            cfg.use_abs_pos = False
            cfg.use_rel_pos = False
            
            transforms = SamImageProcessor.from_pretrained("facebook/sam-vit-large")
            foundation_model = SamVisionModel.from_pretrained(
                "facebook/sam-vit-large",
                config=cfg,
                ignore_mismatched_sizes=True,
                low_cpu_mem_usage=True,
            )
            logger.info(f"[Foundation Model] Loaded foundation model SAM from transformers")

        return foundation_model, transforms

    elif model_type == "radio":
        ckpt_path = os.path.join("offline_models", "radio_l_2.5.pt")
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location="cpu")
            foundation_model = AutoModel.from_pretrained(ckpt_path, trust_remote_code=True)
            logger.info(f"[Foundation Model] Loaded foundation model RADIO from {ckpt_path}")
        else:
            foundation_model = AutoModel.from_pretrained("nvidia/RADIO-L", trust_remote_code=True)
            logger.info(f"[Foundation Model] Loaded foundation model RADIO from transformers")

        return foundation_model, None

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

    elif model_type in models.DeTok_models:
        foundation_model = models.DeTok_models[model_type](
            img_size=256,
            patch_size=16,
            token_channels=16,
            gamma=0.0,
            mask_ratio=0.0,
            num_register_tokens=0,
        )
        weights = torch.load("work_dirs/tokenizer_training/detokBB-ch16-g3.0-m0.7-auxdinov2/checkpoints/epoch_0199.pth", weights_only=False, map_location="cpu")
        if "model_ema" in weights:
            weights = weights["model_ema"]
            msg = foundation_model.load_state_dict(weights, strict=False)
            logger.info(f"[Aux Tokenizer] Missing keys: {msg.missing_keys}")
            logger.info(f"[Aux Tokenizer] Unexpected keys: {msg.unexpected_keys}")
        else:
            weights = weights["model"] if "model" in weights else weights
            msg = foundation_model.load_state_dict(weights, strict=False)
            logger.info(f"[Aux Tokenizer] Missing keys: {msg.missing_keys}")
            logger.info(f"[Aux Tokenizer] Unexpected keys: {msg.unexpected_keys}")
        logger.info(f"[Foundation Model] Loaded foundation model {model_type} from models.detok")
        
        return foundation_model, nn.Identity()
    else:
        raise ValueError(f"Unsupported foundation model type: {model_type}")
    
    # Initialize preprocessor
    data_config = timm.data.resolve_model_data_config(foundation_model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    return foundation_model, transforms