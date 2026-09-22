# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conversion script for the Stable Diffusion checkpoints."""

import os
from contextlib import nullcontext
from io import BytesIO
from urllib.parse import urlparse

import requests
import yaml
from transformers import (
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)

from ..models import UNet2DConditionModel
from ..schedulers import (
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from ..utils import is_accelerate_available, logging


if is_accelerate_available():
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

CONFIG_URLS = {
    "v1": "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml",
    "v2": "https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/configs/stable-diffusion/v2-inference-v.yaml",
    "xl": "https://raw.githubusercontent.com/Stability-AI/generative-models/main/configs/inference/sd_xl_base.yaml",
    "xl_refiner": "https://raw.githubusercontent.com/Stability-AI/generative-models/main/configs/inference/sd_xl_refiner.yaml",
    "upscale": "https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/configs/stable-diffusion/x4-upscaling.yaml",
    "controlnet": "https://raw.githubusercontent.com/lllyasviel/ControlNet/main/models/cldm_v15.yaml",
}

CHECKPOINT_KEY_NAMES = {
    "v2": "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight",
    "xl_base": "conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.bias",
    "xl_refiner": "conditioner.embedders.0.model.transformer.resblocks.9.mlp.c_proj.bias",
}

SCHEDULER_DEFAULT_CONFIG = {
    "beta_schedule": "scaled_linear",
    "beta_start": 0.00085,
    "beta_end": 0.012,
    "interpolation_type": "linear",
    "num_train_timesteps": 1000,
    "prediction_type": "epsilon",
    "sample_max_value": 1.0,
    "set_alpha_to_one": False,
    "skip_prk_steps": True,
    "steps_offset": 1,
    "timestep_spacing": "leading",
}

DIFFUSERS_TO_LDM_MAPPING = {
    "unet": {
        "layers": {
            "time_embedding.linear_1.weight": "time_embed.0.weight",
            "time_embedding.linear_1.bias": "time_embed.0.bias",
            "time_embedding.linear_2.weight": "time_embed.2.weight",
            "time_embedding.linear_2.bias": "time_embed.2.bias",
            "conv_in.weight": "input_blocks.0.0.weight",
            "conv_in.bias": "input_blocks.0.0.bias",
            "conv_norm_out.weight": "out.0.weight",
            "conv_norm_out.bias": "out.0.bias",
            "conv_out.weight": "out.2.weight",
            "conv_out.bias": "out.2.bias",
        },
        "class_embed_type": {
            "class_embedding.linear_1.weight": "label_emb.0.0.weight",
            "class_embedding.linear_1.bias": "label_emb.0.0.bias",
            "class_embedding.linear_2.weight": "label_emb.0.2.weight",
            "class_embedding.linear_2.bias": "label_emb.0.2.bias",
        },
        "addition_embed_type": {
            "add_embedding.linear_1.weight": "label_emb.0.0.weight",
            "add_embedding.linear_1.bias": "label_emb.0.0.bias",
            "add_embedding.linear_2.weight": "label_emb.0.2.weight",
            "add_embedding.linear_2.bias": "label_emb.0.2.bias",
        },
    },
    "vae": {
        "layers": {
            "encoder.conv_in.weight": "encoder.conv_in.weight",
            "encoder.conv_in.bias": "encoder.conv_in.bias",
            "encoder.conv_out.weight": "encoder.conv_out.weight",
            "encoder.conv_out.bias": "encoder.conv_out.bias",
            "encoder.conv_norm_out.weight": "encoder.norm_out.weight",
            "encoder.conv_norm_out.bias": "encoder.norm_out.bias",
            "decoder.conv_in.weight": "decoder.conv_in.weight",
            "decoder.conv_in.bias": "decoder.conv_in.bias",
            "decoder.conv_out.weight": "decoder.conv_out.weight",
            "decoder.conv_out.bias": "decoder.conv_out.bias",
            "decoder.conv_norm_out.weight": "decoder.norm_out.weight",
            "decoder.conv_norm_out.bias": "decoder.norm_out.bias",
            "quant_conv.weight": "quant_conv.weight",
            "quant_conv.bias": "quant_conv.bias",
            "post_quant_conv.weight": "post_quant_conv.weight",
            "post_quant_conv.bias": "post_quant_conv.bias",
        },
        "class_embed_type": {},
        "addition_embed_type": {},
    },
    "text_encoder": {
        "layers": {
            "text_model.embeddings.token_embedding.weight": "token_embedding.weight",
            "text_model.embeddings.position_embedding.weight": "text_model.embeddings.position_embedding.weight",
            "text_model.final_layer_norm.weight": "final_layer_norm.weight",
            "text_model.final_layer_norm.bias": "final_layer_norm.bias",
        },
        "class_embed_type": {},
        "addition_embed_type": {},
    },
    "text_encoder_2": {
        "layers": {
            "text_model.embeddings.token_embedding.weight": "token_embedding.weight",
            "text_model.embeddings.position_embedding.weight": "text_model.embeddings.position_embedding.weight",
            "text_model.final_layer_norm.weight": "final_layer_norm.weight",
            "text_model.final_layer_norm.bias": "final_layer_norm.bias",
        },
        "class_embed_type": {},
        "addition_embed_type": {},
    },
}

MODEL_TYPE_TO_CONFIG = {
    "SDXL": "v2",
    "SDXL-Refiner": "xl_refiner",
    "SD-3": "v2",
    "SD-3-Refiner": "xl_refiner",
}

MODEL_TYPE_TO_CHECKPOINT_KEY_NAME = {
    "SDXL": "xl_base",
    "SDXL-Refiner": "xl_refiner",
    "SD-3": "xl_base",
    "SD-3-Refiner": "xl_refiner",
}

MODEL_TYPE_TO_CHECKPOINT_KEY_NAMES = {
    "SDXL": ["xl_base", "xl_refiner"],
    "SDXL-Refiner": ["xl_base", "xl_refiner"],
    "SD-3": ["xl_base", "xl_refiner"],
    "SD-3-Refiner": ["xl_base", "xl_refiner"],
}

MODEL_TYPE_TO_CONFIG_URL = {
    "SDXL": CONFIG_URLS["xl"],
    "SDXL-Refiner": CONFIG_URLS["xl_refiner"],
    "SD-3": CONFIG_URLS["xl"],
    "SD-3-Refiner": CONFIG_URLS["xl_refiner"],
}

MODEL_TYPE_TO_CHECKPOINT_URL = {
    "SDXL": CONFIG_URLS["xl"],
    "SDXL-Refiner": CONFIG_URLS["xl_refiner"],
    "SD-3": CONFIG_URLS["xl"],
    "SD-3-Refiner": CONFIG_URLS["xl_refiner"],
}

MODEL_TYPE_TO_SCHEDULER_CONFIG = {
    "SDXL": {"prediction_type": "v_prediction"},
    "SDXL-Refiner": {},
    "SD-3": {},
    "SD-3-Refiner": {},
}


def create_diffusers_unet_model_from_ldm(
    pipeline_class_name,
    original_config,
    checkpoint,
    num_in_channels=None,
    image_size=None,
):
    if pipeline_class_name == "StableDiffusion3Img2ImgPipeline":
        num_in_channels = 8

    elif pipeline_class_name == "StableDiffusion3InpaintPipeline":
        num_in_channels = 10

    elif pipeline_class_name == "StableDiffusion3ControlNetImg2ImgPipeline":
        num_in_channels = 11

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintPipeline":
        num_in_channels = 12

    elif pipeline_class_name == "StableDiffusion3InpaintCombinedPipeline":
        num_in_channels = 12

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 13

    elif pipeline_class_name == "StableDiffusion3ControlNetInpaintCombinedPipeline":
        num_in_channels = 