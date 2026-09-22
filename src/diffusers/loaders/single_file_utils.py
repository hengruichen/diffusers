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
            "conv_in.weight": "encoder.conv_in.weight",
            "conv_in.bias": "encoder.conv_in.bias",
            "conv_out.weight": "decoder.conv_out.weight",
            "conv_out.bias": "decoder.conv_out.bias",
            "norm_out.weight": "decoder.conv_out.weight",
            "norm_out.bias": "decoder.conv_out.bias",
            "conv_norm_out.weight": "decoder.conv_out.weight",
            "conv_norm_out.bias": "decoder.conv_out.bias",
            "conv_in.weight": "encoder.conv_in.weight",
            "conv_in.bias": "encoder.conv_in.bias",
            "conv_out.weight": "decoder.conv_out.weight",
            "conv_out.bias": "decoder.conv_out.bias",
            "norm_out.weight": "decoder.conv_out.weight",
            "norm_out.bias": "decoder.conv_out.bias",
            "conv_norm_out.weight": "decoder.conv_out.weight",
            "conv_norm_out.bias": "decoder.conv_out.bias",
        },
        "norm": {
            "norm1.weight": "encoder.down_blocks.0.resnets.0.norm1.weight",
            "norm1.bias": "encoder.down_blocks.0.resnets.0.norm1.bias",
            "norm2.weight": "encoder.down_blocks.0.resnets.0.norm2.weight",
            "norm2.bias": "encoder.down_blocks.0.resnets.0.norm2.bias",
            "norm1.weight": "encoder.down_blocks.0.resnets.0.norm1.weight",
            "norm1.bias": "encoder.down_blocks.0.resnets.0.norm1.bias",
            "norm2.weight": "encoder.down_blocks.0.resnets.0.norm2.weight",
            "norm2.bias": "encoder.down_blocks.0.resnets.0.norm2.bias",
            "norm1.weight": "encoder.down_blocks.1.resnets.0.norm1.weight",
            "norm1.bias": "encoder.down_blocks.1.resnets.0.norm1.bias",
            "norm2.weight": "encoder.down_blocks.1.resnets.0.norm2.weight",
            "norm2.bias": "encoder.down_blocks.1.resnets.0.norm2.bias",
            "norm1.weight": "encoder.down_blocks.1.resnets.0.norm1.weight",
            "norm1.bias": "encoder.down_blocks.1.resnets.0.norm1.bias",
            "norm2.weight": "encoder.down_blocks.1.resnets.0.norm2.weight",
            "norm2.bias": "encoder.down_blocks.1.resnets.0.norm2.bias",
            "norm1.weight": "encoder.down_blocks.2.resnets.0.norm1.weight",
            "norm1.bias": "encoder.down_blocks.2.resnets.0.norm1.bias",
            "norm2.weight": "encoder.down_blocks.2.resnets.0.norm2.weight",
            "norm2.bias": "encoder.down_blocks.2.resnets.0.norm2.bias",
            "norm1.weight": "encoder.down_blocks.2.resnets.0.norm1.weight",
            "norm1.bias": "encoder.down_blocks.2.resnets.0.norm1.bias",
            "norm2.weight": "encoder.down_blocks.2.resnets.0.norm2.weight",
            "norm2.bias": "encoder.down_blocks.2.resnets.0.norm2.bias",
            "norm1.weight": "encoder.down_blocks.3.resnets.0.norm1.weight",
            "norm1.bias": "encoder.down_blocks.3.resnets.0.norm1.bias",
            "norm2.weight": "encoder.down_blocks.3.resnets.0.norm2.weight",
            "norm2.bias": "encoder.down_blocks.3.resnets.0.norm2.bias",
            "norm1.weight": "encoder.down_blocks.3.resnets.0.norm1.weight",
            "norm1.bias": "encoder.down_blocks.3.resnets.0.norm1.bias",
            "norm2.weight": "encoder.down_blocks.3.resnets.0.norm2.weight",
            "norm2.bias": "encoder.down_blocks.3.resnets.0.norm2.bias",
            "norm1.weight": "encoder.mid_block.resnets.0.norm1.weight",
            "norm1.bias": "encoder.mid_block.resnets.0.norm1.bias",
            "norm2.weight": "encoder.mid_block.resnets.0.norm2.weight",
            "norm2.bias": "encoder.mid_block.resnets.0.norm2.bias",
            "norm1.weight": "encoder.mid_block.resnets.0.norm1.weight",
            "norm1.bias": "encoder.mid_block.resnets.0.norm1.bias",
            "norm2.weight": "encoder.mid_block.resnets.0.norm2.weight",
            "norm2.bias": "encoder.mid_block.resnets.0.norm2.bias",
            "norm1.weight": "decoder.up_blocks.0.resnets.0.norm1.weight",
            "norm1.bias": "decoder.up_blocks.0.resnets.0.norm1.bias",
            "norm2.weight": "decoder.up_blocks.0.resnets.0.norm2.weight",
            "norm2.bias": "decoder.up_blocks.0.resnets.0.norm2.bias",
            "norm1.weight": "decoder.up_blocks.0.resnets.0.norm1.weight",
            "norm1.bias": "decoder.up_blocks.0.resnets.0.norm1.bias",
            "norm2.weight": "decoder.up_blocks.0.resnets.0.norm2.weight",
            "norm2.bias": "decoder.up_blocks.0.resnets.0.norm2.bias",
            "norm1.weight": "decoder.up_blocks.1.resnets.0.norm1.weight",
            "norm1.bias": "decoder.up_blocks.1.resnets.0.norm1.bias",
            "norm2.weight": "decoder.up_blocks.1.resnets.0.norm2.weight",
            "norm2.bias": "decoder.up_blocks.1.resnets.0.norm2.bias",
            "norm1.weight": "decoder.up_blocks.1.resnets.0.norm1.weight",
            "norm1.bias": "decoder.up_blocks.1.resnets.0.norm1.bias",
            "norm2.weight": "decoder.up_blocks.1.resnets.0.norm2.weight",
            "norm2.bias": "decoder.up_blocks.1.resnets.0.norm2.bias",
            "norm1.weight": "decoder.up_blocks.2.resnets.0.norm1.weight",
            "norm1.bias": "decoder.up_blocks.2.resnets.0.norm1.bias",
            "norm2.weight": "decoder.up_blocks.2.resnets.0.norm2.weight",
            "norm2.bias": "decoder.up_blocks.2.resnets.0.norm2.bias",
            "norm1.weight": "decoder.up_blocks.2.resnets.0.norm1.weight",
            "norm1.bias": "decoder.up_blocks.2.resnets.0.norm1.bias",
            "norm2.weight": "decoder.up_blocks.2.resnets.0.norm2.weight",
            "norm2.bias": "decoder.up_blocks.2.resnets.0.norm2.bias",
            "norm1.weight": "decoder.up_blocks.3.resnets.0.norm1.weight",
            "norm1.bias": "decoder.up_blocks.3.resnets.0.norm1.bias",
            "norm2.weight": "decoder.up_blocks.3.resnets.0.norm2.weight",
            "norm2.bias": "decoder.up_blocks.3.resnets.0.norm2.bias",
            "norm1.weight": "decoder.up_blocks.3.resnets.0.norm1.weight",
            "norm1.bias": "decoder.up_blocks.3.resnets.0.norm1.bias",
            "norm2.weight": "decoder.up_blocks.3.resnets.0.norm2.weight",
            "norm2.bias": "decoder.up_blocks.3.resnets.0.norm2.bias",
            "norm1.weight": "decoder.mid_block.resnets.0.norm1.weight",
            "norm1.bias": "decoder.mid_block.resnets.0.norm1.bias",
            "norm2.weight": "decoder.mid_block.resnets.0.norm2.weight",
            "norm2.bias": "decoder.mid_block.resnets.0.norm2.bias",
            "norm1.weight": "decoder.mid_block.resnets.0.norm1.weight",
            "norm1.bias": "decoder.mid_block.resnets.0.norm1.bias",
            "norm2.weight": "decoder.mid_block.resnets.0.norm2.weight",
            "norm2.bias": "decoder.mid_block.resnets.0.norm2.bias",
        },
        "down": {
            "conv_shortcut.weight": "down_blocks.0.downsamplers.0.conv.weight",
            "conv_shortcut.bias": "down_blocks.0.downsamplers.0.conv.bias",
            "conv_shortcut.weight": "down_blocks.1.downsamplers.0.conv.weight",
            "conv_shortcut.bias": "down_blocks.1.downsamplers.0.conv.bias",
            "conv_shortcut.weight": "down_blocks.2.downsamplers.0.conv.weight",
            "conv_shortcut.bias": "down_blocks.2.downsamplers.0.conv.bias",
            "conv_shortcut.weight": "down_blocks.3.downsamplers.0.conv.weight",
            "conv_shortcut.bias": "down_blocks.3.downsamplers.0.conv.bias",
        },
        "up": {
            "conv_shortcut.weight": "up_blocks.0.upsamplers.0.conv.weight",
            "conv_shortcut.bias": "up_blocks.0.upsamplers.0.conv.bias",
            "conv_shortcut.weight": "up_blocks.1.upsamplers.0.conv.weight",
            "conv_shortcut.bias": "up_blocks.1.upsamplers.0.conv.bias",
            "conv_shortcut.weight": "up_blocks.2.upsamplers.0.conv.weight",
            "conv_shortcut.bias": "up_blocks.2.upsamplers.0.conv.bias",
            "conv_shortcut.weight": "up_blocks.3.upsamplers.0.conv.weight",
            "conv_shortcut.bias": "up_blocks.3.upsamplers.0.conv.bias",
        },
    },
    "text_encoder": {
        "layers": {
            "text_embedder.linear.weight": "text_model.embeddings.token_embedding.weight",
            "text_embedder.linear.bias": "text_model.embeddings.token_embedding.bias",
            "text_embedder.positional_embedding": "text_model.embeddings.position_embedding.weight",
            "text_embedder.ln_final.weight": "text_model.final_layer_norm.weight",
            "text_embedder.ln_final.bias": "text_model.final_layer_norm.bias",
        },
        "norm": {"text_embedder.ln_final.weight": "text_model.final_layer_norm.weight"},
    },
    "text_encoder_2": {
        "layers": {
            "text_embedder.linear.weight": "text_model.embeddings.token_embedding.weight",
            "text_embedder.linear.bias": "text_model.embeddings.token_embedding.bias",
            "text_embedder.positional_embedding": "text_model.embeddings.position_embedding.weight",
            "text_embedder.ln_final.weight": "text_model.final_layer_norm.weight",
            "text_embedder.ln_final.bias": "text_model.final_layer_norm.bias",
        },
        "norm": {"text_embedder.ln_final.weight": "text_model.final_layer_norm.weight"},
    },
    "controlnet": {
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
}

if is_accelerate_available():
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device


def _get_model_file(
    repo_id,
    *,
    weights_name=None,
    force_download=False,
    proxies=None,
    local_files_only=False,
    token=None,
    revision=None,
    cache_dir=None,
    resume_download=False,
):
    from huggingface_hub import hf_hub_download

    if weights_name is None:
        weights_name = "model.safetensors"

    try:
        file_info = hf_hub_download(
            repo_id=repo_id,
            filename=weights_name,
            force_download=force_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            cache_dir=cache_dir,
        )
    except Exception as e:
        raise ValueError(
            f"Can't load the checkpoint for '{repo_id}'. Make sure that the model is available on the Hugging Face Hub."
        ) from e

    return file_info


def _get_config_file(
    repo_id,
    *,
    config_name=None,
    force_download=False,
    proxies=None,
    local_files_only=False,
    token=None,
    revision=None,
    cache_dir=None,
):
    from huggingface_hub import hf_hub_download

    if config_name is None:
        config_name = "config.yaml"

    try:
        file_info = hf_hub_download(
            repo_id=repo_id,
            filename=config_name,
            force_download=force_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            cache_dir=cache_dir,
        )
    except Exception as e:
        raise ValueError(
            f"Can't load the config for '{repo_id}'. Make sure that the model is available on the Hugging Face Hub."
        ) from e

    return file_info


def create_text_encoder_from_ldm_checkpoint(
    config_name,
    checkpoint,
    prefix="text_model.",
    has_projection=False,
    local_files_only=False,
    **config_kwargs,
):
    config = yaml.safe_load(open(config_name, "r"))

    config = CLIPTextConfig(**config)

    config_kwargs = {
        "attention_dropout": 0.0,
        "hidden_act": "quick_gelu",
        "hidden_size": config.hidden_size,
        "initializer_factor": 1.0,
        "initializer_range": 0.02,
        "intermediate_size": 4096,
        "max_position_embeddings": 77,
        "num_attention_heads": 32,
        "num_hidden_layers": 23,
        "projection_dim": 768,
        "torch_dtype": "float32",
        "transformers_version": "4.25.0.dev0",
        "use_cache": False,
        "vocab_size": 49408,
    }

    config_kwargs.update(config_kwargs)

    text_encoder = CLIPTextModel(config=config)

    text_encoder_dict = checkpoint

    text_encoder_dict = {
        k.replace(prefix, ""): v for k, v in text_encoder_dict.items() if k.startswith(prefix) and "model" not in k
    }

    if has_projection:
        text_encoder_dict = {
            k.replace(prefix, ""): v for k, v in text_encoder_dict.items() if k.startswith(prefix) and "model" not in k
        }

    text_encoder.load_state_dict(text_encoder_dict)

    return text_encoder


def create_text_encoder_from_open_clip_checkpoint(
    config_name,
    checkpoint,
    prefix="text_model.",
    has_projection=False,
    local_files_only=False,
    **config_kwargs,
):
    config = yaml.safe_load(open(config_name, "r"))

    config = CLIPTextConfig(**config)

    config_kwargs = {
        "attention_dropout": 0.0,
        "hidden_act": "quick_gelu",
        "hidden_size": config.hidden_size,
        "initializer_factor": 1.0,
        "initializer_range": 0.02,
        "intermediate_size": 4096,
        "max_position_embeddings": 77,
        "num_attention_heads": 32,
        "num_hidden_layers": 23,
        "projection_dim": 768,
        "torch_dtype": "float32",
        "transformers_version": "4.25.0.dev0",
        "use_cache": False,
        "vocab_size": 49408,
    }

    config_kwargs.update(config_kwargs)

    text_encoder = CLIPTextModel(config=config)

    text_encoder_dict = checkpoint

    text_encoder_dict = {
        k.replace(prefix, ""): v for k, v in text_encoder_dict.items() if k.startswith(prefix) and "model" not in k
    }

    if has_projection:
        text_encoder_dict = {
            k.replace(prefix, ""): v for k, v in text_encoder_dict.items() if k.startswith(prefix) and "model" not in k
        }

    text_encoder.load_state_dict(text_encoder_dict)

    return text_encoder


def create_text_encoders_and_tokenizers_from_ldm(
    original_config,
    checkpoint,
    model_type=None,
    local_files_only=False,
):
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    tokenizer_2 = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="tokenizer_2")

    if model_type in ["SDXL", "SDXL-Refiner"]:
        tokenizer_2 = CLIPTokenizer.from_pretrained(config_name, pad_token="!", local_files_only=local_files_only)
        text_encoder_2 = create_text_encoder_from_open_clip_checkpoint(
            config_name,
            checkpoint,
            prefix=prefix,
            has_projection=True,
            local_files_only=local_files_only,
            **config_kwargs,
        )
    else:
        tokenizer_2 = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="tokenizer_2")
        text_encoder_2 = create_text_encoder_from_ldm_checkpoint(
            config_name,
            checkpoint,
            prefix=prefix,
            has_projection=True,
            local_files_only=local_files_only,
            **config_kwargs,
        )

    return {
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "tokenizer_2": tokenizer_2,
        "text_encoder_2": text_encoder_2,
    }


def create_diffusers_unet_model_from_ldm(
    class_name,
    original_config,
    checkpoint,
    num_in_channels=None,
    image_size=None,
):
    config = yaml.safe_load(open(original_config, "r"))

    config = UNet2DConditionModel.load_config(original_config)

    if num_in_channels is not None:
        config.in_channels = num_in_channels

    if image_size is not None:
        config.sample_size = image_size

    unet = UNet2DConditionModel(**config)

    checkpoint = checkpoint

    checkpoint = {
        k.replace("model.diffusion_model.", ""): v
        for k, v in checkpoint.items()
        if "model.diffusion_model." in k and "model" not in k
    }

    unet.load_state_dict(checkpoint)

    return {"unet": unet}


def create_diffusers_vae_model_from_ldm(
    class_name,
    original_config,
    checkpoint,
    image_size=None,
):
    config = yaml.safe_load(open(original_config, "r"))

    config = AutoencoderKL.load_config(original_config)

    if image_size is not None:
        config.sample_size = image_size

    vae = AutoencoderKL(**config)

    checkpoint = checkpoint

    checkpoint = {
        k.replace("first_stage_model.", ""): v
        for k, v in checkpoint.items()
        if "first_stage_model." in k and "model" not in k
    }

    vae.load_state_dict(checkpoint)

    return {"vae": vae}


def create_diffusers_controlnet_model_from_ldm(
    class_name,
    original_config,
    checkpoint,
    upcast_attention=False,
    image_size=None,
):
    config = yaml.safe_load(open(original_config, "r"))

    config = UNet2DConditionModel.load_config(original_config)

    if image_size is not None:
        config.sample_size = image_size

    unet = UNet2DConditionModel(**config)

    checkpoint = checkpoint

    checkpoint = {
        k.replace("model.diffusion_model.", ""): v
        for k, v in checkpoint.items()
        if "model.diffusion_model." in k and "model" not in k
    }

    unet.load_state_dict(checkpoint)

    return {"controlnet": unet}


def create_scheduler_from_ldm(
    pipeline_class_name,
    original_config,
    checkpoint,
    prediction_type=None,
    scheduler_type="ddim",
    model_type=None,
):
    scheduler_config = get_default_scheduler_config()
    model_type = infer_model_type(original_config, model_type=model_type)

    global_step = checkpoint["global_step"] if "global_step" in checkpoint else None

    num_train_timesteps = getattr(original_config["model"]["params"], "timesteps", None) or 1000
    scheduler_config["num_train_timesteps"] = num_train_timesteps

    if (
        "parameterization" in original_config["model"]["params"]
        and original_config["model"]["params"]["parameterization"] == "v"
    ):
        if prediction_type is None:
            # NOTE: For stable diffusion 2 base it is recommended to pass `prediction_type=="epsilon"`
            # as it relies on a brittle global step parameter here
            prediction_type = "epsilon" if global_step == 875000 else "v_prediction"

    else:
        prediction_type = prediction_type or "epsilon"

    scheduler_config["prediction_type"] = prediction_type

    if model_type in ["SDXL", "SDXL-Refiner"]:
        scheduler_type = "euler"

    else:
        beta_start = original_config["model"]["params"].get("linear_start", 0.02)
        beta_end = original_config["model"]["params"].get("linear_end", 0.085)
        scheduler_config["beta_start"] = beta_start
        scheduler_config["beta_end"] = beta_end
        scheduler_config["beta_schedule"] = "scaled_linear"
        scheduler_config["clip_sample"] = False
        scheduler_config["set_alpha_to_one"] = False

        scheduler_type = "ddim"

    if scheduler_type == "pndm":
        scheduler_config["skip_prk_steps"] = True
        scheduler = PNDMScheduler.from_config(scheduler_config)

    elif scheduler_type == "lms":
        scheduler = LMSDiscreteScheduler.from_config(scheduler_config)

    elif scheduler_type == "heun":
        scheduler = HeunDiscreteScheduler.from_config(scheduler_config)

    elif scheduler_type == "euler":
        scheduler = EulerDiscreteScheduler.from_config(scheduler_config)

    elif scheduler_type == "euler-ancestral":
        scheduler = EulerAncestralDiscreteScheduler.from_config(scheduler_config)

    elif scheduler_type == "dpm":
        scheduler = DPMSolverMultistepScheduler.from_config(scheduler_config)

    elif scheduler_type == "ddim":
        scheduler = DDIMScheduler.from_config(scheduler_config)

    else:
        raise ValueError(f"Scheduler of type {scheduler_type} doesn't exist!")

    if pipeline_class_name == "StableDiffusionUpscalePipeline":
        scheduler = DDIMScheduler.from_pretrained("stabilityai/stable-diffusion-x4-upscaler", subfolder="scheduler")
        low_res_scheduler = DDPMScheduler.from_pretrained(
            "stabilityai/stable-diffusion-x4-upscaler", subfolder="low_res_scheduler"
        )

        return {
            "scheduler": scheduler,
            "low_res_scheduler": low_res_scheduler,
        }

    return {"scheduler": scheduler}


def create_scheduler_from_ldm(
    pipeline_class_name,
    original_config,
    checkpoint,
    prediction_type=None,
    scheduler_type="ddim",
    model_type=None,
):
    scheduler_config = get_default_scheduler_config()
    model_type = infer_model_type(original_config, model_type=model_type)

    global_step = checkpoint["global_step"] if "global_step" in checkpoint else None

    num_train_timesteps = getattr(original_config["model"]["params"], "timesteps", None) or 1000
    scheduler_config["num_train_timesteps"] = num_train_timesteps

    if (
        "parameterization" in original_config["model"]["params"]
        and original_config["model"]["params"]["parameterization"] == "v"
    ):
        if prediction_type is None:
            # NOTE: For stable diffusion 2 base it is recommended to pass `prediction_type=="epsilon"`
