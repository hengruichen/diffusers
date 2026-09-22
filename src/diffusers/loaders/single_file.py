# Copyright 2023 The HuggingFace Team. All rights reserved.
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
import os
import re

from huggingface_hub.utils import validate_hf_hub_args
from transformers import AutoFeatureExtractor

from ..models.modeling_utils import load_state_dict
from ..utils import (
    logging,
)
from ..utils.hub_utils import _get_model_file
from .single_file_utils import (
    create_diffusers_controlnet_model_from_ldm,
    create_diffusers_unet_model_from_ldm,
    create_diffusers_vae_model_from_ldm,
    create_scheduler_from_ldm,
    create_text_encoders_and_tokenizers_from_ldm,
    fetch_original_config,
    infer_model_type,
)


logger = logging.get_logger(__name__)


VALID_URL_PREFIXES = ["https://huggingface.co/", "huggingface.co/", "hf.co/", "https://hf.co/"]
# Pipelines that support the SDXL Refiner checkpoint
REFINER_PIPELINES = [
    "StableDiffusionXLImg2ImgPipeline",
    "StableDiffusionXLInpaintPipeline",
    "StableDiffusionXLControlNetImg2ImgPipeline",
]


def _extract_repo_id_and_weights_name(pretrained_model_name_or_path):
    pattern = r"([^/]+)/([^/]+)/(?:blob/main/)?(.+)"
    weights_name = None
    repo_id = (None,)
    for prefix in VALID_URL_PREFIXES:
        pretrained_model_name_or_path = pretrained_model_name_or_path.replace(prefix, "")
    match = re.match(pattern, pretrained_model_name_or_path)
    if not match:
        return repo_id, weights_name

    repo_id = f"{match.group(1)}/{match.group(2)}"
    weights_name = match.group(3)

    return repo_id, weights_name


def build_sub_model_components(
    pipeline_components,
    pipeline_class_name,
    component_name,
    original_config,
    checkpoint,
    checkpoint_path_or_dict,
    local_files_only=False,
    load_safety_checker=False,
    **kwargs,
):
    if component_name in pipeline_components:
        return {}

    model_type = kwargs.get("model_type", None)
    image_size = kwargs.pop("image_size", None)

    if component_name == "unet":
        num_in_channels = kwargs.pop("num_in_channels", None)
        unet_components = create_diffusers_unet_model_from_ldm(
            pipeline_class_name, original_config, checkpoint, num_in_channels=num_in_channels, image_size=image_size
        )
        return unet_components

    if component_name == "vae":
        vae_components = create_diffusers_vae_model_from_ldm(
            pipeline_class_name, original_config, checkpoint, image_size
        )
        return vae_components

    if component_name == "scheduler":
        scheduler_type = kwargs.get("scheduler_type", "ddim")
        prediction_type = kwargs.get("prediction_type", None)

        scheduler_components = create_scheduler_from_ldm(
            pipeline_class_name,
            original_config,
            checkpoint,
            scheduler_type=scheduler_type,
            prediction_type=prediction_type,
            model_type=model_type,
        )

        return scheduler_components

    if component_name in ["text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2"]:
        text_encoder_components = create_text_encoders_and_tokenizers_from_ldm(
            original_config,
            checkpoint,
            model_type=model_type,
            local_files_only=local_files_only,
        )
        return text_encoder_components

    if component_name == "safety_checker":
        if load_safety_checker:
            from ..pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

            safety_ch
# ... [truncated] ...
l_config_file = kwargs.pop("original_config_file", None)
        resume_download = kwargs.pop("resume_download", False)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        token = kwargs.pop("token", None)
        cache_dir = kwargs.pop("cache_dir", None)
        local_files_only = kwargs.pop("local_files_only", None)
        revision = kwargs.pop("revision", None)
        torch_dtype = kwargs.pop("torch_dtype", None)
        use_safetensors = kwargs.pop("use_safetensors", True)

        class_name = cls.__name__
        file_extension = pretrained_model_link_or_path.rsplit(".", 1)[-1]
        from_safetensors = file_extension == "safetensors"

        if from_safetensors and use_safetensors is False:
            raise ValueError("Make sure to install `safetensors` with `pip install safetensors`.")

        if os.path.isfile(pretrained_model_link_or_path):
            checkpoint = load_state_dict(pretrained_model_link_or_path)
        else:
            repo_id, weights_name = _extract_repo_id_and_weights_name(pretrained_model_link_or_path)
            checkpoint_path = _get_model_file(
                repo_id,
                weights_name=weights_name,
                force_download=force_download,
                cache_dir=cache_dir,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
            )
            checkpoint = load_state_dict(checkpoint_path)

        # some checkpoints contain the model state dict under a "state_dict" key
        while "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        original_config = fetch_original_config(class_name, checkpoint, original_config_file)

        if class_name == "AutoencoderKL":
            image_size = kwargs.pop("image_size", None)
            component = create_diffusers_vae_model_from_ldm(
                class_name, original_config, checkpoint, image_size=image_size
            )
            return component["vae"]

        if class_name == "ControlNetModel":
            upcast_attention = kwargs.pop("upcast_attention", False)
            image_size = kwargs.pop("image_size", None)

            component = create_diffusers_controlnet_model_from_ldm(
                class_name, original_config, checkpoint, upcast_attention=upcast_attention, image_size=image_size
            )
            return component["controlnet"]

        from ..pipelines.pipeline_utils import _get_pipeline_class

        pipeline_class = _get_pipeline_class(
            cls,
            config=None,
            cache_dir=cache_dir,
        )

        expected_modules, optional_kwargs = cls._get_signature_keys(pipeline_class)
        passed_class_obj = {k: kwargs.pop(k) for k in expected_modules if k in kwargs}
        passed_pipe_kwargs = {k: kwargs.pop(k) for k in optional_kwargs if k in kwargs}

        init_kwargs = {}
        for name in expected_modules:
            if name in passed_class_obj:
                init_kwargs[name] = passed_class_obj[name]
            else:
                components = build_sub_model_components(
                    init_kwargs,
                    class_name,
                    name,
                    original_config,
                    checkpoint,
                    pretrained_model_link_or_path,
                    **kwargs,
                )
                if not components:
                    continue
                init_kwargs.update(components)

        additional_components = set_additional_components(class_name, original_config, **kwargs)
        if additional_components:
            init_kwargs.update(additional_components)

        init_kwargs.update(passed_pipe_kwargs)
        pipe = pipeline_class(**init_kwargs)

        if torch_dtype is not None:
            pipe.to(dtype=torch_dtype)

        return pipe

