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
import tempfile
import unittest

import numpy as np
import torch
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipelineLegacy,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
    StableDiffusionUpscalePipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import (
    LoraLoaderMixin,
    TextualInversionLoaderMixin,
    StableDiffusionXLLoraLoaderMixin,
)
from diffusers.models import ControlNetModel
from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    LoRAAttnProcessor2_0,
    LoRAXFormersAttnProcessor,
    XFormersAttnProcessor,
)
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    load_image,
    load_video,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
