# Community Examples

> **For more information about community pipelines, please have a look at [this issue](https://github.com/huggingface/diffusers/issues/841).**

**Community** examples consist of both inference and training examples that have been added by the community.
Please have a look at the following table to get an overview of all community examples. Click on the **Code Example** to get a copy-and-paste ready code example that you can try out.
If a community doesn't work as expected, please open an issue and ping the author on it.

| Example                                                                                                                               | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | Code Example                                                                              | Colab                                                                                                                                                                                                              |                                                        Author |
|:--------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------:|
| Marigold Monocular Depth Estimation                                                                                                   | A universal monocular depth estimator, utilizing Stable Diffusion, delivering sharp predictions in the wild. (See the [project page](https://marigoldmonodepth.github.io) and [full codebase](https://github.com/prs-eth/marigold) for more details.)                                                                                                                                                                                                                                                        | [Marigold Depth Estimation](#marigold-depth-estimation)                                   | [![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-yellow)](https://huggingface.co/spaces/toshas/marigold) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12G8reD13DdpMie5ZQlaFNo2WCGeNUH-u?usp=sharing) | [Bingxin Ke](https://github.com/markkua) and [Anton Obukhov](https://github.com/toshas) |
| LLM-grounded Diffusion (LMD+)                                                                                                         | LMD greatly improves the prompt following ability of text-to-image generation models by introducing an LLM as a front-end prompt parser and layout planner. [Project page.](https://llm-grounded-diffusion.
# ... [truncated] ...
         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            res.append(image)

    vidcap.release()
    return res

input_video_path = 'path/to/video'
input_interval = 10
frames = video_to_frame(
    input_video_path, input_interval)

control_frames = []
# get canny image
for frame in frames:
    np_image = cv2.Canny(frame, 50, 100)
    np_image = np_image[:, :, None]
    np_image = np.concatenate([np_image, np_image, np_image], axis=2)
    canny_image = Image.fromarray(np_image)
    control_frames.append(canny_image)

# You can use any ControlNet here
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny").to('cuda')

# You can use any fintuned SD here
pipe = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, custom_pipeline='./forked/diffusers/examples/community/ip_adapter_face_id.py').to('cuda')

# Optional: you can download vae-ft-mse-840000-ema-pruned.ckpt to enhance the results
# pipe.vae = AutoencoderKL.from_single_file(
#     "path/to/vae-ft-mse-840000-ema-pruned.ckpt").to('cuda')

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

generator = torch.manual_seed(0)
frames = [Image.fromarray(frame) for frame in frames]
output_frames = pipe(
    "a beautiful woman in CG style, best quality, extremely detailed",

    frames,
    control_frames,
    num_inference_steps=20,
    strength=0.75,
    controlnet_conditioning_scale=0.7,
    generator=generator,
    warp_start=0.0,
    warp_end=0.1,
    mask_start=0.5,
    mask_end=0.8,
    mask_strength=0.5,
    negative_prompt='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
).frames

export_to_video(
    output_frames, "/path/to/video.mp4", 5)
```

### IP Adapter Face ID
IP Adapter FaceID is an experimental IP Adapter model that uses image embeddings generated by `insightface`, so no image encoder needs to be loaded.
You need to install `insightface` and all its requirements to use this model.
You must pass the image embedding tensor as `image_embeds` to the StableDiffusionPipeline instead of `ip_adapter_image`.
You have to disable PEFT BACKEND in order to load weights.

```py
import diffusers
diffusers.utils.USE_PEFT_BACKEND = False
import torch
from diffusers.utils import load_image
import cv2
import numpy as np
from diffusers import DiffusionPipeline, AutoencoderKL, DDIMScheduler
from insightface.app import FaceAnalysis


noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(dtype=torch.float16)
pipeline = DiffusionPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V4.0_noVAE",
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    vae=vae,
    custom_pipeline='./forked/diffusers/examples/community/ip_adapter_face_id.py'
)
pipeline.load_ip_adapter_face_id("h94/IP-Adapter-FaceID", "ip-adapter-faceid_sd15.bin")
pipeline.to("cuda")

generator = torch.Generator(device="cpu").manual_seed(42)
num_images=2

image = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/ai_face2.png")

app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
faces = app.get(image)
image = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
images = pipeline(
    prompt="A photo of a girl wearing a black dress, holding red roses in hand, upper body, behind is the Eiffel Tower",
    image_embeds=image,
    negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality", 
    num_inference_steps=20, num_images_per_prompt=num_images, width=512, height=704, 
    generator=generator
).images

for i in range(num_images):
    images[i].save(f"c{i}.png")
```

