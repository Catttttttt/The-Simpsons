# Style-specific Colorization
CS182 Final Project

*Abstract:*

In recent years, the process of automatic image colorization has become of great interest. Image colorization is the process of restoring color to black-and-white or grayscale images using computational techniques. Several of the most common applications of image colorization include restoration of aged or degraded images. Another application of image colorization is for the task of filling colors into black and white cartoons. In the comic and 2d animation workflow, coloring in black and white drawings take a significant amount of time. Automatic image colorization is a potential tool for increasing production efficiency in this industry. However, existing learning-based image colorization models have been trained mostly on real-world image and are not suitable for cartoons. However, most comics or animations have a unique style and color scheme. Therefore, for cartoon image colorization, producing style-specific colorized images becomes extremely important.

We aim to address the limitations of existing models that are primarily trained on real-world images by fine-tuning them. We mainly focus on fine-tuning a generative model for the colorization of greyscale images from *The Simpsons* cartoons. In addition, we will also provide a comparison between the performance of generative and discriminative models.

This repository contains the following folders corresponding to the results with different types of models:
__1) DCGAN__ - based on code from ...
__2) Diffusion__ - Finetuning InstructPix2Pix
__3) Discriminative__
*Baseline model: Colorful Image Colorization; Richard Zhang, Phillip Isola, Alexei A. Efros. In ECCV, 2016.*

## 1) Generative Model:
*Baseline model: Image Colorization using Generative Adversarial Networks; Kamyar Nazeri, Eric Ng, and Mehran Ebrahimi. In Perales, F., Kittler, J. (eds) Articulated Motion and Deformable Objects. AMDO 2018.*

To replicate the results, please run the following command first:
```
cd Colorizing-with-GANs
mkdir checkpoints
```
Download all the datasets [here](https://drive.google.com/drive/folders/1N6WZpio230vRFiOqj0zWXf-tzfenbhpd?usp=sharing) and place them in the `Colorizing-with-GANs` folder.

To fine-tune the model pretrained on the Places365 dataset, please download the pretrained weights of the Places365 model [here](https://drive.google.com/drive/folders/1vzbx5qXJEJP5KkMkyF_5iXl6fQvud7gO?usp=sharing), place them in the `checkpoints` folder, and run the following commands:
```
python finetune.py \
    --checkpoints-path ./checkpoints/simpsons_new \
    --dataset-path ./dataset/simpsons_train_256 \
    --dataset simpsons \
    --batch-size 40 \
    --epochs 100 \
    --lr 5e-4 \
    --save-interval 80 \
    --lr-decay-rate 0.5 \
    --lr-decay-steps 50 \
    --validate 1 \
    --validate-interval 1 \
    --log 1 \
    --sample-interval 10 
```
To produce outputs, please either fine-tune the model first or download our pretrained weights [here](https://drive.google.com/drive/folders/11Qq_I6wQbfd_NxAtB18XzGbnu0sAerYK?usp=sharing) and place them in the `checkpoints` folder. Then, run the following commands: 
- If you fine-tuned the model: 
```
python test.py \
  --checkpoints-path ./checkpoints/simpsons_new \
  --test-input ./dataset/simpsons_test_256 \
  --test-output ./output/test 
```
- If you use our pretrained weights: 
```
python test.py \
  --checkpoints-path ./checkpoints/simpsons11 \
  --test-input ./dataset/simpsons_test_256 \
  --test-output ./output/test 
```
<!-- 
2) *TODO: Instructions for Diffusion replication*
Work on finetuning this model is ongoing and further instructions and results will be added soon. -->

## 2) Diffusion Model: Finetuning InstructPix2Pix
_[Performing the fine-tuning requires a cuda GPU due to the StableDiffusion requirements]_

The pre-trained weights and training/validation/test dataset are provided at the following links:
1) Pre-trained weights: https://huggingface.co/wid4soe/ip2p-simpsons
2) Dataset: https://huggingface.co/datasets/wid4soe/182_simpsons_train

Below are the instructions to finetune and run the model. 
We have also included a sample notebook called finetune_and_run.ipynb in the repo under the folder 4_instructpix2pix.

Clone the repo and change into the correct directory:
```
git clone https://github.com/Catttttttt/The-Simpsons.git
cd ./The-Simpsons/4_instructpix2pix/finetuning
```
Install the required requirements:
```
pip install -r requirements.txt
```
Log into Hugging Face using a WRITE access token from your Hugging Face account:
```
huggingface-cli login [HUGGINGFACE_TOKEN]
```
To perform the finetuning, run the following code, substituting the OUTPUT_DIR with the desired output directory for your model. If the environment variable does not work, you can paste the string in the corresponding argument.
```
export OUTPUT_DIR="<output_dir>"

accelerate launch --mixed_precision="fp16" ft_instruct_pix2pix.py \
  --pretrained_model_name_or_path="timbrooks/instruct-pix2pix" \
  --dataset_name="wid4soe/182_simpsons_train" \
  --use_ema \
  --resolution=256 --random_flip \
  --train_batch_size=2 --gradient_accumulation_steps=4 --gradient_checkpointing \
  --max_train_steps=1000 \
  --checkpointing_steps=300 --checkpoints_total_limit=1 \
  --learning_rate=5e-05 --lr_warmup_steps=0 \
  --mixed_precision=fp16 \
  --val_image_url="https://datasets-server.huggingface.co/assets/wid4soe/182_3/--/f33bb01840d693bfdb02485894413633d62531a1/--/default/train/0/original_image/image.jpg" \
  --validation_prompt="Color in the style of the Simpsons. a man in a suit and tie, with a woman in a suit and tie" \
  --seed=42 \
  --output_dir=$OUTPUT_DIR \
  --report_to=tensorboard \
  --push_to_hub
```
Our finetuned model is hosted on HuggingFace at the following link: https://huggingface.co/wid4soe/ip2p-simpsons

To run the finetuned model on your own images (requires CUDA due to the StableDiffusion pipeline requirements):
Install the required dependencies:
```pip install diffusers accelerate safetensors transformers```
Login to HuggingFace with an access token:
```huggingface-cli login [HUGGINGFACE_TOKEN]```

Initialize the pipeline. You may also replace model_id with the address of your local model.
```
import pandas as pd
from PIL import Image, ImageEnhance
from io import BytesIO
from IPython.display import display
import random
import PIL
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

model_id = "wid4soe/ip2p-simpsons"
pipe_simp = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
pipe_simp.to("cuda")
pipe_simp.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe_simp.scheduler.config)
```
Generate images using the finetuned model. Replace `{INPUT_IMAGE}` and `{OUTPUT_PATH}` with where you would like to load your image from and save the colorized image to.
```
from PIL import ImageOps
img=Image.open('{INPUT_IMAGE}')
text = "Color in the style of the Simpsons: {YOUR DESCRIPTION}"
prompt = "Color in the style of the Simpsons: " + "Your instruction"
img = ImageOps.grayscale(img)
img = img.convert('RGB')
images = pipe_simp(prompt, image=img, num_inference_steps=10, image_guidance_scale=1).images
img.save("/content/182_final_proj/test_ip2p/mono/img_mono.jpg")    # save the grayscale input image
images[0].save("{OUTPUT_PATH}/img_color.jpg")                         # save the colorized output image
```

## 3) Discriminative Model:
*Baseline model: Colorful Image Colorization; Richard Zhang, Phillip Isola, Alexei A. Efros. In ECCV, 2016.*

Image colorization results using only the pretrained model are located in ./3_discriminative/colorful_baseline_results
To replicate these results, please run the following commands:
```
cd 3_discriminative/colorful_demo
pip install -r requirements.txt
```
Next, run the following command for each INPUT_IMAGE in the folder ./3_discriminative/colorful_baseline_results
```
python demo_release.py -i ../colorful_baseline_results/input_[INPUT_IMAGE].jpg
```
Work on finetuning this model is ongoing and further instructions and results will be added soon.
