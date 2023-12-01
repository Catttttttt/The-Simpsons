# Style-specific Colorization
CS182 Final Project

*Abstract:*

In recent years, the process of automatic image colorization has become of great interest. Image colorization is the process of restoring color to black-and-white or grayscale images using computational techniques. Several of the most common applications of image colorization include restoration of aged or degraded images. Another application of image colorization is for the task of filling colors into black and white cartoons. In the comic and 2d animation workflow, coloring in black and white drawings take a significant amount of time. Automatic image colorization is a potential tool for increasing production efficiency in this industry. However, existing learning-based image colorization models have been trained mostly on real-world image and are not suitable for cartoons. However, most comics or animations have a unique style and color scheme. Therefore, for cartoon image colorization, producing style-specific colorized images becomes extremely important.

We aim to address the limitations of existing models that are primarily trained on real-world images by fine-tuning them. We mainly focus on fine-tuning a generative model for the colorization of greyscale images from *The Simpsons* cartoons. In addition, we will also provide a comparison between the performance of generative and discriminative models.

This repository contains the following folders corresponding to the results with different types of models:
__1) DCGAN__ - based on code from ...
__2) Diffusion__ - based on the Colorized-diffusion repository
__3) Discriminative__
*Baseline model: Colorful Image Colorization; Richard Zhang, Phillip Isola, Alexei A. Efros. In ECCV, 2016.*

1) *TODO: Instructions for DCGAN*
Work on finetuning this model is ongoing and further instructions and results will be added soon.

2) *TODO: Instructions for Diffusion replication*
Work on finetuning this model is ongoing and further instructions and results will be added soon.

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