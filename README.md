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

## 2) Discriminative Model:
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