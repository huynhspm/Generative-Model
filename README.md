# **Diffusion-Models**

## **Introduction**
Diffusion model is a type of generative model. Its approach is different from GAN, VAE and Flow-based models. In my repository, I re-setup the model to do some experiments.I'm looking forward to utilizing it, exploring something new and creating something useful from diffusion model

## **Set Up**
  ### 1. Clone the repository
    
    https://github.com/huynhspm/Diffusion-Model
    
  ### 2. Install packages

    cd Diffusion-Model
    pip install -r requirements

  ### 3. Run
  I don't upload the checkpoints so you need to train before inference
  
  Before training, go to "config/train.yaml" to change parameters

  Before inference, go to "config/inference.yaml" to change parameters

    Train: python src/train.py 
    Inference: python src/inference.py

## **Model**
### **Dataset**
  - Mnist dataset
  - Mnist fashion dataset
  - Cifar10 dataset
  - [Gender dataset](https://www.kaggle.com/datasets/yasserhessein/gender-dataset)
### **Version 1**
- Using basic Unet to learn the noises added to image
#### Generated Images
#### MINST
![MNIST Generation](/outputs/inference/MNIST/version_1.gif)
#### FASHION
![FASHION Generation](/outputs/inference/FASHION/version_1.gif)
#### CIFAR
![CIFAR Generation](/outputs/inference/CIFAR/version_1.gif
)    
#### GENDER
![GENDER Generation](/outputs/inference/GENDER/version_1.gif)
### **Version 2**
- Backbone: Resnet
- Attention: Self_attention_wrapper
- Time embedding: not only embed by (sin, cos) but also using convolution layer
- #### FASHION
![Fashion Generation](/outputs/inference/FASHION/version_2.gif)
#### GENDER
![GENDER Generation](/outputs/inference/GENDER/version_2.gif)