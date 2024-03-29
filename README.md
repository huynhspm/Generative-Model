# **Diffusion-Models**

## **1. Introduction**
Diffusion model is a type of generative model. Its approach is different from GAN, VAE and Flow-based models. In my repository, I re-setup diffusion model from scratch to do some experiments:
* Diffusion Model: Training with simple loss
* Inference with DDPM and  DDIM
* Using (label, image, text) as condition for diffusion model
* Latent diffusion: Image space to latent space with VAE
* Stable diffusion: Latent + Condition Diffusion
* Classifier-free guidance

## **2. Set Up**
  ### **Clone the repository**
    https://github.com/huynhspm/Generative-Model
    
  ### **Install environment packages**
    cd Generative-Model
    conda create -n diffusion python=3.10
    conda activate diffusion 
    pip install -r requirements.txt

  ### **Training Diffusion**
  set-up CUDA_VISIBLE_DEVICES and WANDB_API_KEY before training
  
    export CUDA_VISIBLE_DEVICES=0,1
    export WANDB_API_KEY=???
    python src/train.py experiment=diffusion_mnist trainer.devices=2
  ### **Training VAE**
  set-up CUDA_VISIBLE_DEVICES and WANDB_API_KEY before training
  
    export CUDA_VISIBLE_DEVICES=0
    export WANDB_API_KEY=???
    python src/train.py experiment=vq_vae_celeba trainer.devices=1
  ### **Inference**
    Inference: 
    
## **3. Diffusion Model**
### **3.1. Dataset**
  - MNIST DATASET
  - FASHION-MNIST DATASET
  - CIFAR10 DATASET
  - [GENDER DATASET](https://www.kaggle.com/datasets/yasserhessein/gender-dataset)
  - [CELEBA DATASET](https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256)
  - [AFHQ DATASET](https://www.kaggle.com/datasets/andrewmvd/animal-faces) 
  - [FFHQ DATASET](https://www.kaggle.com/datasets/greatgamedota/ffhq-face-data-set)
### **3.2. Attention**
  - Self Attention
  - Cross Attention
  - Spatial Transformer
### **3.3. Backbone**
  - ResNet Block
  - VGG Block
  - DenseNet Block
  - Inception Block
### **3.4 Embedder**
  - Label
  - Time
  - Image
  - Text: not implemented
### **3.5. Sampler**
  - DDPM: Denoising Diffusion Probabilistic Models
  - DDIM: Denoising Diffusion Implicit Models
### **3.6. Model**
  - Unet
  - Unconditional diffusion model
  - Conditional diffusion model (label, image, text - need to implement text embedder model)
  - Variational autoencoder: Vanilla (only work for reconstruction), VQ
  - Latent diffusion model
  - Classifier-free; not work
## **4. RESULTS**
### **4.1. Unconditional Diffusion**
#### **MINST and FASHION-MNIST (32x32)**
![Mnist Generation](results/udm/mnist.png)
![Fashion Generation](results/udm/fashion.jpg)
#### **CIFAR10 (32x32)**
![Cifar10 Generation](results/udm/cifar10.jpg)    
### **4.2. Conditional Diffusion**
#### **MINST and FASHION-MNIST (32x32)**
![Mnist Generation](results/cdm/mnist.jpg)
![Fashion Generation](results/cdm/fashion.jpg)
#### **CIFAR10 (32x32)**
![Cifar10 Generation](results/cdm/cifar10.jpg)
#### **GENDER (64x64)**: 
- Male and Female

![Male Generation](results/cdm/gender/male.jpg)
![Male Generation](results/cdm/gender/male.gif)
![Female Generation](results/cdm/gender/female.jpg)
![Female Generation](results/cdm/gender/female.gif)


#### **CELEBA (64x64)**
- Sketch2Image (Sketch, Fake, Real)
  
![AFHQ Sketch](results/cdm/celeba/sketch.png)
![AFHQ Fake](results/cdm/celeba/fake.png)
![AFHQ Real](results/cdm/celeba/real.png)
### **4.3 DDPM and DDIM**
#### **DDPM (64x64)**
![DDPM Generation](results/udm/gender/ddpm.jpg)
#### **DDIM (64x64)**
![DDIM Generation](results/udm/gender/ddim.jpg)
### **4.4 DIFFUSION INTERPOLATION (64x64)**
![Interpolation Generation](results/udm/gender/interpolation.jpg)
### **4.5 VAE RECONSTRUCTION**
#### **CIFAR10**
![Cifar10 Reconstruction](results/vae/cifar10/reconstruction.jpg)
#### **AFHQ**
![AFHQ Reconstruction](results/vae/afhq/reconstruction.jpg)
#### **GENDER**
![Gender Reconstruction](results/vae/gender/reconstruction.jpg)
#### **CELEBA**
![Celeba Reconstruction](results/cae/../vae/celeba/reconstruction.jpg)
### **4.5 VAE INTERPOLATION**
#### **CIFAR10 (32x32)**
![Cifar10 Interpolation](results/vae/cifar10/interpolation.jpg)
#### **AFHQ (64x64)**
![AFHQ Interpolation](results/vae/afhq/interpolation.jpg)
#### **CELEBA (128x128)**
![Celeba Interpolation](results/vae/celeba/interpolationion.jpg)
### **4.6 Latent Diffusion**
#### **GENDER (128x128)**
![Gender Generation](results/ldm/gender.png)
#### **AFHQ (256x256)**
![AFHQ Generation](results/ldm/afhq.png)
