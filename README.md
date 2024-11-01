# Image_Generation_Using_GAN


This project implements a Generative Adversarial Network (GAN) using TensorFlow and Keras to generate images of cats and dogs from the CIFAR-10 dataset. The GAN architecture consists of a generator that creates images from random noise and a discriminator that distinguishes between real and generated images.


## Getting Started

To run this project, you'll need to have Python and TensorFlow installed on your machine. You can set up a virtual environment for this project using `venv` or `conda`.

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Mohsin860/Image_Generation_Using_GAN.git
   cd Image_Generation_Using_GAN
Requirements
Python 3.6 or higher
TensorFlow 2.x
Matplotlib
NumPy
Dataset
The CIFAR-10 dataset is used for this project, which contains 60,000 32x32 color images in 10 classes, with 6,000 images per class. This project specifically focuses on generating images of cats (label 3) and dogs (label 5).

Model Architecture
Generator
The generator is built using several layers, including:

Dense layers for initial processing of random noise input.
Batch normalization layers to stabilize training.
Transposed convolutional layers to upsample the feature maps to the desired image size.
Discriminator
The discriminator consists of:

Convolutional layers to extract features from the input images.
Dropout layers for regularization.
A final dense layer with a sigmoid activation to output the probability that the input image is real.
Training
The model is trained over a specified number of epochs. During each epoch, the generator creates fake images from random noise, and the discriminator evaluates both real and fake images. The losses for both the generator and discriminator are calculated and optimized using Adam optimizers.

Hyperparameters
Number of epochs: 150
Batch size: 64
Learning rate: 1e-5
Loss Functions
Binary Crossentropy is used as the loss function for both the generator and discriminator.
Results
During training, images are generated and saved every 10 epochs. Losses for both the generator and discriminator are plotted to visualize the training process.
