#architecture borrowed from: https://github.com/roatienza/Deep-Learning-Experiments
#importing relevant libraries

from IPython.display import Image


from tensorflow.keras.layers import UpSampling2D, Lambda, Input, Dense, Reshape, Conv2DTranspose, BatchNormalization, ReLU, Activation, Conv2D, Flatten, Dropout, LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

import os
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm
import matplotlib.pyplot as plt


# Join various path components
tumors_images_path = os.path.join(os.getcwd(), "data", "tumor", "resized")
notumor_images_path = os.path.join(os.getcwd(), "data", "notumor")
training_images_path = os.path.join(os.getcwd(), "data", "training", "images")
training_masks_path = os.path.join(os.getcwd(), "data", "training", "masks")


class ImageLoader:
    '''
    Takes path to image folder
    Loads images from folder and returns 4D numpy array (num_images, pixel_width, pixel_heights, channels)
    '''

    def __init__(self, image_path):
        self.image_path = image_path

    def load_image(self, path):
        ''' Loads single RGB image as np array and scale data to the range of [0, 1] '''
        color_image = Image.open(path)
        gray_image = np.array(ImageOps.grayscale(color_image), dtype=float) / 255
        return gray_image


    # load images, masks, and (if applicable) roi's
    def load_data(self, type, paths=None):
        ''' Loads all images '''

        # Create list of paths to each image file in the input path
        list_of_files = []
        for root, dirs, files in os.walk(self.image_path):
	        for file_ in files:
		        list_of_files.append(os.path.join(root, file_))

        # load each image in the list of file paths
        images = []
        for file_name in list_of_files:
            images.append(self.load_image(file_name))


        # shapes each image to correct size
        if type == 'CNN':
            ''' loads 2d images '''
            images_all = images
            images_all_out = np.asarray(images_all)

            images_all_out = images_all_out.reshape(images_all_out.shape[0], 180, 180, 1)

        else:
            ''' loads 1d images '''
            images_all = []
            for i in range(len(list_of_files)):
                t = images[i].flatten()
                images_all.append(t)

            images_all_out = np.asarray(images_all)  # each element of the array is (1024,)

        if 'train' in self.image_path:
            print(f'Loaded {len(list_of_files)} images for the training set!')
        else:
            print(f'Loaded {len(list_of_files)} images for the validation set!')


        return images_all_out


# MNIST dataset
#(x_train, y_train), (x_test, y_test) = mnist.load_data()


image_loader_ = ImageLoader(tumors_images_path)
#tumor_images = image_loader_.load_data('CNN')
x_train = image_loader_.load_data('CNN')

image_size = x_train.shape[1]
original_dim = image_size * image_size
#x_train= x_train.astype('float32')/ 255.0
#x_test = x_test.astype('float32')/ 255.0

#x_train = np.expand_dims(x_train, axis=3)
print(x_train.shape)
# network parameters
batch_size = 10

n=10
plt.figure(figsize=(20, 2))
for i in range(1,n):
    ax = plt.subplot(1, n, i)
    new_img=x_train[i+10,:,:,0]
    plt.imshow(new_img)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()



#Setting up generator:
# image from: https://towardsdatascience.com/gan-by-example-using-keras-on-tensorflow-backend-1a6d515a60d0

dropout = 0.4
depth=64
#dim=7  # mnist dim size 28/4=7 so 180/4 = 45 will work???
dim=45
input_dim_ = 50

generator = Sequential()
generator.add(Dense(dim*dim*depth, input_dim=input_dim_))
generator.add(BatchNormalization(momentum=0.9))
generator.add(ReLU())
generator.add(Reshape((dim,dim,depth)))
generator.add(Dropout(dropout))
generator.add(UpSampling2D())
generator.add(Conv2D(int(depth/2),5,1,padding='same'))
generator.add(BatchNormalization(momentum=0.9))
generator.add(ReLU())
generator.add(UpSampling2D())
generator.add(Conv2D(int(depth/4),5,1,padding='same'))
generator.add(BatchNormalization(momentum=0.9))
generator.add(ReLU())
generator.add(Conv2D(int(depth/8),5,1,padding='same'))
generator.add(BatchNormalization(momentum=0.9))
generator.add(ReLU())
generator.add(Conv2D(1,5,1,padding='same'))
generator.add(Activation('sigmoid'))
generator.summary()

print("Generator set up.")


#setting up discriminator
# image from: https://towardsdatascience.com/gan-by-example-using-keras-on-tensorflow-backend-1a6d515a60d0

dropout = 0.4
depth = 8

#mnist_original_shape = (28,28,1)
brain_original_shape = (180, 180, 1)


discriminator=Sequential()
#discriminator.add(Input(shape=(28,28,1), name='image'))
discriminator.add(Conv2D(depth,5,strides=2,padding='same',input_shape=brain_original_shape, activation=LeakyReLU(alpha=0.2)))
discriminator.add(Dropout(dropout))

discriminator.add(Conv2D(depth*2,5,strides=2,padding='same',input_shape=brain_original_shape, activation=LeakyReLU(alpha=0.2)))
discriminator.add(Dropout(dropout))

discriminator.add(Conv2D(depth*4,5,strides=2,padding='same',input_shape=brain_original_shape, activation=LeakyReLU(alpha=0.2)))
discriminator.add(Dropout(dropout))

discriminator.add(Conv2D(depth*8,5,strides=1,padding='same',input_shape=brain_original_shape, activation=LeakyReLU(alpha=0.2)))
discriminator.add(Dropout(dropout))



discriminator.add(Flatten())
discriminator.add(Dense(1,activation='sigmoid'))
discriminator.summary()

print("Discriminator set up.")

#Optimizer for discriminator (binary cross entropy)
optimizer = optimizers.RMSprop(lr=0.0002, decay=6e-8)
discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
discriminator.summary()

print("Optimizer set up.")

#creating GAN and optimizer for it (also using binary cross entropy)
discriminator.trainable = False
GAN=Sequential()
GAN.add(generator)
GAN.add(discriminator)
optimizer = optimizers.RMSprop(lr=0.0001, decay=3e-8)
GAN.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
GAN.summary()

print("GAN set up.")

#creating ground truth labels for discriminator
valid = np.ones((batch_size))
fake = np.zeros((batch_size))
y=np.concatenate((valid,fake))


print("Training...")

#training procedure
for i in range(500):
    #sample random images from training set
    idx = np.random.randint(0, x_train.shape[0], batch_size)
    #print(f"index: {idx}")
    #print(f"xtrain shape {x_train.shape}")
    imgs = x_train[idx,]
    #sample random noise and put it through generator
    noise = np.random.uniform(-1.0, 1.0, size=[batch_size, input_dim_])
    images_fake = generator.predict(noise)
    #print(f"Noise shape: {noise.shape}")

    #print(f"True Images Shape: {imgs.shape}")
    #print(f"Fake Images Shape: {images_fake.shape}")

    #create training set minibatch
    x=np.concatenate((imgs,images_fake))
    print(x.shape)
    #train discriminator
    d_loss=discriminator.train_on_batch(x,y)
    #train generator (entire GAN)
    noise = np.random.uniform(-1.0, 1.0, size=[batch_size, input_dim_])
    g_loss = GAN.train_on_batch(noise, valid)
    print('{} d_loss: {}, g_loss{}'.format(i,d_loss,g_loss))



    #plotting generated images

print("Trained. Plotting")


n=10
plt.figure(figsize=(20, 2))
for i in range(1,n):
    ax = plt.subplot(1, n, i)
    noise = np.random.uniform(-1.0, 1.0, size=[1,input_dim_])
    new_img=generator.predict(noise)
    image = new_img[0, :, :, :]
    image = np.reshape(image, [180, 180])
    plt.imshow(image, cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

print("End")