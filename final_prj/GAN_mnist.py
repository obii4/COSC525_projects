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

# MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

image_size = x_train.shape[1]
original_dim = image_size * image_size
x_train= x_train.astype('float32')/ 255.0
x_test = x_test.astype('float32')/ 255.0

x_train = np.expand_dims(x_train, axis=3)
print(x_train.shape)
# network parameters
batch_size = 256

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
dim=7

generator = Sequential()
generator.add(Dense(dim*dim*depth, input_dim=100))
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

discriminator=Sequential()
#discriminator.add(Input(shape=(28,28,1), name='image'))
discriminator.add(Conv2D(depth,5,strides=2,padding='same',input_shape=(28,28,1), activation=LeakyReLU(alpha=0.2)))
discriminator.add(Dropout(dropout))

discriminator.add(Conv2D(depth*2,5,strides=2,padding='same',input_shape=(28,28,1), activation=LeakyReLU(alpha=0.2)))
discriminator.add(Dropout(dropout))

discriminator.add(Conv2D(depth*4,5,strides=2,padding='same',input_shape=(28,28,1), activation=LeakyReLU(alpha=0.2)))
discriminator.add(Dropout(dropout))

discriminator.add(Conv2D(depth*8,5,strides=1,padding='same',input_shape=(28,28,1), activation=LeakyReLU(alpha=0.2)))
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
for i in range(1000):
    #sample random images from training set
    idx = np.random.randint(0, x_train.shape[0], batch_size)
    print(f"xtrain shape {x_train.shape}")
    imgs = x_train[idx]
    #sample random noise and put it through generator
    noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
    images_fake = generator.predict(noise)
    print(f"Noise shape: {noise.shape}")

    print(f"True Images Shape: {imgs.shape}")
    print(f"Fake Images Shape: {images_fake.shape}")
    #create training set minibatch
    x=np.concatenate((imgs,images_fake))
    print(x.shape)
    #train discriminator
    d_loss=discriminator.train_on_batch(x,y)
    #train generator (entire GAN)
    noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
    g_loss = GAN.train_on_batch(noise, valid)
    print('{} d_loss: {}, g_loss{}'.format(i,d_loss,g_loss))



    #plotting generated images

print("Trained. Plotting")


n=10
plt.figure(figsize=(20, 2))
for i in range(1,n):
    ax = plt.subplot(1, n, i)
    noise = np.random.uniform(-1.0, 1.0, size=[1,100])
    new_img=generator.predict(noise)
    image = new_img[0, :, :, :]
    image = np.reshape(image, [28, 28])
    plt.imshow(image, cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

print("End")