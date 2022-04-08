import os, sys, glob, csv, re
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report

#monitor image loading loop for some visual feedback
from tqdm import tqdm

# image processing
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten


# define path to images and label files, the _slim.csv
# file only contains the features of interest
image_train_path = '/Users/chrisobrien/Desktop/p3/train/'
gender_label_train_path = '/Users/chrisobrien/Desktop/grad school/courses/spring 2022/cosc 525/COSC525_projects/prj3/labels/fairface_label_train_gender.csv'
race_label_train_path = '/Users/chrisobrien/Desktop/grad school/courses/spring 2022/cosc 525/COSC525_projects/prj3/labels/fairface_label_train_race.csv'
age_label_train_path = '/Users/chrisobrien/Desktop/grad school/courses/spring 2022/cosc 525/COSC525_projects/prj3/labels/fairface_label_train_age.csv'

image_val_path = '/Users/chrisobrien/Desktop/p3/val/'
gender_label_val_path = '/Users/chrisobrien/Desktop/grad school/courses/spring 2022/cosc 525/COSC525_projects/prj3/labels/fairface_label_val_gender.csv'
race_label_val_path = '/Users/chrisobrien/Desktop/grad school/courses/spring 2022/cosc 525/COSC525_projects/prj3/labels/fairface_label_val_race.csv'
age_label_val_path = '/Users/chrisobrien/Desktop/grad school/courses/spring 2022/cosc 525/COSC525_projects/prj3/labels/fairface_label_val_age.csv'




# load all of file names to a variable
# sort in numerical order so that it matches the label file order ie 0, 1, 2, etc
file_names_train = [os.path.basename(f) for f in glob.glob(image_train_path+'*.jpg')]
file_names_train.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
#file_names_train = file_names_train[0:1000] #uncomment if you want to debug with subset

file_names_val = [os.path.basename(f) for f in glob.glob(image_val_path+'*.jpg')]
file_names_val.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
#file_names_val = file_names_val[0:100] #uncomment if you want to debug with subset



def label_encoder(path, path2):
    '''
    Loads labels and encodes.

    Input:
    path -> train label path
    path2 -> val label path

    Output:
    out1 -> encoded train labels
    out2 -> encoded val labels
    '''

    #train label load
    with open(path, 'r') as f:
        load = csv.reader(f, delimiter=',')
        next(load) # skips header
        text = [text for text in load]
    data_array = np.asarray(text)
    #data_array = data_array[0:1000] #uncomment if you want to debug with subset

    # val label load
    with open(path2, 'r') as f:
        load = csv.reader(f, delimiter=',')
        next(load) # skips header
        text = [text for text in load]
    data_array2 = np.asarray(text)
    #data_array2 = data_array2[0:100] #uncomment if you want to debug with subset

    oHenc = OneHotEncoder()
    oHenc.fit(data_array)
    # print(enc.categories_) #if you want to view the encoded categories
    train_lab = oHenc.transform(data_array).toarray()
    val_lab = oHenc.transform(data_array2).toarray()

    #convert each label to array
    out1 = np.array([np.array(xi) for xi in train_lab]) #train labels encoded
    out2 = np.array([np.array(xi) for xi in val_lab]) #val labels encoded

    return out1, out2


class ImageLoader:
    '''
    Loads images.

    Input:
    file_names -> names of all images
    image_path -> path to location of images

    Output:
    images
    '''

    def __init__(self, file_names, image_path):
        self.file_names = file_names
        self.image_path = image_path

    def __len__(self):
        return len(self.file_names)

    # load RGB image
    def load_image(self, path):
        '''
        Loads images as np array and scale data to the range of [0, 1]
        '''
        image = np.array(Image.open(path), dtype=float) / 255
        return image


    # load images, masks, and (if applicable) roi's
    def load_data(self, type, paths=None):
        '''
        loads images
        '''

        images = []
        file_names = paths if paths is not None else self.file_names
        for file_name in tqdm(file_names):
            image_name = self.image_path + file_name
            images.append(self.load_image(image_name))

        if type == 'CNN':
            '''
            loads 2d images
            '''
            images_all = images
            images_all_out = np.asarray(images_all)

            images_all_out = images_all_out.reshape(images_all_out.shape[0], 32, 32, 1)

        else:
            '''
            loads 1d images
            '''
            images_all = []
            for i in range(len(images)):
                t = images[i].flatten()
                images_all.append(t)

            images_all_out = np.asarray(images_all)  # each element of the array is (1024,)

        if 'train' in self.image_path:
            print(f'Loaded {len(self.file_names)} images for the training set!')
        else:
            print(f'Loaded {len(self.file_names)} images for the validation set!')


        return images_all_out

def task1_model(Xtrain, Ytrain, Xval, Yval):

    '''
    Used for the completion of task 1
    '''

    print("Initializing network...")
    model = Sequential()
    model.add(Dense(1024, input_shape=(1024,), activation="tanh"))
    model.add(Dense(512, activation="sigmoid"))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(Yval.shape[1], activation="softmax"))

    #model.summary()
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001) #SGD(0.00001) #0.0000001
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                metrics=["accuracy"])
    model_out = model.fit(Xtrain, Ytrain, validation_data=(Xval, Yval),
                  epochs=150, batch_size=640, callbacks=[early_stopping])
    best_score = max(model_out.history['val_accuracy'])

    print(f'Max validation acc {best_score}')
    return model_out



def acc_loss_plotting(mod):
    '''
    Takes a trained model and summarizes training through
    plotting:
        1. Accuracy vs Epoch
        2. Loss vs Epoch
    '''
    plt.figure()
    plt.plot(mod.history['accuracy'])
    plt.plot(mod.history['val_accuracy'])
    #plt.title('Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'])
    plt.show()

    plt.figure()
    plt.plot(mod.history['loss'])
    plt.plot(mod.history['val_loss'])
    #plt.title('Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'])
    plt.show()


if __name__ == "__main__":
    # # Testing Code
    if (sys.argv[1] == 'test'):
        print('ok')
        # Load images
        train_set = ImageLoader(file_names_train, image_train_path)
        val_set = ImageLoader(file_names_val, image_val_path)

        type = 'CNN'
        Xtrain = train_set.load_data(type)
        Xval = val_set.load_data(type)

        # Load and encode labels
        Ytrain, Yval = label_encoder(gender_label_train_path, gender_label_val_path)

        # Check the shape of everything
        print(f'Xtrain is of shape: {Xtrain.shape}')
        print(f'Ytrain is of shape: {Ytrain.shape}')
        print(f'Ytrain has {Ytrain.shape[1]} features....')

        print(f'Xval is of shape: {Xval.shape}')
        print(f'Yval is of shape: {Yval.shape}')
        print(f'Yval has {Ytrain.shape[1]} features....')



        # #### random test network ####
        # print("Initializing network...")
        # model = Sequential()
        # model.add(Dense(1024, input_shape=(1024,), activation="tanh"))
        # model.add(Dense(512, activation="sigmoid"))
        # model.add(Dense(100, activation="relu"))
        # model.add(Dense(Yval.shape[1], activation="softmax"))
        # model.summary()
        #
        #
        # # train the model using ADAM
        # opt = tf.keras.optimizers.Adam(learning_rate=0.001) #SGD(0.00001) #0.0000001
        # model.compile(loss="categorical_crossentropy", optimizer=opt,
        #             metrics=["accuracy"])
        # H = model.fit(Xtrain, Ytrain, validation_data=(Xval, Yval),
        #               epochs=10, batch_size=640) #callbacks=[model_checkpoint_callback]
        # building a linear stack of layers with the sequential model

        #### random test CNN ####
        model = Sequential()
        # convolutional layer
        model.add(
            Conv2D(40, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='relu', input_shape=(32, 32, 1)))
        model.add(MaxPool2D(pool_size=(2, 2)))
        # flatten output of conv
        model.add(Flatten())
        # hidden layer
        model.add(Dense(100, activation='relu'))
        # output layer
        model.add(Dense(Yval.shape[1], activation='softmax'))
        # model.summary()
        # compiling the sequential model
        opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=opt)
        # training the model for 10 epochs
        H = model.fit(Xtrain, Ytrain, batch_size=128, epochs=100, validation_data=(Xval, Yval), callbacks=[early_stopping])

        acc_loss_plotting(H)




    # Lo = Loader(file_names_train, image_train_path, label_train_path)
    # test, test2 = Lo.load_data()
    # print(len(test))
    # tt = test[0]
    # plt.imshow(tt[0], cmap='gray');
    # plt.show()

    elif (sys.argv[1] == 'task1'):
        train_set = ImageLoader(file_names_train, image_train_path)
        val_set = ImageLoader(file_names_val, image_val_path)

        type = '1d'
        Xtrain = train_set.load_data(type)
        Xval = val_set.load_data(type)

        #Ytrain_gender, Yval_gender = label_encoder(gender_label_train_path, gender_label_val_path)
        #Ytrain_race, Yval_race = label_encoder(race_label_train_path, race_label_val_path)
        Ytrain_age, Yval_age = label_encoder(age_label_train_path, age_label_val_path)

        #gen = task1_model(Xtrain, Ytrain_gender, Xval, Yval_gender)
        #rac = task1_model(Xtrain, Ytrain_race, Xval, Yval_race)
        ag = task1_model(Xtrain, Ytrain_age, Xval, Yval_age)

        #acc_loss_plotting(rac)
        #acc_loss_plotting(gen)
        acc_loss_plotting(ag)





    # elif (sys.argv[0] == 'task2'):
    # elif (sys.argv[0] == 'task4'):
    # elif (sys.argv[0] == 'task5'):



