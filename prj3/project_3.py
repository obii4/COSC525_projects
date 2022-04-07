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
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD


# define path to images and label files, the _slim.csv
# file only contains the features of interest
image_train_path = '/Users/chrisobrien/Desktop/p3/train/'
label_train_path = '/Users/chrisobrien/Desktop/grad school/courses/spring 2022/cosc 525/COSC525_projects/prj3/fairface_label_train_gender.csv'

image_val_path = '/Users/chrisobrien/Desktop/p3/val/'
label_val_path = '/Users/chrisobrien/Desktop/grad school/courses/spring 2022/cosc 525/COSC525_projects/prj3/fairface_label_val_gender.csv'


# load all of file names to a variable
# sort in numerical order so that it matches the label file order ie 0, 1, 2, etc
file_names_train = [os.path.basename(f) for f in glob.glob(image_train_path+'*.jpg')]
file_names_train.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
#file_names_train = file_names_train[0:10000]

file_names_val = [os.path.basename(f) for f in glob.glob(image_val_path+'*.jpg')]
file_names_val.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
#file_names_val = file_names_val[0:1000]



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
    with open(path, 'r') as f:
        load = csv.reader(f, delimiter=',')
        next(load)
        text = [text for text in load]
    data_array = np.asarray(text)
    #data_array = data_array[0:10000]

    with open(path2, 'r') as f:
        load = csv.reader(f, delimiter=',')
        next(load)
        text = [text for text in load]
    data_array2 = np.asarray(text)
    #data_array2 = data_array2[0:1000]

    oHenc = OneHotEncoder()
    oHenc.fit(data_array)
    # print(enc.categories_) #if you want to view the encoded categories
    train_lab = oHenc.transform(data_array).toarray()
    val_lab = oHenc.transform(data_array2).toarray()

    #convert each label to array
    out1 = np.array([np.array(xi) for xi in train_lab])
    out2 = np.array([np.array(xi) for xi in val_lab])

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
    def load_data(self, paths=None):
        '''
        loads images
        '''

        images = []
        file_names = paths if paths is not None else self.file_names
        for file_name in tqdm(file_names):
            image_name = self.image_path + file_name
            images.append(self.load_image(image_name))

        images_all = []
        for i in range(len(images)):
            t = images[i].flatten()
            images_all.append(t)
        images_all_out = np.asarray(images_all) #each element of the array is (1024,)

        if 'train' in self.image_path:
            print(f'Loaded {len(self.file_names)} images w/ labels for the training set!')
        else:
            print(f'Loaded {len(self.file_names)} images w/ labels for the validation set!')


        return images_all_out


if __name__ == "__main__":
    # Testing Code

    # Load images
    train_set = ImageLoader(file_names_train, image_train_path)
    val_set = ImageLoader(file_names_val, image_val_path)
    Xtrain = train_set.load_data()
    Xval = val_set.load_data()

    # Load and encode labels
    Ytrain, Yval = label_encoder(label_train_path, label_val_path)

    print(Yval.shape[1])
    # Check the shape of everything
    print(f'Xtrain is of shape: {Xtrain.shape}')
    print(f'Ytrain is of shape: {Ytrain.shape}')
    print(f'Ytrain has {Ytrain.shape[1]} features....')

    print(f'Xval is of shape: {Xval.shape}')
    print(f'Yval is of shape: {Yval.shape}')
    print(f'Yval has {Ytrain.shape[1]} features....')



    #### random test network ####
    print("Initializing network...")
    model = Sequential()
    model.add(Dense(1024, input_shape=(1024,), activation="tanh"))
    model.add(Dense(512, activation="sigmoid"))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(Yval.shape[1], activation="softmax"))
    model.summary()

    #confusion = tf.confusion_matrix(labels=model_, predictions=model, num_classes=num_classes)

    # model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    #     filepath='/Users/chrisobrien/Desktop/cp.cpkt',
    #     save_weights_only=True,
    #     monitor='val_accuracy',
    #     mode='max',
    #     save_best_only=True)

    # train the model using SGD
    sgd = SGD(0.00001) #0.0000001
    model.compile(loss="categorical_crossentropy", optimizer=sgd,
                metrics=[tf.keras.metrics.CategoricalAccuracy()])
    H = model.fit(Xtrain, Ytrain, validation_data=(Xval, Yval),
                  epochs=1000, batch_size=640) #callbacks=[model_checkpoint_callback]

    print("[INFO] evaluating network...")
    predictions = model.predict(Xval, batch_size=640)
    #print(predictions.shape[1])
    #print(predictions)


    preds = (predictions > 0.5)#.long()
    #print(preds)


    #
    # u = tf.stack([Yval, predictions], axis=1)
    # print(u)
    #print(classification_report(Yval, predictions))



    # Lo = Loader(file_names_train, image_train_path, label_train_path)
    # test, test2 = Lo.load_data()
    # print(len(test))
    # tt = test[0]
    # plt.imshow(tt[0], cmap='gray');
    # plt.show()

    # if (sys.argv[0] == 'task1'):
    # elif (sys.argv[0] == 'task2'):
    # elif (sys.argv[0] == 'task4'):
    # elif (sys.argv[0] == 'task5'):



