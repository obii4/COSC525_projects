import os, sys, glob, csv, re
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder

# image processing
from PIL import Image

# define path to images and label files, the _slim.csv
# file only contains the features of interest
image_train_path = '/Users/chrisobrien/Desktop/p3/train/'
label_train_path = '/Users/chrisobrien/Desktop/p3/fairface_label_train_slim.csv'

image_val_path = '/Users/chrisobrien/Desktop/p3/val/'
label_val_path = '/Users/chrisobrien/Desktop/p3/fairface_label_val_slim.csv'


# load all of file names to a variable then sort in numerical order so
# that is matches the label file order ie 0, 1, 2, etc
file_names_train = [os.path.basename(f) for f in glob.glob(image_train_path+'*.jpg')]
file_names_train.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
file_names_train[0:10]

file_names_val = [os.path.basename(f) for f in glob.glob(image_val_path+'*.jpg')]
file_names_val.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

class Loader:
    '''
    Loads images and corresponding labels together.

    Input:
    file_names -> names of all images
    image_path -> path to location of images
    label_path -> path to location of label csv

    Output:
    images_and_labels
    '''

    def __init__(self, file_names, image_path, label_path):
        self.file_names = file_names
        self.image_path = image_path
        self.label_path = label_path

    def __len__(self):
        return len(self.file_names)

    # load RGB image
    def load_image(self, path):
        '''
        Loads images as np array and scale data to the range of [0, 1]
        '''
        image = np.array(Image.open(path), dtype=float) / 255
        return image

    def load_labels(self, path):
        '''
        Loads csv file of labels as np array
        '''
        with open(path, 'r') as f:
            load = csv.reader(f, delimiter=',')
            next(load)
            text = [text for text in load]
        data_array = np.asarray(text)
        return data_array[0:10] #change this based off of how many filenames you are loading

    def encode(self, data_array):
        '''
        encodes labels and returns as np array
        '''
        enc = OneHotEncoder()
        enc.fit(data_array)
        #print(enc.categories_) #if you want to view the encoded categories
        endcoded_array = enc.transform(data_array).toarray()
        return endcoded_array



    # load images, masks, and (if applicable) roi's
    def load_data(self, paths=None):
        '''
        where the magic happens
        '''
        encoded_labels = self.encode(self.load_labels(self.label_path))
        images, images_and_labels = [], []
        file_names = paths if paths is not None else self.file_names
        for file_name in file_names:
            image_name = self.image_path + file_name
            images.append(self.load_image(image_name))

        for i, j in zip(images, encoded_labels):
            images_and_labels.append((i, j))
        return images_and_labels


if __name__ == "__main__":
    # Testing Code
    Lo = Loader(file_names_train, image_train_path, label_train_path)
    test = Lo.load_data()
    print(len(test))
    tt = test[0]
    plt.imshow(tt[0], cmap='gray');
    plt.show()



