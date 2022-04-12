import os, sys, glob, csv, re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn #for heatmaps

from sklearn.preprocessing import OneHotEncoder

#monitor image loading loop for some visual feedback
from tqdm import tqdm

# image processing
from PIL import Image

import tensorflow as tf
from tensorflow.math import confusion_matrix
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, Input

# define path to project folder
path_to_prj = os.getcwd()


# define path to images and label files
image_train_path = '/Users/chrisobrien/Desktop/p3/train/'
gender_label_train_path = path_to_prj + '/labels/fairface_label_train_gender.csv'
race_label_train_path = path_to_prj + '/labels/fairface_label_train_race.csv'
age_label_train_path = path_to_prj + '/labels/fairface_label_train_age.csv'
all_label_train_path = path_to_prj + '/labels/fairface_label_train_slim.csv'

image_val_path = '/Users/chrisobrien/Desktop/p3/val/'
gender_label_val_path = path_to_prj + '/labels/fairface_label_val_gender.csv'
race_label_val_path = path_to_prj + '/labels/fairface_label_val_race.csv'
age_label_val_path = path_to_prj + '/labels/fairface_label_val_age.csv'
all_label_val_path = path_to_prj + '/labels/fairface_label_val_slim.csv'



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
    #print(oHenc.categories_) #if you want to view the encoded categories
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

def acc_loss_plotting(mod):
    '''
    Takes a trained model and summarizes training through
    plotting:
        1. Accuracy vs Epoch
        2. Loss vs Epoch
    '''
    plt.figure(figsize=(15, 10))
    plt.plot(mod.history['accuracy'])
    plt.plot(mod.history['val_accuracy'])
    plt.title('Race Classification')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'])
    #plt.savefig('/Users/chrisobrien/Desktop/grad school/courses/spring 2022/cosc 525/COSC525_projects/prj3/figs_final/task3_race_acc.png')
    plt.show()

    plt.figure(figsize=(15, 10))
    plt.plot(mod.history['loss'])
    plt.plot(mod.history['val_loss'])
    plt.title('Race Classification')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'])
    #plt.savefig('/Users/chrisobrien/Desktop/grad school/courses/spring 2022/cosc 525/COSC525_projects/prj3/figs_final/task3_race_loss.png')
    plt.show()

def acc_loss_plotting_multi(mod):
    '''
    Takes a trained model and summarizes training through
    plotting:
        1. Accuracy vs Epoch
        2. Loss vs Epoch
    '''
    plt.figure(figsize=(15, 10))
    plt.plot(mod.history['gender_accuracy'])
    plt.plot(mod.history['val_gender_accuracy'])
    plt.title('Gender Classification')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'])
    #plt.savefig('/Users/chrisobrien/Desktop/grad school/courses/spring 2022/cosc 525/COSC525_projects/prj3/figs_final/task4_gender_acc.png')
    plt.show()

    plt.figure(figsize=(15, 10))
    plt.plot(mod.history['age_accuracy'])
    plt.plot(mod.history['val_age_accuracy'])
    plt.title('Age Classification')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'])
    #plt.savefig('/Users/chrisobrien/Desktop/grad school/courses/spring 2022/cosc 525/COSC525_projects/prj3/figs_final/task4_age_acc.png')
    plt.show()

    plt.figure(figsize=(15, 10))
    plt.plot(mod.history['race_accuracy'])
    plt.plot(mod.history['val_race_accuracy'])
    plt.title('Race Classification')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'])
    #plt.savefig('/Users/chrisobrien/Desktop/grad school/courses/spring 2022/cosc 525/COSC525_projects/prj3/figs_final/task4_race_acc.png')
    plt.show()

    plt.figure(figsize=(15, 10))
    plt.plot(mod.history['gender_loss'])
    plt.plot(mod.history['val_gender_loss'])
    plt.title('Gender Classification')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'])
    #plt.savefig('/Users/chrisobrien/Desktop/grad school/courses/spring 2022/cosc 525/COSC525_projects/prj3/figs_final/task4_gender_loss.png')
    plt.show()

    plt.figure(figsize=(15, 10))
    plt.plot(mod.history['age_loss'])
    plt.plot(mod.history['val_age_loss'])
    plt.title('Age Classification')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'])
    #plt.savefig('/Users/chrisobrien/Desktop/grad school/courses/spring 2022/cosc 525/COSC525_projects/prj3/figs_final/task4_age_loss.png')
    plt.show()

    plt.figure(figsize=(15, 10))
    plt.plot(mod.history['race_loss'])
    plt.plot(mod.history['val_race_loss'])
    plt.title('Race Classification')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'])
    #plt.savefig('/Users/chrisobrien/Desktop/grad school/courses/spring 2022/cosc 525/COSC525_projects/prj3/figs_final/task4_race_loss.png')
    plt.show()

def gen_conf_mat(Yval, pred):
    cm = confusion_matrix(tf.argmax(Yval, axis=1), tf.argmax(pred, axis=1))
    cm = (cm.numpy()).tolist()
    plt.figure(figsize = (15,10))
    ax = sn.heatmap(cm, annot=True, fmt='g')
    ax.set_xlabel('Predicted labels');
    ax.set_ylabel('True labels');

    if Yval.shape[1] == 2: #gender
        ax.set_title('Gender Classifcation Confusion Matrix')
        ax.xaxis.set_ticklabels(['Female', 'Male']);
        ax.yaxis.set_ticklabels(['Female', 'Male']);
        #plt.savefig('/Users/chrisobrien/Desktop/grad school/courses/spring 2022/cosc 525/COSC525_projects/prj3/figs_final/task4_gender_cm.png')
        plt.show()

    elif Yval.shape[1] == 7: #race
        ax.set_title('Race Classification Confusion Matrix')
        ax.xaxis.set_ticklabels(['Black', 'East Asian', 'Indian', 'Latino Hispanic',
       'Middle Eastern', 'Southeast Asian', 'White']);
        ax.yaxis.set_ticklabels(['Black', 'East Asian', 'Indian', 'Latino Hispanic',
       'Middle Eastern', 'Southeast Asian', 'White']);
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=30)
        #plt.savefig('/Users/chrisobrien/Desktop/grad school/courses/spring 2022/cosc 525/COSC525_projects/prj3/figs_final/task4_race_cm.png')
        plt.show()

    elif Yval.shape[1] == 9: #age
        ax.set_title('Age Classification Confusion Matrix')
        ax.xaxis.set_ticklabels(['0-2', '10-19', '20-29', '3-9', '30-39', '40-49', '50-59',
       '60-69', '70+']);
        ax.yaxis.set_ticklabels(['0-2', '10-19', '20-29', '3-9', '30-39', '40-49', '50-59',
       '60-69', '70+']);
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=30)
        #plt.savefig('/Users/chrisobrien/Desktop/grad school/courses/spring 2022/cosc 525/COSC525_projects/prj3/figs_final/task4_age_cm.png')
        plt.show()



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

    model.summary()
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001) #SGD(0.00001) #0.0000001
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                metrics=["accuracy"])
    model_out = model.fit(Xtrain, Ytrain, validation_data=(Xval, Yval),
                  epochs=150, batch_size=640, callbacks=[early_stopping]) #150

    best_score = max(model_out.history['val_accuracy'])

    predictions = model.predict(Xval)

    gen_conf_mat(Yval, predictions)
    acc_loss_plotting(model_out)

    print(f'Max validation acc {best_score}')
    return model_out

def task2_model(Xtrain, Ytrain, Xval, Yval):

    '''
    Used for the completion of task 2
    '''

    print("Initializing network...")
    model = Sequential()
    model.add(
        Conv2D(40, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='relu', input_shape=(32, 32, 1)))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(Yval.shape[1], activation='softmax'))
    #model.summary()
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=opt)
    # training the model for 10 epochs
    model_out = model.fit(Xtrain, Ytrain, batch_size=128, epochs=100, validation_data=(Xval, Yval), callbacks=[early_stopping])

    best_score = max(model_out.history['val_accuracy'])

    predictions = model.predict(Xval)

    gen_conf_mat(Yval, predictions)
    acc_loss_plotting(model_out)

    print(f'Max validation acc {best_score}')
    return model_out

def task3_model(Xtrain, Ytrain, Xval, Yval):
    '''
    Used for the completion of task 3
    '''
    print("Initializing network...")
    model = Sequential()
    model.add(
        Conv2D(90, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='relu', input_shape=(32, 32, 1)))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(
        Conv2D(80, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='relu', input_shape=(14, 14, 40)))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(Yval.shape[1], activation='softmax'))
    model.summary()
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=opt)
    model_out = model.fit(Xtrain, Ytrain, batch_size=128, epochs=100, validation_data=(Xval, Yval),
                  callbacks=[early_stopping])
    best_score = max(model_out.history['val_accuracy'])

    predictions = model.predict(Xval)

    gen_conf_mat(Yval, predictions)
    acc_loss_plotting(model_out)

    print(f'Max validation acc {best_score}')
    return model_out

def task4_model(Xtrain, Ytrain_gender, Ytrain_age, Ytrain_race, Xval, Yval_gender, Yval_age, Yval_race):
    '''
    Used for the completion of task 4
    '''
    print("Initializing network...")
    input = Input(shape=(32, 32, 1))
    layer1 = Conv2D(90, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='relu')
    maxpool1 = MaxPool2D(pool_size=(2, 2))
    layer2 = Conv2D(80, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='relu',
                    input_shape=(14, 14, 40))
    maxpool2 = MaxPool2D(pool_size=(2, 2))

    model = layer1(input)
    model = maxpool1(model)
    model = layer2(model)
    model = maxpool2(model)
    model = Flatten()(model)

    #develop branchs for each task
    b1 = Dense(1000, activation='relu')(model)
    b1 = Dense(2, activation='softmax', name='gender')(b1)

    b2 = Dense(1000, activation='relu')(model)
    b2 = Dense(9, activation='softmax', name='age')(b2)

    b3 = Dense(1000, activation='relu')(model)
    b3 = Dense(7, activation='softmax', name='race')(b3)

    model = Model(inputs=[input], outputs=[b1, b2, b3])
    model.summary()

    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=opt)
    model_out = model.fit(Xtrain, [Ytrain_gender, Ytrain_age, Ytrain_race], batch_size=128, callbacks=[early_stopping],
              validation_data=(Xval, [Yval_gender, Yval_age, Yval_race]), epochs=4) #100

    #obtain predictions for each class
    predictions = model.predict(Xval)
    gen_pred = predictions[0]
    age_pred = predictions[1]
    race_pred = predictions[2]

    gen_conf_mat(Yval_gender, gen_pred)
    gen_conf_mat(Yval_age, age_pred)
    gen_conf_mat(Yval_race, race_pred)

    acc_loss_plotting_multi(model_out)

    return model_out

if __name__ == "__main__":
    # # Testing Code
    if (sys.argv[1] == 'test'):
        print('ok, working with the test code')
        #Load images
        train_set = ImageLoader(file_names_train, image_train_path)
        val_set = ImageLoader(file_names_val, image_val_path)

        type = 'CNN'
        Xtrain = train_set.load_data(type)
        Xval = val_set.load_data(type)

        # Load and encode labels
        Ytrain, Yval = label_encoder(race_label_train_path, race_label_val_path)

        # Check the shape of everything
        print(f'Xtrain is of shape: {Xtrain.shape}')
        print(f'Ytrain is of shape: {Ytrain.shape}')
        print(f'Ytrain has {Ytrain.shape[1]} features....')

        print(f'Xval is of shape: {Xval.shape}')
        print(f'Yval is of shape: {Yval.shape}')
        print(f'Yval has {Ytrain.shape[1]} features....')


    elif (sys.argv[1] == 'task1'):
        train_set = ImageLoader(file_names_train, image_train_path)
        val_set = ImageLoader(file_names_val, image_val_path)

        type = '1d'
        Xtrain = train_set.load_data(type)
        Xval = val_set.load_data(type)

        Ytrain_gender, Yval_gender = label_encoder(gender_label_train_path, gender_label_val_path)
        Ytrain_race, Yval_race = label_encoder(race_label_train_path, race_label_val_path)
        Ytrain_age, Yval_age = label_encoder(age_label_train_path, age_label_val_path)

        gen = task1_model(Xtrain, Ytrain_gender, Xval, Yval_gender)
        rac = task1_model(Xtrain, Ytrain_race, Xval, Yval_race)
        ag = task1_model(Xtrain, Ytrain_age, Xval, Yval_age)

    elif (sys.argv[1] == 'task2'):
        train_set = ImageLoader(file_names_train, image_train_path)
        val_set = ImageLoader(file_names_val, image_val_path)

        type = 'CNN'
        Xtrain = train_set.load_data(type)
        Xval = val_set.load_data(type)

        Ytrain_gender, Yval_gender = label_encoder(gender_label_train_path, gender_label_val_path)
        Ytrain_race, Yval_race = label_encoder(race_label_train_path, race_label_val_path)
        Ytrain_age, Yval_age = label_encoder(age_label_train_path, age_label_val_path)

        gen = task2_model(Xtrain, Ytrain_gender, Xval, Yval_gender)
        rac = task2_model(Xtrain, Ytrain_race, Xval, Yval_race)
        ag = task2_model(Xtrain, Ytrain_age, Xval, Yval_age)

    elif (sys.argv[1] == 'task3'):
        train_set = ImageLoader(file_names_train, image_train_path)
        val_set = ImageLoader(file_names_val, image_val_path)

        type = 'CNN'
        Xtrain = train_set.load_data(type)
        Xval = val_set.load_data(type)

        Ytrain_gender, Yval_gender = label_encoder(gender_label_train_path, gender_label_val_path)
        Ytrain_race, Yval_race = label_encoder(race_label_train_path, race_label_val_path)
        Ytrain_age, Yval_age = label_encoder(age_label_train_path, age_label_val_path)

        gen = task3_model(Xtrain, Ytrain_gender, Xval, Yval_gender)
        rac = task3_model(Xtrain, Ytrain_race, Xval, Yval_race)
        ag = task3_model(Xtrain, Ytrain_age, Xval, Yval_age)

    elif (sys.argv[1] == 'task4'):
        train_set = ImageLoader(file_names_train, image_train_path)
        val_set = ImageLoader(file_names_val, image_val_path)

        type = 'CNN'
        print(type)
        Xtrain = train_set.load_data(type)
        Xval = val_set.load_data(type)

        Ytrain_gender, Yval_gender = label_encoder(gender_label_train_path, gender_label_val_path)
        Ytrain_race, Yval_race = label_encoder(race_label_train_path, race_label_val_path)
        Ytrain_age, Yval_age = label_encoder(age_label_train_path, age_label_val_path)

        final = task4_model(Xtrain, Ytrain_gender, Ytrain_age, Ytrain_race, Xval, Yval_gender, Yval_age, Yval_race)

    # elif (sys.argv[0] == 'task5'):



