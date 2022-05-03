import sys
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import random

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, SimpleRNN, MaxPool2D, Flatten, Input, LSTM

import sklearn.model_selection as sk
from tensorflow.keras.utils import to_categorical

class generate:
    '''
    loads text file -> encodes -> one hot encodes.

    Input:
    sample -> text string of txt file name
    win_size -> moving window size
    stride_size -> stride increments

    Output:
    x_oneHOT_out -> og uncaoded pairs
    y_oneHOT_out -> encoded letter vectors
    '''

    def __init__(self, sample, win_size, stride_size):
        self.sample = sample
        self.win_size = win_size
        self.stride_size = stride_size

    def load_text(self, sample):
        with open(sample) as f:
            contents = f.read()
            f.close()
        return contents

    def x_y_one_hot(self, x, y):
        x_oneHOT = to_categorical(x) #num_classes=48) #num_classes=47)
        y_oneHOT = to_categorical(y) #num_classes=47)
        return x_oneHOT, y_oneHOT

    def idiot_convert_to(self, sample):
        '''
        Encodes each character

        Input:
        sample -> string of text

        output:
        encoded -> numerical encoding

        '''
        random.seed(13)
        ugh = list(set(sample))
        dict_convert_to = dict(zip(ugh, range(0, len(ugh))))

        encoded = []
        for i in range(len(sample)):
            convert = dict_convert_to[sample[i]]
            encoded.append(convert)
        return encoded

    def idiot_convert_out(self, sample, encoded):
        '''
        decodes each character

        Input:
        sample -> OG string of text
        encoded -> numerical encoding

        output:
        decoded -> letters and chars
        '''
        random.seed(13)
        ugh = list(set(sample))
        dict_convert_out = dict(zip(range(0, len(ugh)), ugh))

        decoded = []
        for i in range(len(encoded)):
            convert = dict_convert_out[encoded[i]]
            decoded.append(convert)
        return decoded

    def text_encoder(self):
        '''
        Encodes each character of text to numeric val
        and generates x,y vector according to a user
        defined window and stride size. finally,
        one hot encodes x and y vectors..

        Input:
        sample -> text string
        win_size -> moving window size
        stride_size -> stride increments

        Output:
        x_oneHOT_out -> og pairs
        y_oneHOT_out -> encoded letter vectors
        '''

        sample = self.load_text(self.sample)
        encoded = self.idiot_convert_to(sample)
        print(type(encoded))

        # ****decode back to text from numbers**** #
        #de = self.idiot_convert_out(sample, encoded)

        og = [] #old
        en = [] #new
        for i in range(len(encoded) - self.win_size + 1):
            og.append(encoded[i: i + self.win_size])

        x = og[::self.stride_size] #keeps only the items of specified stride size
        x = np.asarray(x)

        # print(f'unique count {len(np.unique(x))}')
        # print(f'unique {np.unique(x)}')

        for i in range(len(encoded) - self.win_size + 1):
            en.append(encoded[i+1: i + self.win_size + 1]) #+ 1 added

        y = en[::self.stride_size] #keeps only the items of specified stride size
        y = np.asarray(y)
        x_oneHOT_out, y_oneHOT_out = self.x_y_one_hot(x, y)

        return x_oneHOT_out, y_oneHOT_out

def acc_loss_plotting(mod):
    '''
    Takes a trained model and summarizes training through
    plotting:
        1. Accuracy vs Epoch
        2. Loss vs Epoch
    '''
    #plt.figure(figsize=(15, 10))
    plt.subplot(1, 2, 1)
    plt.plot(mod.history['accuracy'])
    plt.plot(mod.history['val_accuracy'])
    plt.title('Model Results - Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'])
    #plt.savefig('/Users/chrisobrien/Desktop/grad school/courses/spring 2022/cosc 525/COSC525_projects/prj3/figs_final/task3_race_acc.png')

    #plt.figure(figsize=(15, 10))
    plt.subplot(1, 2, 2)
    plt.plot(mod.history['loss'])
    plt.plot(mod.history['val_loss'])
    plt.title('Model Results - Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'])
    #plt.savefig('/Users/chrisobrien/Desktop/grad school/courses/spring 2022/cosc 525/COSC525_projects/prj3/figs_final/task3_race_loss.png')
    plt.tight_layout()
    plt.show()

def LSTM_model(x, y, hid_state_size):
    '''
    generates lstm network
    '''

    Xtrain, Xval, Ytrain, Yval = sk.train_test_split(x, y, test_size=0.33, random_state=42)
    print("Initializing LSTM network...")
    model = Sequential()
    model.add(LSTM(hid_state_size, input_shape=(x.shape[1], x.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Dense(y.shape[2], activation='softmax'))
    model.summary()
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=opt)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    model_out = model.fit(Xtrain, Ytrain, batch_size=1024, epochs=500, validation_data=(Xval, Yval),
                          callbacks=[early_stopping])

    #gen_conf_mat(Yval, predictions)
    acc_loss_plotting(model_out)
    return model_out

# modified from https://stackoverflow.com/questions/36864774/python-keras-how-to-access-each-epoch-prediction
class Gen_text(tf.keras.callbacks.Callback):
    def __init__(self, model, XVal, N):
        self.model = model
        self.XVal = XVal
        self.N = N
        self.epoch_num = 0

    def on_epoch_end(self, epoch, log={}):
        if self.epoch_num % self.N == 0:
            pred = self.model.predict(self.XVal)
            print('y predicted: ', pred)
        self.epoch_num += 1

    # Use callbacks=[CustomCallback(model, x_test, y_test)])

def simpleRNN_model(x, y, hid_state_size):
    '''
     simple rnn lstm network
    '''

    Xtrain, Xval, Ytrain, Yval = sk.train_test_split(x, y, test_size=0.33, random_state=42)

    print("Initializing simpleRNN network...")
    model = Sequential()
    model.add(SimpleRNN(hid_state_size, input_shape=(x.shape[1], x.shape[2]), return_sequences=True))
    model.add(Dense(y.shape[2], activation='softmax'))
    model.summary()
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=opt)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 10)

    model_out = model.fit(Xtrain, Ytrain, batch_size=1024, epochs=500, validation_data=(Xval, Yval),
                          callbacks=[early_stopping]) #callbacks=[early_stopping, Gen_text(model, Xval, 10)])

    acc_loss_plotting(model_out)

    return model_out



if __name__ == "__main__":
    # Testing Code

    if (sys.argv[1] == 'test'):
        test = 'beatles.txt'
        code = generate(test, 5, 3)
        x,y = code.text_encoder()
        print(f' the shape of x: {x.shape} (num of sequences x window size x vocab size)')
        print(f' the shape of y: {y.shape} (num of sequences x window size x vocab size)')


        # # method 1 tester *** output his sample ***

        # sample = 'hello, how are you?'
        # win_size = 5
        # stride_size = 3
        # encoded = []
        # for i in sample:
        #     convert = ord(i)  # ord() accepts a string of length 1 as an
        #     # argument and returns the unicode code point representation of the passed argument
        #     encoded.append(convert)
        #
        # og = []  # old
        # en = []  # new
        # for i in range(len(sample) - win_size + 1):
        #     og.append(sample[i: i + win_size])
        #
        # x = og[::stride_size]
        #
        # for i in range(len(sample) - win_size + 1):
        #     en.append(sample[i + 1: i + win_size + 1])
        #
        # y = en[::stride_size]
        #
        # print(x)
        # for i in range(len(x)):
        #     print(f'x{i}: {x[i]} ~ y{i}: {y[i]}')

    elif (sys.argv[1] == 'lstm'):
        hid_state_size = int(sys.argv[2])
        win_size = int(sys.argv[3])
        stride_size = int(sys.argv[4])
        #temp = int(sys.argv[5])

        file = 'beatles.txt'
        code = generate(file, win_size, stride_size)
        x, y = code.text_encoder()

        print(f' the shape of x: {x.shape} (num of sequences x window size x vocab size)')
        print(f' the shape of y: {y.shape} (num of sequences x window size x vocab size)')

        mod = LSTM_model(x, y, hid_state_size)

        pd.DataFrame.from_dict(mod.history).to_csv(f'history_lstm_{hid_state_size}_winsize{x.shape[1]}_stride{stride_size}.csv',
                                                         index=False)

    elif (sys.argv[1] == 'simplernn'):
        hid_state_size = int(sys.argv[2])
        win_size = int(sys.argv[3])
        stride_size = int(sys.argv[4])
        # temp = int(sys.argv[5])

        file = 'beatles.txt'
        code = generate(file, win_size, stride_size)
        x, y = code.text_encoder()

        print(f' the shape of x: {x.shape} (num of sequences x window size x vocab size)')
        print(f' the shape of y: {y.shape} (num of sequences x window size x vocab size)')

        mod = simpleRNN_model(x, y, hid_state_size)

        pd.DataFrame.from_dict(mod.history).to_csv(f'history_simpleRNN_{hid_state_size}_winsize{x.shape[1]}_stride{stride_size}.csv',
                                                         index=False)




