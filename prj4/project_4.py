import sys
from sklearn.preprocessing import OneHotEncoder
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, SimpleRNN, MaxPool2D, Flatten, Input, LSTM


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
        x_oneHOT = to_categorical(x)
        y_oneHOT = to_categorical(y)
        return x_oneHOT, y_oneHOT

    def text_encoder(self):
        '''
        Encodes each character of text to unicode
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
        encoded = []
        for i in sample:
            convert = ord(i) # ord() accepts a string of length 1 as an
                             # argument and returns the unicode code point representation of the passed argument
            encoded.append(convert)

        og = [] #old
        en = [] #new
        for i in range(len(encoded) - self.win_size + 1):
            og.append(encoded[i: i + self.win_size])

        x = og[::self.stride_size] #keeps only the items of specified stride size

        for i in range(len(encoded) - self.win_size + 1):
            en.append(encoded[i+1: i + self.win_size + 1]) #+ 1 added

        y = en[::self.stride_size] #keeps only the items of specified stride size
        y = np.array(y)
        x_oneHOT_out, y_oneHOT_out = self.x_y_one_hot(x, y)

        return x_oneHOT_out, y_oneHOT_out

def acc_loss_plotting(mod):
    '''
    Takes a trained model and summarizes training through
    plotting:
        1. Accuracy vs Epoch
        2. Loss vs Epoch
    '''
    plt.figure(figsize=(15, 10))
    plt.plot(mod.history['accuracy'])
    plt.title('Race Classification')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    #plt.legend(['train', 'validation'])
    #plt.savefig('/Users/chrisobrien/Desktop/grad school/courses/spring 2022/cosc 525/COSC525_projects/prj3/figs_final/task3_race_acc.png')
    plt.show()

    plt.figure(figsize=(15, 10))
    plt.plot(mod.history['loss'])
    plt.title('Race Classification')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    #plt.legend(['train', 'validation'])
    #plt.savefig('/Users/chrisobrien/Desktop/grad school/courses/spring 2022/cosc 525/COSC525_projects/prj3/figs_final/task3_race_loss.png')
    plt.show()

def LSTM_model(x, y, hid_state_size):
    '''
    generates lstm network
    '''
    print("Initializing LSTM network...")
    model = Sequential()
    model.add(LSTM(hid_state_size, input_shape=(x.shape[1], x.shape[2]), return_sequences=True))
    model.add(Dense(y.shape[2], activation='softmax'))
    model.summary()
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=opt)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=30)
    model_out = model.fit(x, y, batch_size=1024, epochs=100,
                          callbacks=[early_stopping])
    return model_out

def simpleRNN_model(x, y, hid_state_size):
    '''
     simple rnn lstm network
    '''
    print("Initializing simpleRNN network...")
    model = Sequential()
    model.add(SimpleRNN(hid_state_size, input_shape=(x.shape[1], x.shape[2]), return_sequences=True))
    model.add(Dense(y.shape[2], activation='softmax'))
    model.summary()
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=opt)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=30)
    model_out = model.fit(x, y, batch_size=1024, epochs=100,
                          callbacks=[early_stopping])
    return model_out



if __name__ == "__main__":
    # Testing Code

    if (sys.argv[1] == 'test'):
        test = 'beatles.txt'
        code = generate(test, 5, 3)
        x,y = code.text_encoder()
        print(f' the shape of x: {x.shape} (num of sequences x window size x vocab size)')
        print(f' the shape of y: {y.shape} (num of sequences x window size x vocab size)')

        # method 1 tester *** output his sample ***

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
        #
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


    # elif (sys.argv[1] == 'task1'):


