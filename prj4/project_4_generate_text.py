import sys
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt

import random

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, SimpleRNN, MaxPool2D, Flatten, Input, LSTM, Lambda
from tensorflow.keras import layers

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
        with open(sample, encoding='utf-8-sig') as f:
            contents = f.read()
            #print(np.unique(contents))
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
        remove_list = ['the']
        word_list = sample.split()
        sample = sample.join([i for i in word_list if i not in remove_list])
        encoded = self.idiot_convert_to(sample)

        # ****decode back to text from numbers**** #
        de = self.idiot_convert_out(sample, encoded)

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
        y = np.asarray(y, dtype='object')
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

def LSTM_model(x, y, hid_state_size):
    '''
    generates lstm network
    '''

    Xtrain, Xval, Ytrain, Yval = sk.train_test_split(x, y, test_size=0.33, random_state=42)
    print("Initializing LSTM network...")
    model = Sequential()
    model.add(LSTM(hid_state_size, input_shape=(x.shape[1], x.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    #model.add(LSTM(256, return_sequences=True))
    #model.add(Dropout(0.2))
    #model.add(LSTM(128, return_sequences=True))
    #model.add(Dropout(0.2))
    model.add(Dense(y.shape[2], activation='softmax'))
    model.summary()
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=opt)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    model_out = model.fit(Xtrain, Ytrain, batch_size=1024, epochs=50, validation_data=(Xval, Yval),
                          callbacks=[early_stopping])

    #best_score = max(model_out.history['val_accuracy'])


    #gen_conf_mat(Yval, predictions)
    acc_loss_plotting(model_out)
    return model_out

def simpleRNN_model(x, y, hid_state_size, N, temp):
#def simpleRNN_model(x, y, hid_state_size):
    '''
     simple rnn lstm network
    '''

    Xtrain, Xval, Ytrain, Yval = sk.train_test_split(x, y, test_size=0.33, random_state=42)

    print(f"Xtrain shape = {Xtrain.shape}")

    init_seq = Xtrain[3000, :, :]


    print(f"init sequence shape = {init_seq.shape}")

    print("Initializing simpleRNN network...")
    model = Sequential()
    model.add(LSTM(hid_state_size, input_shape=(x.shape[1], x.shape[2]), return_sequences=True)) #SimpleRNN
    # model.add(SimpleRNN(256, return_sequences=True))
    #model.add(Dropout(0.2))
    # model.add(SimpleRNN(128, return_sequences=True))
    # model.add(Dropout(0.2))
    model.add(Dense(y.shape[2], activation='softmax'))
    model.summary()
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=opt)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 10)
    model_out = model.fit(Xtrain, Ytrain, batch_size=1024, epochs=1000, validation_data=(Xval, Yval),
                          callbacks=[early_stopping])
    

    ### Predicting new characters ###

    #model.reset_states()   # reset state of model

    # Add Temperature to Model

    input = layers.Input(shape=(x.shape[1], x.shape[2]))
    hidden_layer_1 = model.layers[0](input)
    temp_layer = Lambda(lambda x: x / temp)(hidden_layer_1)
    softmax_layer = model.layers[1](temp_layer)
    predictor_model = Model(input, softmax_layer)


    # Get initial characters to go off of


    init_seq = Xtrain[3000, :, :].reshape(1,x.shape[1],x.shape[2])   # get a sequence and reshape to 3D



    new_characters = []

    with open("beatles.txt", encoding='utf-8-sig') as f:
        contents = f.read()
        f.close()

    generator = generate(contents,1,1)

    #look at sequence


    # Predict N new characters
    for i in range(N):

        # predict new one_hot output
        pred_onehot = predictor_model.predict(init_seq)

        ans_1char = pred_onehot[0,0,:] # take only the first character's onehotencoding from the predicted sequence
        ans_number = np.argmax(ans_1char, axis=-1) # turn the onehotencoding to a number
        


        # transform answer to character and append to list
        pred_char = generator.idiot_convert_out(contents, [ans_number])   # turn the number into a character (string)
        new_characters.append(pred_char)

        # change initial sequence based on predicted character
        ans_1char = ans_1char.reshape(1,1,x.shape[2])  # Transform onehotencoding of character to 3D to add to init_seq


        #init_seq = np.concatenate((ans_1char, init_seq[:, 0:-1, :]))   # remove last character onehotenc, and insert onehotenc predicted to beginning
        init_seq = np.reshape(np.append(ans_1char, init_seq[:,  0:-1, :]), (1, x.shape[1],x.shape[2]))


    return model_out, new_characters




if __name__ == "__main__":
    # Testing Code

    if (sys.argv[1] == 'test'):
        test = 'beatles.txt'
        code = generate(test, 5, 3)
        x,y = code.text_encoder()
        print(f' the shape of x: {x.shape} (num of sequences x window size x vocab size)')
        print(f' the shape of y: {y.shape} (num of sequences x window size x vocab size)')

        # t = [',', '5', 'v', ')', '3', '7', 'm', 'l', ':', '9', 'd', 's', 'o', 'n', 'k', 'x', 't', '4', 'a', 'y', '1', 'u',
        #  '0', "'", '2', '’', 'i', 'c', '8', '.', 'z', 'j', 'p', 'e', 'g', '?', 'w', 'h', '-', 'b', ' ', 'q', 'f', 'r',
        #  '\n', '(', '!', '6']
        # num2alphadict = dict(zip(range(0, len(t)), t))
        # num2alphadict2 = dict(zip(t, range(0, len(t))))
        #
        # str = 'a day in the life i read the news today oh boy abo'
        # print(str[0])
        # print(num2alphadict2[str[0]])
        #
        # print(num2alphadict)
        # print(num2alphadict2)


        # t = x[0]
        # print(t)
        # test = np.argmax(x[0:10], axis=-1)
        #
        # u = []
        # for i in range(len(test)):
        #     t = [chr(x) for x in test[i]]
        #     u.append(t)
        # print(u)
        # tt = test[0]
        # print(tt)
        # for k in range(len(test)):
        #         tt = test[k]
        #         print(chr(tt[k]))

        # y = [0, 1, 2, 0, 4, 5]
        # Y = to_categorical(y, num_classes=len(y))
        # print(Y)
        # y = np.argmax(Y, axis=-1)
        # print(y)

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

    elif (sys.argv[1] == 'simplernn'):
        hid_state_size = int(sys.argv[2])
        win_size = int(sys.argv[3])
        stride_size = int(sys.argv[4])
        temp = int(float(sys.argv[5]))

        file = 'beatles.txt'
        code = generate(file, win_size, stride_size)
        x, y = code.text_encoder()

        print(f' the shape of x: {x.shape} (num of sequences x window size x vocab size)')
        print(f' the shape of y: {y.shape} (num of sequences x window size x vocab size)')

        num_new_characters = 30
        mod, pred = simpleRNN_model(x, y, hid_state_size, num_new_characters, temp)

        # print predicted letters
        print(pred)
    # elif (sys.argv[1] == 'task1'):


