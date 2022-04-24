import sys
from sklearn.preprocessing import OneHotEncoder


class Method1:
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
        oHenc_x = OneHotEncoder()
        oHenc_x.fit(x)
        x_oneHOT = oHenc_x.transform(x).toarray()
        # print(oHenc_x.categories_) #if you want to view the encoded categories

        oHenc_y = OneHotEncoder()
        oHenc_y.fit(y)
        y_oneHOT = oHenc_y.transform(y).toarray()
        #print(oHenc_y.categories_) #if you want to view the encoded categories
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

        x_oneHOT_out, y_oneHOT_out = self.x_y_one_hot(x, y)

        return x_oneHOT_out, y_oneHOT_out




if __name__ == "__main__":
    # # Testing Code
    test = 'beatles.txt'
    code = Method1(test, 5, 3)
    x,y = code.text_encoder()



#method 1 tester *** output his sample ***

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


    # elif (sys.argv[1] == 'task1'):


