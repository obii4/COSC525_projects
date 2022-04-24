import sys
from sklearn.preprocessing import OneHotEncoder

def load_text(sample):
    with open(sample) as f:
        contents = f.read()
        f.close()
    return contents

def text_encoder(sample, win_size, stride_size):
    '''
    Encodes each character of text to unicode
    and defines y vector according to a user
    defined window and stride size.

    Input:
    sample -> text string
    win_size -> moving window size
    stride_size -> stride increments

    Output:
    x -> og uncaoded pairs
    y -> encoded letter vectors
    '''
    encoded = []
    for i in sample:
        convert = ord(i) # ord() accepts a string of length 1 as an
                         # argument and returns the unicode code point representation of the passed argument
        encoded.append(convert)

    og = [] #old
    en = [] #new
    for i in range(len(sample) - win_size + 1):
        og.append(sample[i: i + win_size + 1])

    x = og[::stride_size]

    for i in range(len(encoded) - win_size + 1):
        en.append(encoded[i: i + win_size + 1])

    y = en[::stride_size]

    return x, y

if __name__ == "__main__":
    # # Testing Code
    test = 'beatles.txt'
    u = load_text(test)

#method 1 tester
    sample = 'hello, how are you?'
    x, y = text_encoder(sample, 5, 3)

    for i in range(len(x)):
         print(f'x{i}: {x[i]}')

    for i in range(len(y)):
        print(f'y{i}: {y[i]}')


        #Write a method which given a text file name, a window size and a stride it creates the
        # training data to perform back propagation through time. First, encode each character
        # as a number. Then break the data into multiple sequences of length windowsize + 1,
        # with a moving window of size stride.




    # elif (sys.argv[1] == 'task1'):


