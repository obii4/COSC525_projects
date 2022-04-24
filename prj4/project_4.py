import sys
from sklearn.preprocessing import OneHotEncoder


if __name__ == "__main__":
    # # Testing Code
    if (sys.argv[1] == 'test'):
        with open('beatles.txt') as f:
            contents = f.read()
            f.close()

    #method 1
        sample = 'hello, how are you?'

        #encode each character as a number
        encoded = []
        for i in contents:
            convert = ord(i)
            encoded.append(convert)
        print(encoded[0:4])

    # elif (sys.argv[1] == 'task1'):


