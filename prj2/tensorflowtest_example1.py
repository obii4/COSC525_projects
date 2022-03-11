import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from parameters import generateExample1

def example1():
    #Create a feed forward network
    model=Sequential()

    # Add convolutional layers, flatten, and fully connected layer
    model.add(layers.Conv2D(1,3,input_shape=(5,5,1),activation='sigmoid'))
    model.add(layers.Flatten())
    model.add(layers.Dense(1,activation='sigmoid'))

    # Call weight/data generating function
    l1k1,l1b1,l2,l2b,input,output = generateExample1()
    #Set weights to desired values

    #setting weights and bias of first layer.
    l1k1=l1k1.reshape(3,3,1,1)

    w1=l1k1
    model.layers[0].set_weights([w1,np.array([l1b1[0]])]) #Shape of weight matrix is (w,h,input_channels,kernels)

    #setting weights and bias of fully connected layer.
    model.layers[2].set_weights([np.transpose(l2),l2b])

    #Setting input. Tensor flow is expecting a 4d array since the first dimension is the batch size (here we set it to one), and third dimension is channels
    img=np.expand_dims(input,axis=(0,3))


    #print needed values.
    np.set_printoptions(precision=5)
    print('**************************************************************************************************')
    print('TENSOR FLOW OUTPUT')
    print('**************************************************************************************************')
    print('model output before:')
    print(model.predict(img))
    sgd = optimizers.SGD(learning_rate=100)
    model.compile(loss='MSE', optimizer=sgd, metrics=['accuracy'])
    history=model.fit(img,output,batch_size=1,epochs=1)

    print('model output after:')
    print(model.predict(img))

    print('1st convolutional layer, 1st kernel weights:')
    print(np.squeeze(model.get_weights()[0][:,:,0,0]))
    print('1st convolutional layer, 1st kernel bias:')
    print(np.squeeze(model.get_weights()[1][0]))


    print('fully connected layer weights:')
    print(np.squeeze(model.get_weights()[2]))
    print('fully connected layer bias:')
    print(np.squeeze(model.get_weights()[3]))
    print('**************************************************************************************************')


