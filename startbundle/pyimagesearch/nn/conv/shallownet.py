from keras.layers import MaxPooling2D
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K

class ShallowNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model along with the input shape to be
        # 'channels last'
        model = Sequential()
        inputShape = (height, width, depth)

        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        # define the first (and only) CONV => RELU layer
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))

        # softmax classifer
        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        print(model.summary())
        return model

    @staticmethod
    def build2(width, height, depth, classes):
        # initialize the model along with the input shape to be
        # 'channels last'
        inputShape = (height, width, depth)

        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        model = Sequential()

        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                         input_shape=inputShape))
        model.add(Flatten())
        '''

        model.add(Dense(1000, activation='relu'))
        '''
        model.add(Dense(1, activation='softmax'))

        print(model.summary())

        return model
