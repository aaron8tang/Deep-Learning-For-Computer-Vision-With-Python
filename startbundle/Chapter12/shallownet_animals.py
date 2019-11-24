# python shallownet_animals.py  --dataset ../datasets/animals/

import sys
from time import time

sys.path.append("..")
from startbundle.pyimagesearch.preprocessing import ImageToArrayPreprocessor
from startbundle.pyimagesearch.preprocessing import SimplePreprocessor
from startbundle.pyimagesearch.datasets import SimpleDatasetLoader
from startbundle.pyimagesearch.nn.conv import ShallowNet
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse

def shallownet_animals(dataset):
    # grab the list of images that we'll describing
    print("[INFO] loading images...")
    imagePaths = list(paths.list_images(dataset))


    # initialize the image preprocessor
    sp = SimplePreprocessor(32 ,32)
    iap = ImageToArrayPreprocessor()

    # load the dataset from disk then scale the raw pixel intensities
    sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
    (data, labels) = sdl.load(imagePaths, verbose=500)
    #labels = labels.reshape(1, -1)
    data = data.astype("float") / 255.0


    # pritition the into training and testing splits using 75% of
    # the data for training and the remaining 25% fir testing
    '''
    cat和dog的图片总数=512是，
    labels.shape=(512,)
    data.shape=(512, 32, 32, 3)
    trainX.shape=<class 'tuple'>: (384, 32, 32, 3)
    trainY.shape=<class 'tuple'>: (384, 1)
    testY.shape=<class 'tuple'>: (128, 1)
    '''
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

    # convert the labels from integers to vectors
    trainY = LabelBinarizer().fit_transform(trainY)
    testY = LabelBinarizer().fit_transform(testY)


    # initialize the optimizer and model
    print("[INFO] compiling model...")
    opt = SGD(lr=0.005)
    #model = ShallowNet.build(width=32, height=32, depth=3, classes=3)
    model = ShallowNet.build2(width=32, height=32, depth=3, classes=2)
    #model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    # train the network
    print("[INFO] training network...")
    H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=100, verbose=1)


    # evaluate the network
    print('[INFO]: Evaluating the network....')
    predictions = model.predict(testX, batch_size=32)
    #print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=['cat', 'dog', 'panda']))
    print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=['cat', 'dog']))
    # Plot the training loss and accuracy
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(np.arange(0, 100), H.history['loss'], label='train_loss')
    plt.plot(np.arange(0, 100), H.history['val_loss'], label='val_loss')
    plt.plot(np.arange(0, 100), H.history['acc'], label='train_acc')
    plt.plot(np.arange(0, 100), H.history['val_acc'], label='val_acc')
    plt.title('Training Loss and Accuracy')
    plt.xlabel('Epoch #')
    plt.ylabel('Loss/Accuracy')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    '''
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
    args = vars(ap.parse_args())
    
    dateset = args["dataset"]
    '''
    dateset = 'c:\\kuaipan\\cv-data\\dogs-vs-cats\\'
    start = time()
    shallownet_animals(dateset)
    end = time()
    print('function takes %f seconds.' % (end - start))








