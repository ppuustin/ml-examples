import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def do_plot(title, xlab, ylab, what, val=None):
    plt.figure(figsize=(10,5)) # <w,h>
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.plot(what, label=ylab)
    if val != None: plt.plot(val, label='val_'+ylab)
    plt.legend(loc='best')
    #plt.legend(['train', 'test'], loc='upper left')
    plt.grid(which='both')
    plt.show()
    plt.close()

def do_test(model, x_test, y_test):
    y_hats = model.predict(x_test)
    for i in range(10):
        y_hat = y_hats[i].argmax()
        plt.imshow(x_test[i], cmap='gray')
        plt.title(f'y={str(y_test[i])}, y_hat={y_hat}')
        plt.show()
        plt.close()

def get_model(model_name):
    if 'dense' in model_name:
        inputs = tf.keras.layers.Input(shape=(28, 28))
        x = tf.keras.layers.Flatten(input_shape=(28, 28))(inputs)
        x = tf.keras.layers.Dense(512, activation=tf.nn.relu)(x)
        x = tf.keras.layers.Dropout(0.2)(x)
    else:
        x_train = x_train.reshape(x_train.shape[0],28,28,1)
        x_test = x_test.reshape(x_test.shape[0],28,28,1)
        inputs = tf.keras.layers.Input(shape=(28, 28, 1))
        x = tf.keras.layers.Conv2D(32, (5,5), padding='same', activation='relu')(inputs)
        x = tf.keras.layers.MaxPool2D()(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)

    output = tf.keras.layers.Dense(10, activation=tf.nn.softmax)(x) # activation='softmax'
    model = tf.keras.models.Model([inputs], output) 
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
    return model

def main():
    model_name = 'dense' # cnn2d
    f = os.getcwd() + '/input/mnist.npz'
    mnist = tf.keras.datasets.mnist

    (x_train, y_train),(x_test, y_test) = mnist.load_data(path=f) # mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    print('train-set:', x_train.shape)
    print('test-set:', x_test[0].shape)

    model = get_model(model_name)
    history = model.fit(x_train, y_train, epochs=3, verbose=1, validation_split=0.1)
    hist_dict = history.history
    print('loss + metrics:', model.evaluate(x_test, y_test))
    print(hist_dict.keys())

    do_plot('accs', 'epoch', 'acc', hist_dict['acc'], hist_dict['val_acc'])
    do_test(model, x_test, y_test)

if __name__ == '__main__':
    main()

'''
'''