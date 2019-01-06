import numpy as np
np.random.seed(1)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
# Load pre-shuffled MNIST data into train and test sets and set hop size
(X_train, y_train), (X_test, y_test) = mnist.load_data()
hop = 5; X_train = X_train[::hop,:,:]; y_train = y_train[::hop]
# Preprocess and normalize input data
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
# Preprocess class labels
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)
# CNN model architecture
model = Sequential()
model.add(Convolution2D(4, (3, 3), activation='relu', input_shape=(1,28,28),
                        data_format='channels_first'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
# Train model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(X_train, Y_train,
          batch_size=32, nb_epoch=10, verbose=1)
# Evaluate model on test data
score = model.evaluate(X_test, Y_test, verbose=0)
print("Evaluation score: {0:.3f}".format(float(score[1])))