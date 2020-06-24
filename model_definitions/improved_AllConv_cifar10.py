#parts from https://keras.io/examples/cifar10_cnn/

from __future__ import print_function

import time

import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, AveragePooling2D
import matplotlib.pyplot as plt
import os
from library import CustomStopper

batch_size = 256
num_classes = 10
epochs = 100
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.path.dirname(os.getcwd()), 'tests')
save_dir = os.path.join(save_dir, 'saved_models')
model_name = 'improved_AllConv_cifar10_trained_model.h5'

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


def _define_model():
    _model = Sequential()
    _model.add(Conv2D(filters=96, kernel_size=(3, 3), strides=(1, 1)
                      , padding='same'
                      , input_shape=x_train.shape[1:]
                      , activation='relu'))
    _model.add(AveragePooling2D(2, 2, padding='same'))
    _model.add(Conv2D(96, (3, 3), activation='relu', padding='same'))
    _model.add(Conv2D(96, (3, 3), strides=(2, 2), activation='relu', padding='same'))
    _model.add(Conv2D(192, (3, 3), activation='relu', padding='same'))
    _model.add(Conv2D(192, (3, 3), activation='relu', padding='same'))
    _model.add(AveragePooling2D(2, 2, padding='same'))
    _model.add(Dropout(0.3))
    _model.add(Conv2D(192, (3, 3), strides=(2, 2), activation='relu', padding='same'))
    _model.add(Conv2D(192, (3, 3), strides=(2, 2), activation='relu', padding='same'))
    _model.add(Conv2D(192, (1, 1), activation='relu', padding='same'))
    _model.add(Conv2D(10, (1, 1), activation='relu', padding='same'))
    _model.add(AveragePooling2D(6, 1, padding='same'))
    _model.add(Flatten())
    _model.add(Dense(num_classes))
    _model.add(Activation('softmax'))

    return _model


# initialize optimizer & early stopping
opt = keras.optimizers.SGD(lr=1e-2, momentum=0.9, decay=1e-2 / epochs)
callback = CustomStopper.CustomStopper(monitor='val_loss', patience=2, verbose=1, start_epoch=10)

model = _define_model()
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print(model.summary())

#exit() # uncomment if you want to print the model summary only before the start of the training

start_time = time.time()

if not data_augmentation:
    print('Not using data augmentation.')
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=0.1,
                        shuffle=True,
                        #callbacks=[callback],
                        verbose=1)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    history = model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        workers=4)

print(time.time()-start_time)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

# list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(os.path.join('model_history', 'improved_AllConv_CIFAR10_model_accuracy.png'))
plt.clf()  # clear plot

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(os.path.join('model_history', 'improved_AllConv_CIFAR10_model_loss.png'))
