import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.datasets import cifar10
from tensorflow.keras.applications.vgg16 import VGG16
import numpy as np
import os

batch_size = 256
num_classes = 10
epochs = 50
data_augmentation = False
num_predictions = 20
save_dir = os.path.join(os.path.dirname(os.getcwd()), 'tests')
save_dir = os.path.join(save_dir, 'saved_models')
model_name = 'alexnet_VGG16_trained_model.h5'

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)

model = VGG16(
    weights=None,
    include_top=True,
    classes=num_classes,
    input_shape=(32, 32, 3)
)

# Compile the model
model.compile(loss=keras.losses.categorical_crossentropy
              , optimizer='SGD'
              , metrics=["accuracy"])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train,
              y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.1,
              shuffle=True,
              verbose=1)

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
