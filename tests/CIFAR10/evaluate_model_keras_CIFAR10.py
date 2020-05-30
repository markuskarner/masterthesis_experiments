import os
import keras
import tensorflow as tf
from keras.datasets import cifar10

num_classes = 10

save_dir = os.path.join(os.path.dirname(os.getcwd()), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

model = tf.keras.models.load_model(save_dir + '\\' + model_name)

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_test = keras.utils.to_categorical(y_test, num_classes)
y_train = keras.utils.to_categorical(y_train, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

#print(model.metrics_names)
print(model.summary())
