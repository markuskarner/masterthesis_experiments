import os
import numpy as np
import keras
import tensorflow as tf
import time
from library import differential_evolution as de
from library import ImagePickerCIFAR10
from library import multi_gpu
from keras.datasets import cifar10



num_classes = 10
sample_size = 10000
max_generations = 100
multiprocessing = False
multi_gpus = True

save_dir = os.path.join(os.path.dirname(os.getcwd()), 'saved_models')
model_name = 'VGG16_one_pixel_cifar10_trained_model.h5'

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_test = keras.utils.to_categorical(y_test, num_classes)


start_time = time.time()

if multi_gpus:
    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))

    # Open a strategy scope.
    with strategy.scope():
        model = tf.keras.models.load_model(save_dir + '\\' + model_name)

else:
    model = tf.keras.models.load_model(save_dir + '\\' + model_name)

for i in range(sample_size):
    prediction = model.predict(np.expand_dims(x_test[i],axis=0))
    prediction = np.argmax(prediction)
    print('Index: %5d, Prediction: %2d' % (i, prediction))



print(time.time()-start_time)