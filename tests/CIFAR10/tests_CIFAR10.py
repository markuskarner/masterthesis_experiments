import os
# https://stackoverflow.com/questions/40690598/can-keras-with-tensorflow-backend-be-forced-to-use-cpu-or-gpu-at-will
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import keras
import time
from library import differential_evolution as de
from library import ImagePickerCIFAR10
from keras.datasets import cifar10
from multiprocessing import Pool


num_classes = 10
sample_size = 20
max_generations = 100
multiprocessing = True

save_dir = os.path.join(os.path.dirname(os.getcwd()), 'saved_models')
model_name = 'VGG16_one_pixel_cifar10_trained_model.h5'

model = keras.models.load_model(save_dir + '\\' + model_name)

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_test = keras.utils.to_categorical(y_test, num_classes)

# get 500 random images with true predictions, equal number per class
image_indices = ImagePickerCIFAR10.get_cifar10_images(model, sample_size, num_classes)


def f(j):
    for _i in range(10):
        if _i != np.argmax(y_test[j]):
            _result = de.differential_evolution(model
                                                , _i
                                                , x_test[j]
                                                , np.argmax(y_test[j])
                                                , j
                                                , max_generations
                                                =max_generations)
            print(_result)

start_time = time.time()

if multiprocessing:
    if __name__ == '__main__':
        p = Pool(processes=20)
        result = p.map(f, image_indices)
        print(result)
else:
    for j in image_indices:
        f(j)

print(time.time()-start_time)
