import os
import numpy as np
import keras
import tensorflow as tf
import time
from library import differential_evolution as de
from library import ImagePickerCIFAR10
from keras.datasets import cifar10
from multiprocessing import Pool
import copy

#use cpu only
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

num_classes = 10
sample_size = 200
max_generations = 2
multiprocessing = True

save_dir = os.path.join(os.path.dirname(os.getcwd()), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

model = tf.keras.models.load_model(save_dir + '\\' + model_name)

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_test = keras.utils.to_categorical(y_test, num_classes)

# get random images with true predictions, equal number per class
# Use this block first and than copy the result into all_image_indices below, comment again before the run
#image_indices = ImagePickerCIFAR10.get_cifar10_images(model, sample_size, num_classes)
#print(image_indices)
#exit()

all_image_indices = np.array([5717, 5644, 3136])

#function only needed for multi processing
def f(j):
    print('start index: %d' % j)
    inner_start_time = time.time()
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

            with open('results.txt', 'ab') as file:
                np.savetxt(file, _result.reshape(1, _result.shape[0])
                           , fmt='%1.4f'
                           , delimiter=",", newline="\n")
    time.sleep(5)
    print(time.time() - inner_start_time)

np.set_printoptions(suppress=True) # to avoid e+01 etc...

start_time = time.time()

if multiprocessing:
    print('Using multiprocessing')
    if __name__ == '__main__':
        with Pool(processes=3) as p:
            p.map(f, all_image_indices)

else:
    print('start attack')
    for j in (all_image_indices):
        for i in range(10):
            if i != np.argmax(y_test[j]):
                start_time_inner = time.time()
                _result = de.differential_evolution(model
                                                    , i
                                                    , x_test[j]
                                                    , np.argmax(y_test[j])
                                                    , j
                                                    , max_generations
                                                    =max_generations)
                print(_result)
                print(time.time() - start_time_inner)

print(time.time()-start_time)
