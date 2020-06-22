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
max_generations = 100
multiprocessing = True

save_dir = os.path.join(os.path.dirname(os.getcwd()), 'saved_models')
model_name = 'improved_keras_cifar10_trained_model.h5'

model = tf.keras.models.load_model(save_dir + '\\' + model_name)

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_test = keras.utils.to_categorical(y_test, num_classes)

# get random images with true predictions, equal number per class
# Use this block first and than copy the result into all_image_indices below, comment again before the run
#image_indices = ImagePickerCIFAR10.get_cifar10_images(model, sample_size, num_classes)
#print(image_indices)
#exit()

all_image_indices = np.array([20,1053,4120,724,1412,4084,2693,8955,2091,1648,51,7303,3145,7156
,4345,899,7724,7792,897,9919,9556,7912,330,9782,638,7232,1916,3888
,9471,1134,4181,2923,8440,1115,5169,4074,4589,7,1764,5816,3761,599
,6965,1438,2536,1270,5397,867,5036,7464,6851,7984,2598,1650,2908,4901
,2132,6013,6129,4991,8823,7793,7201,3100,9013,4716,7525,6812,4456,3292
,4212,1697,7764,3681,3022,1842,9355,4075,8821,3295,199,5965,4348,5186
,6115,8306,7332,6577,6714,726,3453,8540,3177,6502,3213,6480,5783,3333
,4478,681,3500,1534,82,8991,2188,7587,729,5113,3484,1653,9433,8884
,6505,7701,1046,891,4451,8917,7773,6113,4047,9885,8119,8362,1116,8489
,2875,3386,8767,6418,6090,8399,545,8359,916,3687,169,9167,1949,952
,4022,3239,7870,5391,7098,8353,5114,9621,1213,9108,8429,4312,1024,4368
,2483,6324,851,1537,5526,2266,1271,5625,3487,9719,7101,5760,8380,4508
,5142,661,8446,8263,7796,3656,8058,8574,4479,2742,9701,4041,7321,3892
,1877,2839,7872,5303,1426,4979,6889,2840,6878,7673,4997,3884,8526,1179
,4963,8096,176,4785])

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
        with Pool(processes=50) as p:
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
