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

all_image_indices = np.array([5717, 5644, 3136, 6626, 5237, 4160, 9004, 6081, 2418, 6511, 8774, 1198, 1503, 6363
, 3651, 4238, 1010, 3468, 8230, 532, 2338, 9771, 8247, 548, 2965, 5345, 1641, 8050
, 8342, 5118, 4834, 9182, 6319, 6126, 9661, 5308, 8472, 3951, 8976, 5321, 3907, 4431
, 7655, 3604, 9128, 9656, 1578, 4657, 1168, 8260, 5170, 5083, 7642, 4863, 6084, 6435
, 7747, 1758, 8372, 146, 447, 6626, 4597, 2484, 2608, 5093, 6023, 8455, 8953, 6424
, 2817, 919, 4607, 5846, 3064, 999, 8423, 6318, 535, 7373, 3426, 8327, 7677, 5540
, 6840, 2369, 386, 9638, 9104, 9544, 4813, 6432, 9952, 8075, 6117, 9574, 7732, 7866
, 3328, 3744, 9432, 3712, 9632, 4344, 5765, 3695, 5380, 4274, 8446, 9139, 8253, 9707
, 2764, 8925, 6137, 2609, 4770, 6086, 2073, 6526, 5636, 359, 8184, 4664, 2317, 2281
, 5561, 8698, 3260, 2086, 9557, 3160, 2051, 97, 8425, 5957, 5231, 5694, 6562, 2873
, 637, 6286, 2864, 1000, 1448, 9589, 7105, 7917, 8402, 95, 8538, 8377, 3865, 5969
, 8894, 3071, 2742, 549, 4465, 5141, 3656, 1978, 8725, 3367, 2658, 7860, 6567, 5120
, 5577, 7746, 4165, 2717, 7490, 3206, 9841, 7812, 2787, 1113, 4979, 8890, 2807, 588
, 1914, 7105, 1750, 4324, 2503, 1894, 7932, 684, 3799, 3190, 8362, 2565, 176, 5176
, 8930, 1168, 6333, 1979])

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
    time.sleep(5)
    print(time.time() - inner_start_time)

np.set_printoptions(suppress=True) # to avoid e+01 etc...

start_time = time.time()

if multiprocessing:
    print('Using multiprocessing')
    if __name__ == '__main__':
        with Pool(processes=30) as p:
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
