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
sample_size = 500
max_generations = 4
population_size = 200
multiprocessing = True

save_dir = os.path.join(os.path.dirname(os.getcwd()), 'saved_models')
model_name = 'AllConv_cifar10_trained_model.h5'

model = tf.keras.models.load_model(save_dir + '\\' + model_name)

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_test = keras.utils.to_categorical(y_test, num_classes)

# get random images with true predictions, equal number per class
# Use this block first and than copy the result into all_image_indices below, comment again before the run
# otherwise each process would pick new images
#image_indices = ImagePickerCIFAR10.get_cifar10_images(model, sample_size, num_classes)
#print(image_indices)
#exit()

all_image_indices = np.array([6596, 8202, 3647, 3188, 9481, 799, 3483, 4517, 2735, 7023, 561, 2809, 7850, 8921
, 5018, 1055, 4684, 4049, 9840, 2582, 6846, 9943, 7869, 7256, 7006, 5494, 7093, 2290
, 7273, 7187, 5980, 3945, 9118, 4026, 9682, 8231, 9514, 8171, 7954, 4999, 3253, 9314
, 7430, 2184, 5400, 1054, 1688, 4055, 5094, 6794, 4687, 8155, 1892, 6988, 3539, 9588
, 3441, 9025, 2604, 8585, 8892, 3668, 7404, 5458, 8206, 8902, 5635, 4147, 550, 2935
, 2786, 9540, 4715, 7608, 1672, 78, 847, 6915, 2778, 2269, 304, 4798, 7130, 9631
, 6620, 529, 6735, 8230, 2850, 4458, 1921, 1651, 7838, 39, 2426, 3090, 9589, 2235
, 7968, 6890, 1752, 4862, 5397, 6221, 6780, 9980, 3829, 3189, 4422, 2928, 826, 2377
, 6496, 6758, 896, 5073, 278, 7895, 2654, 1844, 7775, 2640, 3169, 5769, 7692, 3896
, 3732, 410, 4928, 8758, 6560, 5207, 4065, 3674, 6288, 7330, 8705, 9136, 6205, 8465
, 5554, 661, 4952, 4368, 6866, 8435, 4899, 3676, 500, 9285, 3830, 9196, 8910, 6404
, 5030, 9494, 6483, 9016, 6268, 1874, 6431, 5540, 659, 3008, 3005, 9103, 4330, 6139
, 8495, 1235, 42, 3802, 3455, 5145, 1146, 1067, 8693, 7963, 7911, 8409, 800, 3392
, 4150, 8079, 3210, 1132, 4135, 2765, 6847, 2954, 4880, 2389, 8557, 2473, 421, 1017
, 8339, 2901, 3058, 854, 4581, 4625, 9406, 4217, 5363, 6897, 9284, 523, 537, 3765
, 9167, 8204, 192, 425, 377, 2348, 7394, 2297, 4051, 4395, 6421, 4640, 2830, 6175
, 696, 7135, 3580, 5026, 6319, 8455, 6297, 402, 3730, 3182, 8551, 955, 8068, 3522
, 4674, 5377, 8018, 8553, 4179, 7910, 3538, 1635, 3257, 1255, 6710, 216, 5868, 4221
, 5824, 9618, 8769, 4629, 4407, 1849, 6972, 5358, 6240, 126, 4120, 1719, 8637, 413
, 5884, 5690, 8327, 8767, 5096, 7208, 8500, 2820, 7542, 2583, 7687, 1289, 844, 813
, 7937, 8285, 6702, 7109, 9876, 5675, 9004, 2877, 999, 2086, 1070, 9948, 6677, 7995
, 6339, 1615, 7115, 7401, 6670, 1362, 6509, 858, 5906, 3530, 1372, 4575, 750, 4797
, 8626, 6426, 2504, 5550, 9484, 3302, 5001, 9366, 5323, 8232, 1187, 4117, 1498, 2282
, 2550, 1062, 8882, 6485, 9328, 2161, 9202, 6184, 4858, 4935, 3581, 8863, 1012, 3370
, 7613, 5885, 2460, 5623, 5373, 98, 9129, 3007, 7844, 9598, 1694, 5805, 9194, 4115
, 1260, 3160, 9000, 7964, 3109, 3797, 7188, 7529, 466, 967, 6603, 2591, 1610, 1497
, 6721, 6767, 9358, 7431, 231, 2527, 6549, 4130, 5042, 252, 6292, 8379, 8951, 2378
, 7965, 9543, 5447, 6183, 885, 2025, 3903, 9534, 6159, 752, 6632, 969, 7319, 1143
, 7240, 3724, 753, 30, 5423, 1115, 2688, 5750, 1511, 7282, 7081, 4724, 6440, 6307
, 2541, 3176, 6446, 8883, 2046, 3200, 181, 940, 6343, 67, 5827, 3235, 5850, 3930
, 6683, 9660, 3448, 9414, 8659, 570, 480, 4068, 4869, 6593, 9985, 3194, 791, 6278
, 8721, 3066, 6717, 6082, 9856, 7774, 608, 2976, 4845, 959, 2295, 7582, 3532, 2218
, 7392, 4116, 1512, 8960, 2404, 837, 8556, 1310, 2153, 8441, 6074, 6738, 2173, 4223
, 8707, 7165, 3526, 3799, 9656, 6202, 8572, 398, 742, 5852, 1274, 8800, 1266, 1993
, 6340, 2189, 9462, 3602, 1275, 821, 8897, 873, 6675, 9093, 9191, 8219, 5785, 121
, 2828, 2840, 8382, 5822, 5368, 2625, 3710, 5265, 4601, 8841])

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
                                                =max_generations
                                                , population_size
                                                =population_size)
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
                                                    =max_generations
                                                    , population_size
                                                    =population_size)
                print(_result)
                print(time.time() - start_time_inner)

print(time.time()-start_time)
