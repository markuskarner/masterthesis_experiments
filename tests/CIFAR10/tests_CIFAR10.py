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
model_name = 'improved_AllConv_cifar10_trained_model.h5'

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

all_image_indices = np.array([1295, 3548, 1086, 936, 2700, 1314, 921, 1198, 4911, 2407, 7969, 5819, 6810, 8494
, 9922, 8740, 3099, 3369, 5024, 1993, 6499, 1396, 5660, 897, 1673, 7653, 7643, 5964
, 6281, 2736, 5723, 648, 7423, 3437, 2695, 8073, 1668, 5125, 7942, 1173, 9823, 557
, 8, 2512, 9864, 9332, 7997, 5960, 5414, 1800, 4082, 8184, 4832, 6601, 2212, 4395
, 6246, 3001, 8215, 6302, 9093, 6007, 5205, 6998, 7277, 1774, 741, 7100, 48, 4373
, 5669, 4855, 4225, 9184, 9423, 6592, 8270, 8339, 368, 999, 9767, 6891, 2333, 9028
, 8054, 4511, 4768, 6032, 9533, 5665, 7187, 4872, 4012, 8675, 2583, 7832, 396, 4403
, 9420, 9402, 5001, 3767, 1365, 3253, 6189, 7405, 4642, 8290, 9519, 9365, 2698, 9320
, 2607, 5051, 7734, 1855, 1438, 4863, 4494, 5297, 8052, 2211, 1539, 8146, 5427, 4324
, 5443, 3590, 6286, 9775, 3519, 5507, 8829, 6095, 7765, 6048, 5829, 7232, 9685, 1815
, 7111, 212, 9995, 3749, 5704, 6840, 3762, 291, 6034, 4362, 5593, 6528, 7376, 8623
, 7144, 5333, 3045, 8777, 2797, 8090, 1631, 8744, 5602, 7660, 3277, 2881, 3633, 6472
, 5120, 6479, 8257, 5032, 6117, 5052, 4453, 4998, 4817, 6631, 4551, 6815, 8505, 4737
, 8937, 9827, 1445, 8674, 3487, 9143, 1637, 6799, 7639, 7499, 3899, 63, 9459, 1485
, 2222, 6870, 2124, 9748, 5281, 7419, 8425, 1579, 8558, 1192, 2980, 3442, 739, 402
, 6346, 4731, 5538, 2742, 5026, 880, 616, 9133, 546, 9453, 9784, 9755, 1467, 9164
, 6379, 3810, 7332, 1874, 4083, 2237, 4664, 1661, 7986, 3720, 8768, 9205, 2249, 5067
, 8239, 3769, 9495, 8477, 6661, 6911, 949, 9594, 9667, 8813, 8774, 9513, 3515, 1142
, 937, 7413, 5539, 601, 1310, 3643, 8002, 2486, 3469, 4418, 2615, 3522, 3123, 8969
, 2397, 1092, 2593, 841, 3179, 6550, 8084, 4374, 7132, 2058, 6715, 9764, 4498, 3634
, 9282, 1405, 1802, 9276, 1914, 179, 3191, 545, 8285, 7153, 2353, 8971, 3946, 3702
, 3186, 5308, 4161, 3351, 2880, 2809, 1614, 6797, 7124, 7646, 2733, 489, 8214, 1660
, 6139, 1580, 5329, 9634, 847, 604, 9166, 1839, 6450, 6642, 5839, 8440, 963, 2102
, 6465, 6963, 181, 612, 8761, 2539, 511, 4328, 4861, 4588, 8302, 3106, 1250, 5126
, 1022, 6181, 4813, 4238, 4881, 1141, 1949, 1260, 2196, 5442, 4957, 7264, 8719, 9990
, 6735, 7863, 6514, 8848, 7912, 9075, 3781, 7424, 5314, 5845, 3831, 7268, 7315, 5135
, 2609, 3020, 778, 1682, 653, 4158, 9994, 7058, 4769, 5081, 7712, 2418, 2454, 200
, 5270, 3689, 2982, 1709, 8241, 1409, 9726, 9425, 816, 8543, 6420, 6925, 7000, 7346
, 296, 1985, 7114, 3074, 7593, 7594, 308, 1951, 8537, 493, 7318, 8547, 3719, 8722
, 4806, 3319, 4359, 5802, 4037, 1174, 1466, 621, 390, 3553, 4703, 6711, 5453, 9171
, 1541, 5133, 8700, 6856, 5757, 5680, 4386, 5894, 1195, 3334, 6216, 5823, 8247, 4147
, 1931, 7740, 1665, 2563, 6746, 4185, 6240, 6720, 7975, 5689, 7090, 7617, 6199, 8514
, 1088, 3391, 2303, 971, 5376, 9381, 3178, 7651, 8989, 7255, 6978, 8206, 6078, 6263
, 4435, 2845, 5482, 7804, 2523, 797, 6534, 466, 6664, 1971, 8288, 2623, 2660, 2917
, 3090, 1074, 4394, 2815, 7160, 3381, 6941, 2243, 8012, 628, 464, 7970, 4958, 8914
, 7807, 6525, 6624, 3183, 1871, 4458, 4963, 3929, 0, 2643])

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
