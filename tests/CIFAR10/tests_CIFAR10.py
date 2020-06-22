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
model_name = 'VGG16_one_pixel_cifar10_trained_model.h5'

model = tf.keras.models.load_model(save_dir + '\\' + model_name)

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_test = keras.utils.to_categorical(y_test, num_classes)

# get random images with true predictions, equal number per class
# Use this block first and than copy the result into all_image_indices below, comment again before the run
# image_indices = ImagePickerCIFAR10.get_cifar10_images(model, sample_size, num_classes)
# print(image_indices)
# exit()

all_image_indices = np.array([6422, 6333, 6744, 2492, 3152, 38, 5627, 5465, 7338, 8926, 3046, 6323, 8970, 4265
, 4419, 580, 7364, 3891, 3784, 2295, 5225, 6224, 3286, 4932, 5284, 389, 9691, 6415
, 1993, 7889, 7818, 6659, 3368, 9915, 2497, 9575, 9318, 9004, 9273, 8179, 8657, 1630
, 6563, 3113, 7488, 8697, 5495, 2602, 5040, 4015, 9063, 1060, 8083, 8703, 1613, 901
, 2698, 1419, 4826, 1816, 5277, 5133, 6270, 4619, 829, 9343, 3186, 8890, 4226, 6661
, 4690, 3140, 7749, 1778, 6751, 817, 1573, 5551, 4136, 5400, 5502, 809, 3121, 2866
, 2740, 8068, 2960, 3381, 4923, 2488, 3332, 6152, 6874, 7465, 1081, 1450, 2430, 8502
, 5996, 1093, 9494, 5970, 511, 6557, 5518, 2878, 6343, 8708, 5507, 6249, 4846, 8089
, 1354, 5309, 9444, 4465, 5890, 2336, 9685, 8881, 4505, 314, 9698, 4007, 7997, 3599
, 2669, 4187, 4330, 5676, 4231, 1929, 7577, 7521, 6617, 5828, 6093, 9673, 7325, 6250
, 1452, 7322, 1205, 3357, 6762, 8738, 7653, 4087, 9523, 8634, 6824, 5566, 636, 4009
, 8493, 3461, 2553, 9766, 9149, 9281, 8696, 4806, 9019, 2191, 288, 9841, 233, 3612
, 9544, 7393, 4713, 9950, 7563, 3198, 9295, 9763, 2605, 9158, 6438, 7247, 1061, 2214
, 3849, 4069, 706, 1443, 6889, 2053, 2925, 8919, 5962, 4078, 7121, 9294, 9202, 7222
, 6586, 6329, 4382, 2625, 7798, 2809, 820, 1379, 3271, 5165, 4401, 9797, 9736, 1694
, 3127, 8866, 9006, 7967, 4730, 1564, 884, 8510, 4025, 4770, 6548, 4070, 6167, 8082
, 2354, 588, 1606, 2689, 7519, 4091, 5909, 5925, 2076, 4170, 7699, 9927, 8343, 5653
, 8235, 6943, 9885, 8278, 4471, 7660, 3697, 5000, 6978, 2005, 4628, 9819, 1784, 5757
, 6225, 5303, 3756, 4075, 2812, 7469, 7757, 9697, 7587, 6000, 7270, 3676, 5966, 53
, 5980, 7390, 499, 6399, 934, 8791, 8247, 2043, 1851, 5744, 7151, 8740, 6526, 2051
, 7683, 5930, 1792, 5146, 3064, 2068, 2271, 9477, 2246, 7346, 7506, 9621, 1070, 7721
, 4567, 4318, 109, 7437, 8612, 7624, 2062, 3092, 9366, 2670, 9790, 5258, 7083, 8998
, 2090, 475, 1331, 8683, 8898, 1920, 9598, 4783, 1110, 2272, 772, 4714, 7823, 5834
, 6081, 6300, 2853, 9059, 7422, 9194, 736, 3244, 1524, 1410, 8111, 1659, 5015, 5336
, 5710, 3780, 2520, 8454, 8001, 5353, 3147, 4600, 8702, 4699, 3968, 7532, 430, 5124
, 8313, 633, 869, 3293, 6366, 6463, 2856, 1048, 3964, 3698, 7066, 6495, 2758, 5704
, 2445, 9703, 7879, 5304, 4839, 711, 6800, 3282, 4099, 6319, 1518, 4723, 5919, 9060
, 2094, 7326, 112, 4557, 9944, 9479, 4772, 8307, 927, 2225, 7459, 8072, 2752, 2112
, 3456, 7954, 2938, 5972, 141, 5535, 6002, 7927, 3614, 563, 2659, 3382, 6442, 6389
, 1688, 3048, 5354, 9664, 1106, 9938, 1345, 8473, 1857, 8021, 417, 246, 1002, 5584
, 3108, 8053, 6413, 245, 7542, 8074, 2077, 9592, 7167, 6815, 8389, 6348, 8873, 6131
, 343, 9345, 1169, 3927, 9237, 3356, 3515, 7296, 9142, 2212, 5633, 5491, 7794, 7027
, 6746, 8418, 5068, 8102, 716, 9613, 9243, 5357, 113, 5813, 8848, 7829, 903, 3188
, 1112, 3489, 8860, 5328, 3326, 1726, 1278, 4939, 6304, 2861, 6945, 1611, 661, 2173
, 2733, 6105, 6444, 4108, 1673, 5076, 5228, 4987, 933, 5779, 5707, 3620, 2618, 5062
, 5798, 496, 7125, 1579, 2388, 8795, 8909, 7869, 5948, 3177])

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
        with Pool(processes=4) as p:
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
