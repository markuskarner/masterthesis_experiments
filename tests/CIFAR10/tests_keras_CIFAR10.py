from library import differential_evolution as de
import os
import numpy as np
import keras
from keras.datasets import cifar10
from multiprocessing import Pool
import time

num_classes = 10
max_generations = 10

save_dir = os.path.join(os.path.dirname(os.getcwd()), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

model = keras.models.load_model(save_dir + '\\' + model_name)


# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_test = keras.utils.to_categorical(y_test, num_classes)


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

if __name__ == '__main__':
    p = Pool(processes=4)
    result = p.map(f, range(0, 101))
    print(result)

print(time.time()-start_time)
