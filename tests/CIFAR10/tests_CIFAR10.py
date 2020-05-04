from library import differential_evolution as de
import os
import numpy as np
import keras
from keras.datasets import cifar10

num_classes = 10
max_generations = 10

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

model = keras.models.load_model(save_dir + '\\' + model_name)


# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_test = keras.utils.to_categorical(y_test, num_classes)


for j in range(10, 20):
    print("start - j: %0d" % j)
    for i in range(10):
        if i != np.argmax(y_test[j]):
            print(de.differential_evolution(model, i, x_test[j]
                                            , max_generations=
                                                    max_generations))
    print("end - j: %0d" % j)

