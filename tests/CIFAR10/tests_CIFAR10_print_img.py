import os
from matplotlib import pyplot as plt
import numpy as np
import keras
from keras.datasets import cifar10
from library import differential_evolution as de
from library import attack_evaluation as atk
num_classes = 10

#Frog
#perturbation1 = np.array([8, 15, 217, 230, 144])
#perturbation2 = np.array([16, 12, 251, 194, 252])
#indexOfImage = 4

#Frog
perturbation1 = np.array([11, 24, 178, 114, 122])
perturbation2 = np.array([13, 13, 24, 60, 7])
indexOfImage = 12



# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_test = keras.utils.to_categorical(y_test, num_classes)

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

model = keras.models.load_model(save_dir + '\\' + model_name)

atk.print_attack_result(model, x_test[indexOfImage]
                        , y_test[indexOfImage]
                        , perturbation1
                        , perturbation2
                        )