from library import differential_evolution as de
import os
import numpy as np
import keras
from keras.datasets import mnist
from datetime import datetime

num_classes = 10
max_generations = 50

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_mnist_trained_model.h5'

model = keras.models.load_model(save_dir + '\\' + model_name)


# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

OverallTimeTracker = datetime.now()


for j in range(1000):
    file_object = open('results_test_mnist.txt', 'a')
    for i in range(10):
        if i != y_test[j]:
            result = np.append(y_test[j]
                               , de.differential_evolution(model, i
                                                        , x_test[j]
                                                        , max_generations=
                                                        max_generations
                                                        , population_size=200))
            result = np.append(j, result)

            file_object.write('\n')
            file_object.write(np.array2string(result
                                              , precision=2
                                              , separator=','
                                              , suppress_small=True))
    print(j)
    file_object.close()

print("Overall time: " + str(datetime.now() - OverallTimeTracker))

