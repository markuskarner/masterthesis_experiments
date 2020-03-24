from matplotlib import pyplot as plt
import numpy as np
import keras
from keras.datasets import cifar10
from library import differential_evolution as de
num_classes = 10


def get_category_name(result_tuple):
    categories = {
        0: "airplane",
        1: "automobile",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck"
    }

    if hasattr(result_tuple, "__len__"):
        i = np.argmax(result_tuple)
    else:
        i = result_tuple

    return categories.get(i, "InvalidCategory")


# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_test = keras.utils.to_categorical(y_test, num_classes)

columns = 2
rows = 1

fig, axs = plt.subplots(rows, columns)

axs[0].imshow(x_test[1], interpolation='nearest')
axs[0].set_title(get_category_name(y_test[1]))

perturbed_image = de.add_perturbation_cifar10(np.array([[22, 5, 167, 222, 254]])
                                              , x_test[1])

axs[1].imshow(perturbed_image[0], interpolation='nearest')
axs[1].set_title(get_category_name(1))

plt.show()
