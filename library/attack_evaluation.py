import numpy as np
from matplotlib import pyplot as plt
from library import differential_evolution as de

def get_cifar10_category_name(result_tuple):
    """This function returns the category name from the cifar10 image dataset
    based on the a result tuple (array length 10) or an integer value

    Parameters:

        :param result_tuple: array length 10 OR int
            Result from model.predict on a cifar10 image or an integer

    Returns:

        :return: string
                0: "airplane"
                1: "automobile"
                2: "bird"
                3: "cat"
                4: "deer"
                5: "dog"
                6: "frog"
                7: "horse"
                8: "ship"
                9: "truck"
            Either from the category with the highest probability
            or from the integer
    """
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


def print_attack_result(model, original_image, y_original_image, perturbation1, perturbation2=None, dataset_name='cifar10'):
    plt.rcParams.update({'font.size': 15})
    columns = 2
    rows = 1

    font = {'weight': 'bold',
            'size': 30}

    if perturbation2 is not None:
        columns = 3

    fig, axs = plt.subplots(rows, columns)

    axs[0].imshow(original_image, interpolation='nearest', cmap='gray')
    if dataset_name == 'cifar10':
        axs[0].set_title(get_cifar10_category_name(y_original_image), **font)
    else:
        axs[0].set_title(y_original_image, **font)
    perturbed_image = de.add_perturbation(
        np.expand_dims(np.array(perturbation1), axis=0), original_image)

    if dataset_name == 'cifar10':
        prediction = model.predict(perturbed_image)
    else:
        prediction = model.predict(np.expand_dims(perturbed_image, axis=3))

    axs[1].imshow(perturbed_image[0], interpolation='nearest', cmap='gray')
    if dataset_name == 'cifar10':
        axs[1].set_title(get_cifar10_category_name(prediction), **font)
    else:
        axs[1].set_title(np.argmax(prediction), **font)

    if perturbation2 is not None:
        perturbed_image = de.add_perturbation(
            np.expand_dims(np.array(perturbation2), axis=0), original_image)

        if dataset_name == 'cifar10':
            prediction = model.predict(perturbed_image)
        else:
            prediction = model.predict(np.expand_dims(perturbed_image, axis=3))

        axs[2].imshow(perturbed_image[0], interpolation='nearest', cmap='gray')
        if dataset_name == 'cifar10':
            axs[2].set_title(get_cifar10_category_name(prediction), **font)
        else:
            axs[2].set_title(np.argmax(prediction), **font)

    plt.show()


