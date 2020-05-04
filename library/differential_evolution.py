import numpy as np
from datetime import datetime


def initiate_population(population_size: int = 400
                        , x_dim: int = 32
                        , y_dim: int = 32
                        , color_scheme: str = 'RGB'):
    """Function to initiate a population of image perturbations for CIFAR-10

    The Function will return an array of one-pixel perturbations
    for the CIFAR-10 image dataset.
    Each perturbation consists of the following 5 Elements
        - Random x coordinate of the Image, U[0,32]
        - Random y coordinate of the Image, U[0,32]
        - Random red color code, U[0,256]
        - Random green color code, U[0,256]
        - Random blue color code, U[0,256]

    Parameters:
        :param population_size: int, optional
            desired population size (default is 400)
        :param x_dim: int, optional
            x dimension of images (default is 32)
        :param y_dim: int, optional
            y dimension of images (default is 32)
        :param color_scheme: str, optional
            color_scheme for population, either 'RGB' or 'bw' for black-white
            (default is 'RGB')

    Return:
        :return array of one-pixel perturbations for images in color-scheme
            RGB or bw (black and white)
            if color_scheme input is neither RGB nor bw return -1
    """
    population = []

    for i in range(population_size):
        random_x = np.random.randint(0, x_dim)
        random_y = np.random.randint(0, y_dim)

        if color_scheme == 'RGB':
            random_r = np.random.randint(0, 256)
            random_g = np.random.randint(0, 256)
            random_b = np.random.randint(0, 256)

            v = [random_x, random_y, random_r, random_b, random_g]

        elif color_scheme == 'bw':
            random_bw = np.random.randint(0, 256)

            v = [random_x, random_y, random_bw]

        else:
            return -1

        population.append(v)

    return np.asarray(population)


def mutate_population(population
                      , v_min=np.zeros(5, int)
                      , v_max=np.array([32, 32, 256, 256, 256])
                      , f=0.5):
    """Function to mutate a population with differential evolution

    The Function mutates the input population according to the formular:
        population[x1] + f * (population[x2] - population[x3])
    Each perturbation of the population is mutated (x1) with two
    perturbations x2 and x3 where x1 != x2 != x3
    f is a scaleparameter between 0 and 1

    A new population of the same population size will be returned

    Parameters:
        :param population:
            a population of perturbations that should be mutated
        :param f: float between 0 and 1, optional
            scaleparameter (default is 0.5)
        :param v_min: array with same length as perturbation, optional
            vector with minimum values of each perturbation element
            (default is [0, 0, 0, 0, 0])
        :param v_max: array with same length as perturbation, optional
            vector with maximum values of each perturbation element
            (default is [32, 32, 256, 256, 256])
    Return:
        :return: new population of same size
    """
    n_population, n_dimension = population.shape
    new_population = []

    for i in range(n_population):
        x1, x2, x3 = i, i, i

        while x2 == i:
            x2 = np.random.randint(n_population)
        while x3 == i or x3 == x2:
            x3 = np.random.randint(n_population)

        v = population[x1] + f * (population[x2] - population[x3])

        for j in range(n_dimension):
            if v[j] < v_min[j]:
                # to avoid lower and upper boundary is the same
                if v_min[j] == population[x1][j]:
                    v[j] = v_min[j]
                else:
                    v[j] = np.random.randint(v_min[j], population[x1][j])
            # -1 added, as 32 is already out of bounds
            if v[j] > v_max[j]-1:
                v[j] = np.random.randint(population[x1][j], v_max[j])

        new_population.append(v.astype(int))

    return np.asarray(new_population)


def add_perturbation(population, original_image):
    """Function to add perturbations of a population to an image

    This function adds all perturbations of a population to one image
    and then returns an array of images with the size of the population

    Parameters:

        :param population:
            a population of perturbations that should be added to an image
        :param original_image: array
            array with shape [32, 32 ,3] if image from the cifar10 dataset
            array with shape [28, 28] if image from MNIST

    Returns:

        :return: an array of perturbated images based on the population
            -1 if original image neither cifar10 nor mnist

    """
    perturbed_images = []
    n_population = population.shape[0]
    rgb_bw = original_image.shape

    for i in range(n_population):
        x = population[i][0]
        y = population[i][1]

        perturbed_image = np.copy(original_image)
        if rgb_bw == (32, 32, 3):
            perturbed_image[x][y] = population[i][2: 5]
        elif rgb_bw == (28, 28):
            perturbed_image[x][y] = population[i][2]
        else:
            return -1

        perturbed_images.append(perturbed_image)

    return np.asarray(perturbed_images)


def evaluate_fitness(model, target_category, image, population=None):
    """Function to evaluate the fitness of an image or the fitness of
    a population of perturbations against one image (CIFAR-10)

    The fitness is simply the confidence of the target_category

    Parameters:

        :param model:
            a model that allows image classification
        :param target_category: int, 0-9
            the target category to calculate the fitness, represents one of
            the 10 classes in CIFAR-10
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
        :param image: array with shape [32,32,3]
            image from CIFAR-10 dataset
        :param population: optional
            a population of perturbations

    Returns

        :return: depends on the input. Without a population the function
            returns the fitness of the image. With a population the function
            returns an array holding the the fitness of each perturbed image
            that can be created with the population and the original image.
    """
    if population is not None:
        population_fitness = []
        num_population = population.shape[0]

        perturbated_images = add_perturbation(population
                                              , original_image=image)

        for i in range(num_population):

            _image = np.expand_dims(perturbated_images[i], axis=0)
            if _image.shape == (1, 28, 28):
                _image = np.expand_dims(_image, axis=3)

            prediction = model.predict(_image)
            fitness = prediction[0][target_category]
            population_fitness.append(fitness)

        return np.asarray(population_fitness)

    else:
        prediction = model.predict(np.expand_dims(image, axis=0))
        return prediction[0][target_category]


def differential_evolution(model, target_category, original_image
                           , population_size: int=400
                           , max_generations: int=100
                           , early_stopping: bool=True
                           , early_stopping_threshold: float=0.9):
    """This function tries to find one-pixel perturbations on the CIFAR-10
       Dataset that trick the model so that it perdicts the target category
       instead of the original (mostly true) category

    Parameters:

        :param model:
            a model, in this case mostly CNNs
        :param target_category: int, 0-9
                the target category to calculate the fitness, represents one of
                the 10 classes in CIFAR-10
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
        :param original_image: array with shape [32,32,3]
            image from CIFAR-10 dataset
        :param population_size: int, optional
            desired population size (default is 400)
        :param max_generations: int, optional
            desired number of generations (default is 100)
        :param early_stopping: bool, optional
            set early stopping on or of
        :param early_stopping_threshold: float 0-1, optional
            if early stopping is true a threshold can be set

    Returns:

        :return:
    """

    #OverallTimeTracker = datetime.now()
    rgb_bw = original_image.shape

    if rgb_bw == (32, 32, 3):
        population = initiate_population(population_size)
    elif rgb_bw == (28, 28):
        population = initiate_population(population_size, 28, 28, 'bw')

    for g in range(max_generations):
        if rgb_bw == (32, 32, 3):
            mutated_population = mutate_population(population)
        elif rgb_bw == (28, 28):
            mutated_population = mutate_population(population
                                                   , v_min=np.zeros(3)
                                                   , v_max=[28, 28, 256]
                                                   )
        ##
        timetracker = datetime.now()
        ##
        fitness_population = evaluate_fitness(model
                                              , target_category
                                              , original_image
                                              , population)

        fitness_mut_population = evaluate_fitness(model
                                                  , target_category
                                                  , original_image
                                                  , mutated_population)

        #Replace old generation
        for i in range(population_size):
            if fitness_mut_population[i] >= fitness_population [i]:
                population[i] = mutated_population[i]

        #calculate best fitness of generation, take fitness
        #of mutated population as it always replaces the old population
        #if it is better or equal (this avoids calculating the fitness of
        #the next generation at this point)

        index_argmax = np.argmax(fitness_mut_population)
        max_fitness = fitness_mut_population[index_argmax]

        #print("Generation %0d. Maximum fitness: %4d" % (g, max_fitness))
        # early_stopping
        if early_stopping:
            if max_fitness >= early_stopping_threshold:
                print("early stopping :)")
                break

    #print("Overall time: " + str(datetime.now() - OverallTimeTracker))

    return_array = np.append(population[index_argmax]
                             , np.array(target_category))
    return_array = np.append(return_array, np.array(max_fitness))

    return return_array