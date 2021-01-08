"""Module which implements the train system."""

from . import network as n

import itertools
import functools
import random


class Train:
    """Class which implements the neural network training."""

    def __init__(self):
        """Initialise the items needed for training."""
        self.__dataset = None
        self.__network = None

    def generate_xor_dataset(self, input_size):
        """
        Generate an XOR dataset for the training.

        This generates a 2^(input_size) data set of xor combinations.
        """
        xor_input_combinations = list(itertools.product([0, 1],
                                      repeat=input_size))
        self.__dataset = {}

        # set the output for supervised learning
        for combination in xor_input_combinations:
            self.__dataset[combination] = \
                [functools.reduce(lambda x, y: x ^ y, combination)]

        return self.__dataset

    def set_network(self, network):
        """Set the network for the training."""
        self.__network = network

    def get_dataset(self):
        """Return the set dataset."""
        if self.__dataset:
            return self.__dataset
        else:
            print("The dataset is not set.")
            return None

    def set_dataset(self, dataset):
        """Set the dataset for the training."""
        self.__dataset = dataset

    def setup_network(self,
                      input_count,
                      network_layout,
                      activation_functions,
                      activation_functions_derivations):
        """
        Set up the network for the training.

        In order to generate the network with random weights, we need to know
        the network layout and the number of inputs for the layout.
        """
        network_weights = self.generate_random_network_weights(input_count,
                                                               network_layout)

        network_activation_functions = activation_functions
        network_activation_functions_derivatives = \
            activation_functions_derivations

        self.__network = \
            n.Network(network_weights,
                      network_activation_functions,
                      network_activation_functions_derivatives)

        return self.__network

    def chu_chu(self,
                learning_rate,
                acceptable_squared_mean_error,
                max_epocs=None):
        """
        Train the network.

        For the training we need the learning rate, the acceptable squared mean
        error which will be set as a goal for the network. When the squared
        mean error of the network is less or equal than this, the training is
        considered succesfull and stopps. The max_epocs parameters is needed in
        case we don't want necessarly to achieve the goal mean squared error,
        rather have our network run for a specified number of iterations.
        """
        # This is needed only to ensure that the we get into the while.
        all_squared_mean_errors = acceptable_squared_mean_error + 1

        iteration = 0
        while (all_squared_mean_errors > acceptable_squared_mean_error
                and ((max_epocs is None) or iteration < max_epocs)):

            all_squared_mean_errors = 0

            for input, output in self.__dataset.items():
                computed_output = self.__network.forward_propagate(input)
                self.__network.backwards_propagation(output)
                self.__network.adjust(learning_rate, input)

                squared_mean_error = self.__network.squared_mean_error(output)
                all_squared_mean_errors += squared_mean_error

                # This is needed only for limiting the number of prints.
                if iteration % 1 == 0:
                    print("Computed outputs and expected outputs are {} and {}"
                          "for the following inputs: {}"
                          .format(computed_output, output, input))

            # Put the mean in squared mean error.
            all_squared_mean_errors = \
                all_squared_mean_errors / len(self.__dataset)

            # This is needed only for limiting the number of prints.
            if iteration % 1 == 0:
                print("\nNetwork at iteration {} with all squared mean errors"
                      "of {}.".format(iteration, all_squared_mean_errors))
                self.__network.display()

            iteration += 1

    def check_chu_chu(self):
        """
        Fowrard propagate the dataset through the network.

        After the training, check the results of the trained network on the
        used dataset.
        """
        for input, output in self.__dataset.items():
            outputs = self.__network.forward_propagate(input)
            print("Outputs are {}".format(outputs))

    def display_network(self):
        """Display network."""
        self.__network.display()

    @staticmethod
    def generate_random_network_weights(input_count,
                                        neuron_count_per_layer) -> list:
        """
        Generate random network weights for network initialisation.

        In order to set up the initial random network weights, the number of
        inputs is needed for the first layer. The network weights are composed
        of an array for each layer, each layer has an array of neurons, and
        each neuron has an array of weights.
        """
        network_weights = []

        for nc in neuron_count_per_layer:
            layer_weights = Train.generate_random_layer_weights(
                            weights_per_neuron_count=input_count,
                            neuron_count=nc)
            network_weights.append(layer_weights)
            input_count = nc

        return network_weights

    @staticmethod
    def generate_random_layer_weights(weights_per_neuron_count,
                                      neuron_count) -> list:
        """
        Generate random layer weights for network initialisation.

        Each layer will have a 2D weights form. Each layer has an array of
        neurons, and each neuron has an array of weights.
        """
        layer_weights = \
            [Train.generate_random_neuron_weights(weights_per_neuron_count)
                for x in range(neuron_count)]

        return layer_weights

    @staticmethod
    def generate_random_neuron_weights(weights_per_neuron_count) -> list:
        """
        Generate random neuron weights for network initialisation.

        Each neuron has an array of weights which represents a tie with another
        neuron.
        """
        # For now this generates only between 0 and 9.
        neuron_weights = \
            [random.random()*2-1 for x in range(weights_per_neuron_count + 1)]

        return neuron_weights
