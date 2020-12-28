from . import network as n

import itertools
import functools
import random


class Train:

    def __init__(self):
        self.__dataset = None
        self.__network = None

    def generate_xor_dataset(self, input_size):
        xor_input_combinations = list(itertools.product([0, 1], repeat=input_size))
        self.__dataset = {}

        for combination in xor_input_combinations:
            self.__dataset[combination] = [functools.reduce(lambda x, y: x ^ y, combination)]
            # self.__dataset[combination].append(1 - self.__dataset[combination][0])

        return self.__dataset

    def set_network(self, network):
        self.__network = network

    def set_dataset(self, dataset):
        self.__dataset = dataset

    def setup_network(self, input_count, network_layout, activation_functions, activation_functions_derivations):
        network_weights = self.generate_random_network_weights(input_count,
                                                                    network_layout)

        network_activation_functions = activation_functions
        network_activation_functions_derivatives = activation_functions_derivations

        self.__network = n.Network(weights=network_weights,
                            activation_functions=network_activation_functions,
                            activation_functions_derivatives=network_activation_functions_derivatives)

        return self.__network

    def chu_chu(self, learning_rate, acceptable_squared_mean_error, max_epocs=None):
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

                if iteration % 1 == 0:
                        print("Computed outputs and expected outputs are {} and {} for the following inputs: {}".format(computed_output, output, input))

            all_squared_mean_errors = all_squared_mean_errors / len(self.__dataset)

            if iteration % 1 == 0:
                print("\nNetwork at iteration {} with all squared mean errors of {}.".format(iteration, all_squared_mean_errors))
                self.__network.display()

            iteration += 1

    def check_chu_chu(self):
        for input, output in self.__dataset.items():
            outputs = self.__network.forward_propagate(input)
            print("Outputs are {}".format(outputs))

    def display_network(self):
        self.__network.display()

    @staticmethod
    def generate_random_network_weights(input_count, neuron_count_per_layer) -> list:
        network_weights = []

        for nc in neuron_count_per_layer:
            layer_weights = Train.generate_random_layer_weights(weights_per_neuron_count=input_count,
                                                                  neuron_count=nc)
            network_weights.append(layer_weights)
            input_count = nc

        return network_weights

    @staticmethod
    def generate_random_layer_weights(weights_per_neuron_count, neuron_count) -> list:
        layer_weights = [Train.generate_random_neuron_weights(weights_per_neuron_count) for x in range(neuron_count)]

        return layer_weights

    @staticmethod
    def generate_random_neuron_weights(weights_per_neuron_count) -> list:
        # for now this generates only between 0 and 9
        neuron_weights = [random.random()*2 - 1 for x in range(weights_per_neuron_count + 1)]

        return neuron_weights
