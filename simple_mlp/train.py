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

        return self.__dataset

    def setup_network(self, input_count, network_layout, activation_functions, activation_functions_derivations):
        network_weights = self.generate_random_network_weights(input_count,
                                                                    network_layout)

        network_activation_functions = activation_functions
        network_activation_functions_derivatives = activation_functions_derivations

        self.__network = n.Network(weights=network_weights,
                            activation_functions=network_activation_functions,
                            activation_functions_derivatives=network_activation_functions_derivatives)

        return self.__network

    def chu_chu(self, learning_rate, acceptable_squared_mean_error):
        squared_mean_error = acceptable_squared_mean_error + 1

        while True:
            iteration = 0
            for input, output in self.__dataset.items():
                self.__network.forward_propagate(input)
                self.__network.backwards_propagation(output)
                self.__network.adjust(learning_rate, input)

                squared_mean_error = self.__network.squared_mean_error(output)

                print("\nNetwork at iteration {} with squared mean error of {}.".format(iteration, squared_mean_error))
                self.__network.display()

                if squared_mean_error <= acceptable_squared_mean_error:
                    break
                    
                iteration += 1

            if squared_mean_error <= acceptable_squared_mean_error:
                break

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
