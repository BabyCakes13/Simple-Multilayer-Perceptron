from main import network as n

from functools import reduce
import unittest


class TestNetwork(unittest.TestCase):

    def test_compute_output(self):
        input = TestNetwork.generate_one_array(2)
        network_layout = [3, 2]
        network_weights = TestNetwork.generate_one_network_weights(input_count=len(input),
                                                                    neuron_count_per_layer_list=network_layout)

        network_activation_functions = TestNetwork.generate_network_activation_functions(network_layout)

        network = n.Network(weights=network_weights,
                            activation_functions=network_activation_functions)

        outputs = network.compute_output(input)

        expected_output = reduce(lambda x, y: x*y, network_layout)
        expected_outputs = [expected_output] * network_layout[-1]

        self.assertEqual(expected_outputs, outputs)

    def generate_one_neuron_weights(weights_per_neuron_count):
        neuron_weights = [1] * weights_per_neuron_count
        return neuron_weights

    def generate_one_layer_weights(weights_per_neuron_count, neuron_count):
        layer_weights = [TestNetwork.generate_one_neuron_weights(weights_per_neuron_count)] * neuron_count
        return layer_weights

    def generate_one_network_weights(input_count, neuron_count_per_layer_list):
        network_weights = []
        for nc in neuron_count_per_layer_list:
            network_weights.append(TestNetwork.generate_one_layer_weights(weights_per_neuron_count=input_count,
                                                                          neuron_count=nc))
            input_count = nc

        return network_weights

    def generate_one_array(n):
        array = [1] * n
        return array

    def generate_one_matrix(weights_count, neuron_count):
        matrix = [TestNetwork.generate_one_array(weights_count)] * neuron_count
        return matrix

    def generate_layer_activation_functions(neuron_count):
        activation_functions = [TestNetwork.activation_function] * neuron_count
        return activation_functions

    def generate_network_activation_functions(network_layout):
        netwrok_activation_functions = []

        for neuron_count in network_layout:
            netwrok_activation_functions.append(TestNetwork.generate_layer_activation_functions(neuron_count))

        return netwrok_activation_functions

    def activation_function(x):
        return max(0, x)
