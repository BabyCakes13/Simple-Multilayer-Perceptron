from main import network as n
from . import utils

from functools import reduce
import unittest


class TestNetwork(unittest.TestCase):

    def test_one_forward_propagate(self):
        input = utils.generate_array(2)
        network_layout = [3, 2]
        bias = 1

        network_weights = utils.generate_one_network_weights(input_count=len(input),
                                                             neuron_count_per_layer_list=network_layout,
                                                             bias=bias)

        network_activation_functions = utils.generate_network_activation_functions(network_layout)

        network = n.Network(weights=network_weights,
                            activation_functions=network_activation_functions)

        outputs = network.forward_propagate(input)

        input_hidden_network_laout = network_layout
        input_hidden_network_laout.insert(0, len(input))
        input_hidden_network_laout = input_hidden_network_laout[:-1]

        input = input[0]

        for nc in input_hidden_network_laout:
            expected_output = nc * input + bias
            input = expected_output

        # expected_output = reduce(lambda x, y: x*y + 1, network_layout)
        expected_outputs = [expected_output] * network_layout[-1]

        self.assertEqual(expected_outputs, outputs)

    def test_adjust(self):
        input = utils.generate_array(2)
        network_layout = [3, 2]
        output = 1
        delta = 1
        bias = 2
        learning_rate = 1
        adjustment = 1

        network_weights = utils.generate_one_network_weights(input_count=len(input),
                                                             neuron_count_per_layer_list=network_layout,
                                                             bias=bias)

        network_activation_functions = utils.generate_network_activation_functions(network_layout)

        network = n.Network(weights=network_weights,
                            activation_functions=network_activation_functions)

        for layer in network.get_layers():
            for neuron in layer.get_neurons():
                neuron.set_output(output)
                neuron.set_delta(delta)

        network.adjust(learning_rate, input)


        for layer, ilw in zip(network.get_layers(), network_weights):
            for neuron, inw in zip(layer.get_neurons(), ilw):
                inw_without_bias = inw[1:]
                for weight, initial_weight in zip(neuron.get_weights(), inw_without_bias):
                    adjustment = initial_weight + learning_rate * output * delta
                    self.assertEqual(adjustment, weight)
