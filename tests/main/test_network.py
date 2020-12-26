from main import network as n
from . import utils

from functools import reduce
import unittest


class TestNetwork(unittest.TestCase):

    def test_compute_output(self):
        input = utils.generate_one_array(2)
        network_layout = [3, 2]
        network_weights = utils.generate_one_network_weights(input_count=len(input),
                                                                    neuron_count_per_layer_list=network_layout)

        network_activation_functions = utils.generate_network_activation_functions(network_layout)

        network = n.Network(weights=network_weights,
                            activation_functions=network_activation_functions)

        outputs = network.compute_output(input)

        expected_output = reduce(lambda x, y: x*y, network_layout)
        expected_outputs = [expected_output] * network_layout[-1]

        self.assertEqual(expected_outputs, outputs)
