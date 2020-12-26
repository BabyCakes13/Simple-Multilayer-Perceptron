from main import layer as l
from . import utils

import unittest


class TestLayer(unittest.TestCase):

    def test_compute_output_different_neurons_weights_count(self):
        weights_count = [10, 5]
        neuron_count = [10, 10]
        bias = 1

        for wc, nc in zip(weights_count, neuron_count):
            inputs = utils.generate_one_array(wc)
            weights = utils.generate_one_layer_weights(wc, nc, bias)
            activation_functions = [utils.activation_function] * nc

            layer = l.Layer(weights, activation_functions)

            layer_output = layer.compute_output(inputs)
            expected_output = [wc + bias] * nc

            self.assertEqual(expected_output, layer_output)
