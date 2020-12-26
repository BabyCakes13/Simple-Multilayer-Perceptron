from main import layer as l
from . import utils

import unittest


class TestLayer(unittest.TestCase):

    def test_compute_output(self):
        weights_count = [10, 5]
        neuron_count = [10, 10]

        for wc, nc in zip(weights_count, neuron_count):
            inputs = utils.generate_one_array(wc)
            weights = utils.generate_one_matrix(wc, nc)
            activation_functions = [utils.activation_function] * nc

            layer = l.Layer(weights, activation_functions)

            layer_output = layer.compute_output(inputs)
            expected_output = [wc] * nc

            self.assertEqual(expected_output, layer_output)
