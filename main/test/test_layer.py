from main import layer as l
from . import utils

import unittest


class TestLayer(unittest.TestCase):

    def test_compute_output_different_neurons_weights_count(self):
        weights_count = [10, 5]
        neuron_count = [10, 10]
        bias = 1

        for wc, nc in zip(weights_count, neuron_count):
            inputs = utils.generate_array(wc)
            weights = utils.generate_one_layer_weights(wc, nc, bias)
            activation_functions = [utils.activation_function] * nc

            layer = l.Layer(weights, activation_functions)

            layer_output = layer.forward_propagate(inputs)
            expected_output = [wc + bias] * nc

            self.assertEqual(expected_output, layer_output)

    def test_backward_propagate_output(self):
        layer = l.Layer(weights=utils.generate_one_layer_weights(3, 3, 314),
                        activation_functions=range(3),
                        activation_function_derivatives=utils.generate_layer_activation_function_derivatives(3))
        actual_outputs = []

        for neuron in layer.get_neurons():
            neuron.set_output(1)
            actual_outputs.append(2)

        layer.backward_propagate_output(actual_outputs)

        for neuron in layer.get_neurons():
            self.assertEqual(1, neuron.get_delta())
