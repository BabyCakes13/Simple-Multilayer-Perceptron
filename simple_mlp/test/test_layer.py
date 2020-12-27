from simple_mlp import layer as l
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

    def test_backward_propagate_hidden(self):
        # generate the before layer which will be asked its errors
        before_layer = l.Layer(weights=utils.generate_one_layer_weights(3, 3, 1),
                        activation_functions=utils.generate_layer_activation_functions(3),
                        activation_function_derivatives=utils.generate_layer_activation_function_derivatives(3))

        for neuron in before_layer.get_neurons():
            neuron.set_delta(1)

        # generate the current hidden layer we are working with
        layer = l.Layer(weights=utils.generate_one_layer_weights(1, 3, 1),
                        activation_functions=utils.generate_layer_activation_functions(3),
                        activation_function_derivatives=utils.generate_layer_activation_function_derivatives(3))

        for neuron in layer.get_neurons():
            neuron.set_output(1)

        layer.backward_propagate_hidden(before_layer)

        for neuron in layer.get_neurons():
            self.assertEqual(3, neuron.get_delta())

    def test_adjust(self):
        weights_per_neuron_count = 2
        before_neuron_count = 3
        current_neuron_count = 5
        bias = 1

        before_layer = l.Layer(weights=utils.generate_one_layer_weights(weights_per_neuron_count,
                                                                        before_neuron_count,
                                                                        bias),
                        activation_functions=utils.generate_layer_activation_functions(before_neuron_count),
                        activation_function_derivatives=utils.generate_layer_activation_function_derivatives(before_neuron_count))

        for neuron in before_layer.get_neurons():
            neuron.set_output(1)

        current_layer = l.Layer(weights=utils.generate_one_layer_weights(before_neuron_count,
                                                                         current_neuron_count,
                                                                         bias),
                        activation_functions=utils.generate_layer_activation_functions(current_neuron_count),
                        activation_function_derivatives=utils.generate_layer_activation_function_derivatives(current_neuron_count))

        for neuron in current_layer.get_neurons():
            neuron.set_delta(1)

        current_layer.adjust(1, before_layer)
