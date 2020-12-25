from main import layer as l
import unittest


class TestLayer(unittest.TestCase):

    def test_compute_output(self):
        weights_count = [10, 5]
        neuron_count = [10, 10]

        for wc, nc in zip(weights_count, neuron_count):
            inputs = TestLayer.generate_one_array(wc)
            weights = TestLayer.generate_one_matrix(wc, nc)
            activation_functions = [TestLayer.activation_function] * nc

            layer = l.Layer(weights, activation_functions)

            layer_output = layer.compute_output(inputs)
            expected_output = [wc] * nc

            # print(layer_output)

            self.assertEqual(expected_output, layer_output)

    def generate_one_array(n):
        array = [1] * n
        return array

    def generate_one_matrix(weights_count, neuron_count):
        matrix = [TestLayer.generate_one_array(weights_count)] * neuron_count
        return matrix

    def activation_function(x):
        return max(0, x)

    def generate_layer_activation_functions(neuron_count):
        activation_functions = [TestLayer.activation_function] * neuron_count
        return activation_functions
