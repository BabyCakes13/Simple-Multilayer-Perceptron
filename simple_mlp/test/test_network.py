from simple_mlp import network as n
from . import utils

from functools import reduce
import unittest
import math


class TestNetwork(unittest.TestCase):

    def test_one_forward_propagate(self):
        input = utils.generate_array(2)
        network_layout = [3, 2]
        bias = 1

        network, network_weights = TestNetwork.generate_network(input, network_layout, bias, with_derivatives=False)

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

    def test_forward_propagate(self):
        network_weights = [
                    [
                        [0.763774618976614, 0.13436424411240122, 0.8474337369372327]# one neuron
                    ], # one hidden layer
		            [
                        [0.49543508709194095, 0.2550690257394217],
                        [0.651592972722763, 0.4494910647887381]
                    ]  # one output layer
        ]

        def sigmoid(x):
            return 1 / (1 + math.exp(-x))

        layer_count = len(network_weights)
        activation_functions = [sigmoid] * layer_count

        network = n.Network(network_weights, activation_functions, None)

        fp_input = [1, 0]

        output = network.forward_propagate(fp_input)

        print(output)

    def test_backwards_propagate(self):
        input = utils.generate_array(2)
        network_layout = [3, 2, 2]
        expected_output = [1] * network_layout[-1]
        actual_output = 1.1
        delta = 1
        bias = 2
        learning_rate = 1
        adjustment = 1

        network, network_weights = TestNetwork.generate_network(input, network_layout, bias)

        for layer in network.get_layers():
            for neuron in layer.get_neurons():
                neuron.set_output(actual_output)

        network.display()

        network.backwards_propagation(expected_output)

        # check last layers
        output_layer = network.get_layers()[-1]
        before_sum_deltas = 0
        for neuron, eo in zip(output_layer.get_neurons(), expected_output):
            expected_delta = eo - actual_output
            before_sum_deltas += neuron.get_delta()
            self.assertAlmostEqual(expected_delta, neuron.get_delta())

        print("Check for hidden.")

        hidden_layers = network.get_layers()[:-1]
        for layer in reversed(hidden_layers):
            currrent_sum_deltas = 0
            # print("\n{}".format(layer))
            for neuron in layer.get_neurons():
                # print(neuron.get_delta())
                currrent_sum_deltas += neuron.get_delta()

                print("before sum deltas {} with neuron's delta {}".format(before_sum_deltas, neuron.get_delta()))
                self.assertAlmostEqual(before_sum_deltas, neuron.get_delta())
            before_sum_deltas = currrent_sum_deltas

    def test_backwards_propagate_2(self):
        network_info = [
                    [
                        {'output': 0.7105668883115941, 'weights': [0.763774618976614, 0.13436424411240122, 0.8474337369372327]} # one neuron
                    ], # one hidden layer
		            [
                        {'output': 0.6213859615555266, 'weights': [0.49543508709194095, 0.2550690257394217]},
                        {'output': 0.6573693455986976, 'weights': [0.651592972722763, 0.4494910647887381]}
                    ]  # one output layer
        ]

        network_outputs = [[0.7105668883115941],
                        [0.6213859615555266, 0.6573693455986976]]

        network_weights = [
                    [
                        [0.763774618976614, 0.13436424411240122, 0.8474337369372327]# one neuron
                    ], # one hidden layer
		            [
                        [0.49543508709194095, 0.2550690257394217],
                        [0.651592972722763, 0.4494910647887381]
                    ]  # one output layer
        ]

        expected_outputs = [0, 1]

        def sigmoid(x):
          return 1 / (1 + math.exp(-x))

        def sigmoid_derivative(sigmoud_output):
            return sigmoud_output * (1 - sigmoud_output)

        layer_count = len(network_weights)
        activation_functions = [sigmoid] * layer_count
        activation_functions_derivatives = [sigmoid_derivative] * layer_count

        network = n.Network(network_weights, activation_functions, activation_functions_derivatives)

        for layer, layer_outputs in zip(network.get_layers(), network_outputs):
            for neuron, neuron_output in zip(layer.get_neurons(), layer_outputs):
                neuron.set_output(neuron_output)

        network.backwards_propagation(expected_outputs)

        wanted_deltas = [[-0.0005348048046610517], [-0.14619064683582808, 0.0771723774346327]]

        for layer, layer_deltas in zip(network.get_layers(), wanted_deltas):
            print("\nLayer {}".format(layer))
            for neuron, neuron_delta in zip(layer.get_neurons(), layer_deltas):
                self.assertEqual(neuron.get_delta(), neuron_delta)

    def test_adjust(self):
        input = utils.generate_array(2)
        network_layout = [3, 2]
        output = 1
        delta = 1
        bias = 2
        learning_rate = 1

        network, network_weights = TestNetwork.generate_network(input, network_layout, bias)

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


    def generate_network(input, network_layout, bias, with_derivatives=True):
        network_weights = utils.generate_one_network_weights(input_count=len(input),
                                                             neuron_count_per_layer_list=network_layout,
                                                             bias=bias)
        network_activation_functions = utils.generate_network_activation_functions(network_layout)

        if with_derivatives:
            network_activation_functions_derivatives = utils.generate_network_activation_functions_derivatives(network_layout)
        else:
            network_activation_functions_derivatives = None

        network = n.Network(weights=network_weights,
                            activation_functions=network_activation_functions,
                            activation_functions_derivatives=network_activation_functions_derivatives)

        return network, network_weights
