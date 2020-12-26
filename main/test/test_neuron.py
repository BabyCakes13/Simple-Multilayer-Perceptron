from main import neuron as n
from . import utils

import unittest
import random

class TestNeuron(unittest.TestCase):

    def test_weights_propagation(self):
        weights = utils.generate_one_neuron_weights_with_bias(100, bias=1)
        neuron = n.Neuron(weights, None)
        neuron_weights = neuron.get_weights()

        # since w[0] is going to be the bias in the neuron, we need to ditch one element from our list as well.
        weights = weights[1:]

        self.assertEqual(weights, neuron_weights)

    def test_compute_zero_output(self):
        bias = 1
        input = utils.generate_one_and_minus_one_array(100)
        weights = utils.generate_one_neuron_weights_with_bias(100, bias)

        neuron = n.Neuron(weights, utils.activation_function)
        output = neuron.compute(input)

        self.assertEqual(0 + bias, output)

    def test_compute_size_output(self):
        bias = 1
        input = utils.generate_array(100)
        weights = utils.generate_one_neuron_weights_with_bias(100, bias)

        neuron = n.Neuron(weights, utils.activation_function)
        output = neuron.compute(input)

        self.assertEqual(100 + bias, output)

    def test_different_input_weights_count(self):
        input = utils.generate_one_and_minus_one_array(50)
        weights = utils.generate_one_neuron_weights_with_bias(100, bias=1)

        neuron = n.Neuron(weights, utils.activation_function)

        self.assertRaises(Exception, neuron.compute, args=[input])

    def test_compute_random_float_inputs(self):
        bias = 1
        input = utils.generate_array(100, lambda y: random.random() * 2 - 1)
        weights = utils.generate_one_neuron_weights_with_bias(100, bias)

        print(input)

        neuron = n.Neuron(weights, utils.activation_function)
        output = neuron.compute(input)

        # added huge delta because we want to test if the algorithm works with floats,
        # not how uniform the distribution is (in which case we'd want the error to be
        # as little as possible) 
        self.assertAlmostEqual(0, output, delta=50)
