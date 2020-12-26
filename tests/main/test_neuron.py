from main import neuron as n
from . import utils

import unittest
import random

class TestNeuron(unittest.TestCase):

    def test_weights_propagation(self):
        weights = utils.generate_one_array(100)
        neuron = n.Neuron(weights, None)
        neuron_weights = neuron.get_weights()

        self.assertEqual(weights, neuron_weights)

    def test_compute_zero_output(self):
        input = utils.generate_one_and_minus_one_array(100)
        weights = utils.generate_one_array(100)

        neuron = n.Neuron(weights, utils.activation_function)
        output = neuron.compute(input)

        self.assertEqual(0, output)

    def test_compute_size_output(self):
        input = utils.generate_one_array(100)
        weights = utils.generate_one_array(100)

        neuron = n.Neuron(weights, utils.activation_function)
        output = neuron.compute(input)

        self.assertEqual(100, output)
