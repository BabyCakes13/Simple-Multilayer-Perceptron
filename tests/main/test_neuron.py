from main import neuron as n

import unittest
import random

class TestNeuron(unittest.TestCase):

    def test_weights_propagation(self):
        weights = TestNeuron.generate_one_array(100)
        neuron = n.Neuron(weights, None)
        neuron_weights = neuron.get_weights()

        self.assertEqual(weights, neuron_weights)

    def test_compute_zero_output(self):
        input = TestNeuron.generate_one_and_minus_one_array(100)
        weights = TestNeuron.generate_one_array(100)

        neuron = n.Neuron(weights, TestNeuron.activation_function)
        output = neuron.compute(input)

        # print(weights)
        # print(input)

        self.assertEqual(0, output)

    def test_compute_size_output(self):
        input = TestNeuron.generate_one_array(100)
        weights = TestNeuron.generate_one_array(100)

        neuron = n.Neuron(weights, TestNeuron.activation_function)
        output = neuron.compute(input)

        # print(weights)
        # print(input)

        self.assertEqual(100, output)

    def activation_function(x):
        return max(0, x)

    def generate_one_and_minus_one_array(size):
        array = [1, -1] * (size // 2)
        return array

    def generate_one_array(size):
        array = [1] * size
        return array
