from simple_mlp import train as t

import unittest
import random


random.seed(1)


class TestTrain(unittest.TestCase):

    def test_generate_random_network_weights(self):
        train = t.Train()

        input_count = 3
        network_layout = [3, 2, 2]
        network_weights = train.generate_random_network_weights(input_count, network_layout)

        print(network_weights)

    def test_setup_network(self):
        train = t.Train()

        input_count = 3
        network_layout = [3, 2, 2]

        activation_functions = [lambda x:x, lambda x:x, lambda x: max(0, x)]
        activation_functions_derivatives = [
            lambda x: 1,
            lambda x: 1,
            lambda x: 0 if x < 0 else ( 1 if x > 0 else raise_(Exception("nooooooooo"))) ]

        network = train.setup_network(input_count, network_layout, activation_functions, activation_functions_derivatives)

        network.display()
