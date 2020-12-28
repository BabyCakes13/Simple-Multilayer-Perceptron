from simple_mlp import train as t
from simple_mlp import network as n
import math

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

        network = train.setup_network(input_count, network_layout,
                                      activation_functions,
                                      activation_functions_derivatives)

        network.display()

    def test_train(self):
        dataset = {
        (2.7810836,2.550537003):   [0],
        (1.465489372,2.362125076): [0],
        (3.396561688,4.400293529): [0],
        (1.38807019	,1.850220317): [0],
        (3.06407232	,3.005305973): [0],
        (7.627531214,2.759262235): [1],
        (5.332441248,2.088626775): [1],
        (6.922596716,1.77106367):  [1],
        (8.675418651,-0.242068655):[1],
        (7.673756466,3.508563011): [1]
        }

        network_layout = [2, 1]
        epocs = 20
        input_count = 2
        learning_rate = 0.5
        layer_count = len(network_layout)

        def sigmoid(x):
          return 1 / (1 + math.exp(-x))

        def sigmoid_derivative(sigmoud_output):
            return sigmoud_output * (1 - sigmoud_output)
        activation_functions = [sigmoid] * layer_count
        activation_functions_derivatives = [sigmoid_derivative] * layer_count

        train = t.Train()

        network = train.setup_network(input_count, network_layout,
                                      activation_functions,
                                      activation_functions_derivatives)

        train.set_dataset(dataset)
        train.display_network()
        train.chu_chu(learning_rate, 0.1, epocs)
